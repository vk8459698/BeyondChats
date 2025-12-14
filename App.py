"""
LLM Response Evaluation Pipeline
Evaluates AI chatbot responses for relevance, hallucination, and performance metrics
"""

import json
import time
from typing import Dict, List, Tuple, Any
from dataclasses import dataclass
from datetime import datetime
import re
from collections import Counter
import hashlib


@dataclass
class EvaluationResult:
    """Stores evaluation results for a single response"""
    relevance_score: float
    completeness_score: float
    hallucination_score: float
    factual_accuracy_score: float
    latency_ms: float
    estimated_cost: float
    details: Dict[str, Any]


class LLMResponseEvaluator:
    """
    Main evaluator class for LLM responses.
    Uses lightweight, efficient algorithms suitable for real-time evaluation at scale.
    """
    
    def __init__(self, cost_per_1k_tokens: float = 0.002):
        """
        Initialize evaluator with cost parameters
        
        Args:
            cost_per_1k_tokens: Estimated cost per 1000 tokens for evaluation
        """
        self.cost_per_1k_tokens = cost_per_1k_tokens
        self.cache = {}  # Simple cache for repeated evaluations
        
    def evaluate_response(
        self, 
        conversation: Dict, 
        context_vectors: Dict,
        target_turn: int = None
    ) -> EvaluationResult:
        """
        Main evaluation method that orchestrates all evaluation checks
        
        Args:
            conversation: Full conversation JSON
            context_vectors: Context vectors JSON with source data
            target_turn: Specific turn to evaluate (defaults to last AI response)
            
        Returns:
            EvaluationResult object with all scores and details
        """
        start_time = time.time()
        
        # Extract relevant data
        turns = conversation['conversation_turns']
        if target_turn is None:
            # Find last AI response
            target_turn = max([t['turn'] for t in turns if t['role'] == 'AI/Chatbot'])
        
        ai_response = next(t for t in turns if t['turn'] == target_turn)
        user_message = next(t for t in turns if t['turn'] == target_turn - 1)
        
        # Get context data
        vector_data = context_vectors['data']['vector_data']
        sources_info = context_vectors['data']['sources']
        
        # Run evaluations
        relevance, completeness = self._evaluate_relevance_completeness(
            user_message['message'], 
            ai_response['message'],
            vector_data
        )
        
        hallucination, factual_accuracy = self._evaluate_hallucination_accuracy(
            ai_response['message'],
            vector_data,
            sources_info
        )
        
        # Calculate latency and cost
        latency_ms = (time.time() - start_time) * 1000
        estimated_cost = self._estimate_cost(
            user_message['message'],
            ai_response['message'],
            vector_data
        )
        
        # Compile detailed results
        details = {
            'turn_number': target_turn,
            'user_query': user_message['message'],
            'ai_response': ai_response['message'],
            'evaluation_timestamp': datetime.now().isoformat(),
            'context_sources_used': len(sources_info.get('vectors_used', [])),
            'total_context_available': len(vector_data)
        }
        
        return EvaluationResult(
            relevance_score=relevance,
            completeness_score=completeness,
            hallucination_score=hallucination,
            factual_accuracy_score=factual_accuracy,
            latency_ms=latency_ms,
            estimated_cost=estimated_cost,
            details=details
        )
    
    def _evaluate_relevance_completeness(
        self, 
        user_query: str, 
        ai_response: str,
        context_vectors: List[Dict]
    ) -> Tuple[float, float]:
        """
        Evaluate how relevant and complete the response is
        
        Uses lightweight keyword matching and coverage analysis
        """
        # Extract key terms from user query (simple but effective)
        query_terms = set(self._extract_key_terms(user_query.lower()))
        response_terms = set(self._extract_key_terms(ai_response.lower()))
        
        # Relevance: How many query terms are addressed
        if not query_terms:
            relevance = 1.0
        else:
            addressed_terms = query_terms.intersection(response_terms)
            relevance = len(addressed_terms) / len(query_terms)
        
        # Completeness: Check if response addresses the query type
        completeness = self._assess_completeness(user_query, ai_response, context_vectors)
        
        return round(relevance, 3), round(completeness, 3)
    
    def _assess_completeness(
        self, 
        user_query: str, 
        ai_response: str,
        context_vectors: List[Dict]
    ) -> float:
        """
        Assess if the response adequately answers the query
        """
        query_lower = user_query.lower()
        response_lower = ai_response.lower()
        
        # Check for common completeness indicators
        completeness_score = 0.5  # Base score
        
        # Question answered indicators
        if any(q in query_lower for q in ['how much', 'what is the cost', 'price']):
            # Expect numbers in response
            if re.search(r'\d+', ai_response):
                completeness_score += 0.3
            # Expect currency symbols
            if any(c in ai_response for c in ['₹', 'Rs', '$', 'USD']):
                completeness_score += 0.2
        
        elif any(q in query_lower for q in ['where', 'address', 'location']):
            # Expect address components
            if any(term in response_lower for term in ['road', 'street', 'mumbai', 'colaba']):
                completeness_score += 0.3
            if re.search(r'\d{6}', ai_response):  # Pin code
                completeness_score += 0.2
        
        elif any(q in query_lower for q in ['when', 'time', 'schedule']):
            # Expect time-related information
            if re.search(r'\d{1,2}:\d{2}|morning|afternoon|evening', response_lower):
                completeness_score += 0.3
        
        # Check if response provides actionable information
        if any(action in response_lower for action in ['you can', 'please', 'contact', 'book', 'visit']):
            completeness_score += 0.1
        
        # Check if response includes links/references (helpful completeness)
        if 'http' in ai_response or '[' in ai_response:
            completeness_score += 0.1
        
        return min(1.0, completeness_score)
    
    def _evaluate_hallucination_accuracy(
        self,
        ai_response: str,
        context_vectors: List[Dict],
        sources_info: Dict
    ) -> Tuple[float, float]:
        """
        Detect hallucinations and verify factual accuracy
        
        Key approach: Check if claims in response are supported by context
        """
        # Extract factual claims from response (numbers, specific details)
        response_claims = self._extract_factual_claims(ai_response)
        
        if not response_claims:
            return 1.0, 1.0  # No claims = no hallucination
        
        # Build context text from vectors
        context_text = " ".join(
            v["text"].lower()
            for v in context_vectors
            if isinstance(v, dict) and "text" in v and v["text"]
        )
        
        # Check each claim against context
        supported_claims = 0
        hallucinated_claims = 0
        
        for claim in response_claims:
            if self._is_claim_supported(claim, context_text):
                supported_claims += 1
            else:
                hallucinated_claims += 1
        
        # Calculate scores
        total_claims = len(response_claims)
        hallucination_score = 1 - (hallucinated_claims / total_claims)
        factual_accuracy = supported_claims / total_claims
        
        return round(hallucination_score, 3), round(factual_accuracy, 3)
    
    def _extract_factual_claims(self, text: str) -> List[str]:
        """
        Extract verifiable factual claims from text
        Focus on: numbers, prices, addresses, names, specific details
        """
        claims = []
        
        # Extract numerical claims (prices, costs, numbers)
        numerical_patterns = [
            r'(?:rs|₹|inr)\s*\.?\s*\d+[,\d]*',  # Prices in rupees
            r'\d+[,\d]*\s*(?:rupees|rs)',
            r'us\s*\$\s*\d+[,\d]*',  # USD prices
            r'\d+[,\d]*\s*(?:per night|per day|per cycle)',
            r'\d+\s*(?:minutes|hours|days|weeks)',
        ]
        
        for pattern in numerical_patterns:
            matches = re.finditer(pattern, text.lower())
            claims.extend([m.group() for m in matches])
        
        # Extract location claims
        location_patterns = [
            r'\d{6}',  # Pin codes
            r'(?:mumbai|colaba|bandra|andheri)\s*\d{6}',
            r'\d+[a-z]?\s*,?\s*[a-z\s]+(?:road|street|avenue)',
        ]
        
        for pattern in location_patterns:
            matches = re.finditer(pattern, text.lower())
            claims.extend([m.group() for m in matches])
        
        # Extract specific facility/service claims
        if 'room' in text.lower():
            room_claims = re.findall(r'(?:ac|air-conditioned|non-ac)\s+room', text.lower())
            claims.extend(room_claims)
        
        return list(set(claims))  # Remove duplicates
    
    def _is_claim_supported(self, claim: str, context: str) -> bool:
        """
        Check if a specific claim is supported by context
        Uses fuzzy matching for robustness
        """
        claim_lower = claim.lower().strip()
        context_lower = context.lower()
        
        # Direct match
        if claim_lower in context_lower:
            return True
        
        # Extract key numbers from claim
        claim_numbers = re.findall(r'\d+', claim_lower)
        
        if claim_numbers:
            # Check if the numbers appear in similar context
            for num in claim_numbers:
                # Look for number in context with some surrounding words
                if num in context_lower:
                    # Basic proximity check
                    idx = context_lower.find(num)
                    surrounding = context_lower[max(0, idx-50):idx+50]
                    
                    # Check if surrounding context is similar
                    claim_words = set(claim_lower.split())
                    surrounding_words = set(surrounding.split())
                    
                    overlap = claim_words.intersection(surrounding_words)
                    if len(overlap) >= 2:  # At least 2 words match
                        return True
        
        return False
    
    def _extract_key_terms(self, text: str) -> List[str]:
        """
        Extract meaningful terms from text (remove stop words)
        """
        # Simple stop words list
        stop_words = {
            'the', 'is', 'at', 'which', 'on', 'a', 'an', 'and', 'or', 'but',
            'in', 'with', 'to', 'for', 'of', 'as', 'by', 'from', 'can', 'you',
            'we', 'i', 'my', 'your', 'do', 'does', 'what', 'how', 'where', 'when'
        }
        
        # Tokenize and filter
        words = re.findall(r'\b[a-z]{3,}\b', text.lower())
        return [w for w in words if w not in stop_words]
    
    def _estimate_cost(
        self,
        user_query: str,
        ai_response: str,
        context_vectors: List[Dict]
    ) -> float:
        """
        Estimate the cost of this evaluation
        Based on token counts (approximated by character count)
        """
        # Rough token estimation: ~4 characters per token
        total_chars = len(user_query) + len(ai_response)
        
        # Add context size (vectors actually used)
        for vector in context_vectors[:5]:  # Assume top 5 used
            total_chars += len(vector['text'])
        
        estimated_tokens = total_chars / 4
        cost = (estimated_tokens / 1000) * self.cost_per_1k_tokens
        
        return round(cost, 6)
    
    def batch_evaluate(
        self,
        conversation: Dict,
        context_vectors: Dict,
        ai_turns: List[int] = None
    ) -> List[EvaluationResult]:
        """
        Evaluate multiple turns in a conversation
        Optimized for batch processing
        """
        results = []

        turns = conversation["conversation_turns"]

        if ai_turns is None:
            ai_turns = [
                t["turn"]
                for t in turns
                if t["role"] == "AI/Chatbot"
            ]

        for turn in ai_turns:
            # Skip if there is no previous turn
            if not any(t.get("turn") == turn - 1 for t in turns):
                continue

            result = self.evaluate_response(conversation, context_vectors, turn)

            if result is not None:
                results.append(result)

        return results



def generate_report(results: List[EvaluationResult]) -> Dict:
    """
    Generate summary report from evaluation results
    """
    if not results:
        return {}
    
    return {
        'total_evaluations': len(results),
        'average_relevance': round(sum(r.relevance_score for r in results) / len(results), 3),
        'average_completeness': round(sum(r.completeness_score for r in results) / len(results), 3),
        'average_hallucination_score': round(sum(r.hallucination_score for r in results) / len(results), 3),
        'average_factual_accuracy': round(sum(r.factual_accuracy_score for r in results) / len(results), 3),
        'average_latency_ms': round(sum(r.latency_ms for r in results) / len(results), 2),
        'total_estimated_cost': round(sum(r.estimated_cost for r in results), 6),
        'min_scores': {
            'relevance': min(r.relevance_score for r in results),
            'completeness': min(r.completeness_score for r in results),
            'hallucination': min(r.hallucination_score for r in results),
            'factual_accuracy': min(r.factual_accuracy_score for r in results)
        },
        'flagged_responses': [
            {
                'turn': r.details['turn_number'],
                'issue': 'low_scores',
                'scores': {
                    'relevance': r.relevance_score,
                    'completeness': r.completeness_score,
                    'hallucination': r.hallucination_score,
                    'accuracy': r.factual_accuracy_score
                }
            }
            for r in results
            if r.hallucination_score < 0.7 or r.factual_accuracy_score < 0.7
        ]
    }


# Example usage
if __name__ == "__main__":
    # Load sample data
    with open('sample-chat-conversation-01.json', 'r') as f:
        conversation = json.load(f)
    
    with open('sample_context_vectors-01.json', 'r') as f:
        context_vectors = json.load(f)
    
    # Initialize evaluator
    evaluator = LLMResponseEvaluator()
    
    # Evaluate specific turn (turn 14 has known hallucination)
    result = evaluator.evaluate_response(conversation, context_vectors, target_turn=14)
    
    print("=== Evaluation Results ===")
    print(f"Turn: {result.details['turn_number']}")
    print(f"Relevance Score: {result.relevance_score}")
    print(f"Completeness Score: {result.completeness_score}")
    print(f"Hallucination Score: {result.hallucination_score}")
    print(f"Factual Accuracy: {result.factual_accuracy_score}")
    print(f"Latency: {result.latency_ms:.2f}ms")
    print(f"Estimated Cost: ${result.estimated_cost:.6f}")
    
    # Batch evaluate all AI responses
    print("\n=== Batch Evaluation ===")
    all_results = evaluator.batch_evaluate(conversation, context_vectors)
    report = generate_report(all_results)
    
    print(json.dumps(report, indent=2))
