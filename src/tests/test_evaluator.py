import sys 
sys.path.insert(0, '.')

import pytest 
from src.core.evaluator import RetrievalEvaluator
from src.core.router import ConditionalRouer

def test_evaluator_scoring():
    """Test whether evaluator produces valid confidence scores"""

    evaluator = RetrievalEvaluator(domain='general')

    query = "What are the key components explained in the documents?"

    # Creating fake relevant documents for testing, also we are assinging high scores to simulate relevant documents
    relevant_docs = [
        {
            "text" : "The document explains core principles and methodologies "
            "used in the field, including structured approaches to "
            "problem solving and analysis.",
            "score": 0.9
        },
        {
            "text": "Key concepts include systematic evaluation, evidence-based "
            "reasoning, and iterative refinement of solutions.",
            "score": 0.85
        }
    ]

    confidence, details = evaluator.evaluate(query, relevant_docs)

    assert 0.0 <= confidence <= 1.0, "Confidence score should be between 0.0 and 1.0"

    assert "confidence_score" in details, "Details should contain 'confidence_score' key"

    print(f"Confidence for relevant docs: {confidence:.3f}")
    print(f"Reasoning: {details.get('reasoning', 'N/A')}")

def test_router_routing():
    """Test conditional routing decisions"""

    router = ConditionalRouer({})

    test_cases = [
        (0.9,  "general",   "GENERATE"),
        (0.6,  "general",   "WEB_SEARCH"),
        (0.3,  "general",   "REFUSE"),
        (0.95, "legal",     "GENERATE"),
        (0.75, "legal",     "WEB_SEARCH"),
        (0.80, "technical", "GENERATE"),
        (0.60, "technical", "WEB_SEARCH"),
    ]

    for confidence, domain, expected in test_cases:
        action = router.route(confidence, "test query", domain)
        assert action == expected, \
            f"domain={domain} confidence={confidence} → expected {expected}, got {action}"
        print(f"{domain}: {confidence:.2f} → {action}")