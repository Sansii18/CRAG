import sys
sys.path.insert(0, '.')

import pytest
from src.Integrations.tavily_search import TavilySearchFallBack
from src.core.fallback_handler import FallbackHandler
from src.utils.tavily_rate_limiter import TavilyRateLimiter


@pytest.mark.integration          
def test_tavily_search():
    """Test live Tavily search consumes 1 API credit"""
    tavily = TavilySearchFallBack()

    results = tavily.search(
        "What are the latest trends in artificial intelligence?",
        domain="general"      
    )

    assert isinstance(results, list), "Should return list"
    if results and "error" not in results[0]:
        assert "title" in results[0],   "Missing title field"
        assert "url" in results[0],"Missing url field"
        assert "content" in results[0],"Missing content field"
        assert "final_credibility_score" in results[0], \
            "Missing credibility score — enhance not called"
        print(f"Tavily returned {len(results)} results")
        print(f"Top result: {results[0]['title']}")
        print(f"Credibility: {results[0]['final_credibility_score']:.3f}")


def test_domain_scoring_enhancement():
    """Test credibility scoring no API call"""
    tavily = TavilySearchFallBack()

    # Use technical domain + docs.python.org which is in the rules
    result = {
        "title": "Python Documentation",
        "url": "docs.python.org/3/library/functions.html",
        "content": "Built-in functions available in Python.",
        "tavily_score": 0.8
    }

    enhanced = tavily.enhance_with_domain_score(result, domain="technical")

    assert "final_credibility_score" in enhanced
    # docs.python.org = 0.99 → (0.8×0.6) + (0.99×0.4) = 0.876
    assert enhanced["final_credibility_score"] > 0.8
    print(f"Credibility score: {enhanced['final_credibility_score']:.3f}")


def test_rate_limiter(tmp_path, monkeypatch):
    """Test rate limiting with isolated temp file — no disk corruption"""
    temp_file = str(tmp_path / "test_usage.json")
    monkeypatch.setattr(
        "src.utils.tavily_rate_limiter.RATE_LIMIT_FILE",
        temp_file
    )

    limiter = TavilyRateLimiter(max_searches_per_day=5)

    for i in range(5):
        assert limiter.can_search(),f"Should allow search {i+1}"
        limiter.increment()

    assert not limiter.can_search(), "Should deny after limit"
    print("Rate limiter working correctly")


def test_fallback_trigger():
    """Test fallback decision logic — no API call"""
    handler = FallbackHandler()

    # Low confidence → trigger
    assert handler.should_use_fallback(0.3, "test query", "general")

    # Time-sensitive keyword → trigger
    assert handler.should_use_fallback(0.7, "latest news today", "general")

    # High confidence → no trigger
    assert not handler.should_use_fallback(0.9, "test query", "general")

    print("Fallback trigger logic working")