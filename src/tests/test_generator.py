# Tests the "AnswerGenerator" across all three confidence paths to confirm the correct template, structure, and content is returned. Catches bugs where wrong templates are selected or the LLM response is not properly formatted.

import sys 
sys.path.insert(0, '.')

import pytest
from src.core.AnswerGenerator import AnswerGenerator

@pytest.fixture(scope="module")
def generator():
    return AnswerGenerator(domain="general")

@pytest.mark.integration
def test_high_confidence_path(generator):
    """Test answer generation for high confidence scenario ~ Calls NVIDIA API"""

    result = generator.generate(
        query="What are the key findings in the documents?",         
        context="The documents outline key principles and methodologies "
        "used in structured problem solving and analysis.",
        confidence=0.9,
        sources=[{"title": "Research Paper", "source_type": "local_kb"}]
    )

    assert result["confidence"] == 0.9
    assert result["confidence_level"] == "high_confidence"
    assert "Sources:" in result["answer"]
    assert "answer" in result
    print(f"High confidence generation passed")
    print(f"Answer preview: {result['answer'][:100]}...")

@pytest.mark.integration
def test_medium_confidence_path(generator):
    """Test answer generation for medium confidence scenario ~ Calls NVIDIA API"""

    result = generator.generate(
        query="What is the latest development in this topic?",
        context="Recent developments include several new approaches"
        "to the problem, with promising early results.",
        confidence=0.65,
        sources=[
            {"title": "Local Document", "source_type": "local_kb"},
            {"title": "Web Result",     "source_type": "tavily_web"}
        ]
    )

    assert result["confidence_level"] == "medium_confidence"
    assert "Note:" in result["answer"]
    assert "answer" in result
    print("Medium confidence generation passed")

@pytest.mark.integration
def test_low_confidence_path(generator):
    """Test answer generation for low confidence scenario ~ NO LLM CALL, just template response"""

    result = generator.generate(
        query="Explain an extremely niche concept not covered anywhere",
        context="",
        confidence=0.2,
        sources=[]
    )

    assert result["confidence_level"] == "low_confidence"
    assert result["confidence"] == 0.2
    assert "cannot reliably" in result["answer"]
    assert result["sources"] == []
    print("Low confidence refusal passed")

def test_generator_returns_correct_structure(generator):
    """Test return dict always has required keys ~ NO API call"""

    result = generator.generate(
        query="test query",
        context="test context",
        confidence=0.2,
        sources=[]
    )

    required_keys = ["answer", "confidence", "confidence_level", "sources"]
    for key in required_keys:
        assert key in result, f"Missing key: {key}"

    print("✅ Response structure test passed")