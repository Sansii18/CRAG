# This file is a quality gate, it verifies whether retrieval is working before building Phase 3 on top of it. A broken retrieval system discovered in Phase 5 wastes days of debugging. Discovering it here takes 10 seconds.
import sys 
sys.path.insert(0, '.')

import pytest
from src.core.VectorEmbeddings import GenerateEmbeddings
from src.Integrations.Qdrant_client import StoreQdrant

# Generic queries ~ will work for any document domain
GENERIC_TEST_QUERIES = [
    "What is the main topic of the uploaded documents?",
    "Summarize the key concepts in the documents.",
    "What are the important details mentioned?"
]

def test_retrieval_quality():
    """Test retrieval relevance with known queries"""
    embedder = GenerateEmbeddings()
    qdrant = StoreQdrant()

    for query in GENERIC_TEST_QUERIES:
        query_embeddings = embedder.embed_query(query)
        results = qdrant.search(query_embeddings, top_k=5)

        assert len(results) > 0, f"No results retrieved for query: {query}"

        # Phase 2 (Data Ingestion) threshold: 0.3
        # Phase 5 (Production) threshold: 0.5+
        # Scores range from 0-1: 0.3 = 30% match, 0.5 = 50% match
        assert results[0]["score"] > 0.3, f"Critical: Very low score ({results[0]['score']:.3f}) for: {query}"
        
        if results[0]["score"] < 0.5:
            print(f"WARNING: Score is moderate (0.3-0.5). Documents may need refinement.")
        # Why result[0] only? Because result[0] is the most relevant result returned by Qdrant

        print(f"Query: {query}")
        print(f"Top result: {results[0]['text'][:100]}...")
        print(f"Score: {results[0]['score']:.3f} (Threshold: 0.3 for Phase 2)\n")
