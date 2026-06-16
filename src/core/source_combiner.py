# After Tavily returns web results and Qdrant has local results, the system has two separate lists of documents with different formats and credibility levels. This file is the merge and rank layer that produces one unified, prioritised list ready for the answer generator.

from typing import List, Dict
from src.utils.logger import logger

class SourceCombiner:
    def __init__(self):
        pass 

    def combine_Sources(self, local_docs: List[Dict], tavily_search_results : List[Dict], domain : str = "general") -> List[Dict]:
        """
        Combine local retrieval + Tavily web results into one ranked list.
        Local docs always take priority over web results.
        """

        # Filtering out error results before combining
        valid_tavily = [
            r for r in tavily_search_results
            if "error" not in r and r.get("content", "").strip()
        ]

        # LOCAL DOCUMENTS -> HIGHEST PRIORITY
        all_sources = []
        for doc in local_docs :
            all_sources.append({
                "text": doc.get("text", ""),
                "source_type": "local_db",
                "priority": 1.0,
                "score": doc.get("score", 0.7),
                "url": None
            })

        # TAVILY WEB RESULTS -> SECOND PRIORITY
        for result in valid_tavily:
            all_sources.append({
                "text": result.get("content", ""),
                "source_type": "tavily_web",
                "priority": result.get("final_credibility_score", 0.5),
                "score": result.get("final_credibility_score", 0.5),
                "url": result.get("url", ""),
                "title": result.get("title", "")
            })

        # SORT : LOCAL FIRST, THEN BY HIGHEST SCORE
        ranked = sorted(
            all_sources,
            key=lambda x: (
                0 if x["source_type"] == "local_db" else 1,
                -x["priority"]
            )
        )

        logger.info(
            f"Combined {len(ranked)} sources "
            f"(local: {len(local_docs)}, web: {len(valid_tavily)})"
        )
        return ranked[:5]
