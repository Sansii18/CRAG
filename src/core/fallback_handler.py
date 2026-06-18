# This file is the decision + coordination layer. It answers "should I search the web?" and if yes, runs the full fallback sequence in one call.

from src.Integrations.tavily_search import TavilySearchFallBack
from src.utils.logger import logger
from src.utils.tavily_rate_limiter import TavilyRateLimiter
from src.core.source_combiner import SourceCombiner

class FallbackHandler:
    def __init__(self):
        self.rate_limiter = TavilyRateLimiter()
        self.combiner = SourceCombiner()
        self.tavily = TavilySearchFallBack()

        self.domain_thresholds = {
            "legal":     0.85,
            "financial": 0.80,
            "academic":  0.80,
            "technical": 0.75,
            "general":   0.70
        }

    def should_use_fallback(
            self, 
            confidence: float,
            query:str,
            domain:str = "general"
    ) -> bool:
        # Check rate limit first
        if not self.rate_limiter.can_search():
            logger.info("Skipping web search: Tavily rate limit reached")
            return False
            
        time_keywords = [
            "latest", "2024", "2025", "2026",
            "recent", "today", "current", "now"
        ]

        if any(kw in query.lower() for kw in time_keywords):
            logger.info("Triggering fallback: time-sensitive query")
            return True
            
        threshold = self.domain_thresholds.get(domain, 0.70)
        if confidence < threshold:
            logger.info(
                f"Triggering fallback: {domain} domain, "
                f"confidence={confidence:.2f} < threshold={threshold}"
            )
            return True

        return False
    
    def handle_fallback(
        self,
        query: str,
        local_docs: list,
        domain: str = "general"          
    ) -> list:
        logger.info(f"Executing fallback for: '{query}' (domain={domain})")

        tavily_results = self.tavily.search(query, domain)
        valid = [r for r in tavily_results if "error" not in r]

        if valid:
            self.rate_limiter.increment()
        else:
            logger.warning("Tavily returned no valid results , not incrementing counter")
        
        combined = self.combiner.combine_Sources(
            local_docs,
            tavily_results,
            domain
        )

        if not combined:
            logger.warning("No combined sources — falling back to local docs only")
            return local_docs

        logger.info(f"Fallback complete: {len(combined)} combined sources")
        return combined