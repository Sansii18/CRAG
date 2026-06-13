from typing import Literal 
from src.utils.logger import logger

class ConditionalRouer:
    def __init__(self, config):
        self.domain_thresholds = {
            "general" : {
                "high" : 0.7, 
                "medium" : 0.5,
            },
            "legal" : {
                "high" : 0.9, 
                "medium" : 0.7,
            },
            "technical" : {
                "high" : 0.75, 
                "medium" : 0.5,
            },
            "financial" : {
                "high" : 0.8, 
                "medium" : 0.55,
            },
        }

    def route(self,confidence: float,query: str, domain: str = "general") -> Literal["GENERATE", "WEB_SEARCH", "REFUSE"]:
        """
        Route to next step based on confidence score.
        HIGH  → GENERATE   (use retrieved docs directly)
        MID   → WEB_SEARCH (supplement with web results)
        LOW   → REFUSE     (docs too weak to answer safely)
        """

        thresholds = self.domain_thresholds.get(domain, self.domain_thresholds["general"])

        if confidence >= thresholds["high"]:
            action = "GENERATE"
        elif confidence >= thresholds["medium"]:
            action = "WEB_SEARCH"
        else:
            action = "REFUSE"

        logger.info(
            f"Route | domain={domain} | "
            f"confidence={confidence:.3f} | action={action}"
        )

        return action