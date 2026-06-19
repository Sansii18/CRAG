from datetime import datetime, timedelta
from src.utils.logger import logger
import json 
import os 

RATE_LIMIT_FILE = ".tavily_usage.json"

class TavilyRateLimiter:
    def __init__(self, max_searches_per_day: int = 33):
        self.max_searches_per_day = max_searches_per_day
        self._load()
    
    def _load(self):
        """Load counter from disk."""
        if os.path.exists(RATE_LIMIT_FILE):
            try:
                with open(RATE_LIMIT_FILE, "r") as f:
                    data = json.load(f)
                self.searches_today = data.get("searches_today", 0)
                self.last_reset = datetime.fromisoformat(
                    data.get("last_reset", str(datetime.now().date()))
                ).date()

            except Exception:
                self.searches_today = 0
                self.last_reset = datetime.now().date()

        else:
            self.searches_today = 0
            self.last_reset = datetime.now().date()

    def _save(self):
        """Persist counter to disk."""
        with open(RATE_LIMIT_FILE, "w") as f:
            json.dump({
                "searches_today": self.searches_today,
                "last_reset": str(self.last_reset)
            }, f)

    def reset(self):
        """Reset counter if a new day has started."""
        today = datetime.now().date()
        if today > self.last_reset:
            logger.info("Resetting Tavily rate limiter for a new day.")
            self.searches_today = 0
            self.last_reset = today
            self._save()
            logger.info(f"Rate limiter reset for new day: {today}")

    def can_search(self) -> bool:
        self.reset()
        if self.searches_today >= self.max_searches_per_day:
            logger.warning(
                f"Tavily rate limit reached: "
                f"{self.searches_today}/{self.max_searches_per_day} today"
            )
            return False
        return True
    
    def increment(self):
        self.searches_today += 1
        self._save()                 
        logger.info(
            f"Tavily usage: "
            f"{self.searches_today}/{self.max_searches_per_day} today"
        )

    def get_status(self) -> str:
        self.reset()
        return (
            f"Tavily: {self.searches_today}/"
            f"{self.max_searches_per_day} searches used today"
        )
    