# When the router returns "WEB_SEARCH",the retrieved Qdrant chunks weren't good enough but the question is still answerable. Rather than refusing or hallucinating, the system falls back to live web search via Tavily. This module handles the integration with Tavily's search API, allowing us to fetch relevant information from the web in a structured format that our LLM can easily process.

from tavily import TavilyClient
from typing import Dict, List
from src.utils.config import get_config
from src.utils.logger import logger

class TavilySearchFallBack:
    def __init__(self):
        config = get_config("tavily")

        self.client = TavilyClient(api_key=config.api_key)
        self.search_depth = config.search_depth
        self.max_results = config.max_results

        # For a particular domain, search only within these trusted websites.
        self.domain_whitelists = {
            "general" : [], # No restrictions for general domain
            "legal" : [
                "supremecourt.gov",
                "justice.gov",
                "law.cornell.edu",
                "justia.com",
                "aba.org",
                "nolo.com"
            ],
            "technical" : [
                "github.com",
                "stackoverflow.com",
                "python.org",        
                "docs.python.org",
                "nodejs.org",
                "mozilla.org"
            ],
            "financial" : [
                "sec.gov",
                "reuters.com",
                "ft.com",
                "bloomberg.com",
                "investopedia.com"
            ],
            "academic" : [
                "arxiv.org",
                "semanticscholar.org",
                "pubmed.ncbi.nlm.nih.gov",
                "scholar.google.com"
            ]
        }

        logger.info("Tavily search initialized..... Lesssgoooo!!")

    def search(self, query : str, domain : str = "general") -> List[Dict]:
        """Search using Tavily with domain-aware filtering."""
        try: 
            trusted_domains = self.domain_whitelists.get(domain, [])
            logger.info(f"Tavily search: '{query}' (domain={domain})")

            search_params = {
                "query": query,
                "search_depth": self.search_depth,
                "max_results": self.max_results,
            }
            if trusted_domains:
                search_params["include_domains"] = trusted_domains

            response = self.client.search(**search_params)

            formatted_results = []
            for result in response.get("results", []):
                formatted_results.append({
                    "title": result.get("title", ""),
                    "url": result.get("url", ""),
                    "content": result.get("content", ""),
                    "raw_content": result.get("raw_content", ""),
                    "tavily_score": result.get("score", 0.5),
                    "source": "tavily_web"
                })

            formatted_results = [
                self.enhance_with_domain_score(result, domain) for result in formatted_results
            ]

            logger.info(f"Found {len(formatted_results)} results from Tavily")
            return formatted_results
        
        except Exception as e:
            logger.error(f"Tavily search failed: {e}")
            # Returning error flag instead of silent empty list
            return [{"error": str(e), "content": "", "source": "tavily_error"}]
    
    # After Tavily returns results, assign a trust score to the source website.
    def enhance_with_domain_score(self, result: Dict, domain: str) -> Dict:
        """Combine Tavily score with domain-specific URL credibility."""

        domain_rules = {
            "general": {
                "wikipedia.org": 0.75,
                "reddit.com": 0.40,
                "medium.com": 0.55,
                "gov": 0.90,
                "edu": 0.85,
            },
            "legal": {
                "supremecourt.gov": 1.0,
                "justice.gov": 0.98,
                "law.cornell.edu": 0.95,
                "justia.com": 0.90,
                "nolo.com": 0.85,
                "medium.com": 0.40
            },
            "technical": {
                "github.com": 0.98,
                "stackoverflow.com": 0.95,
                "docs.python.org": 0.99,
                "python.org": 0.99,   # ← fixed
                "nodejs.org": 0.99,
                "mozilla.org": 0.95,
                "medium.com": 0.60,
                "dev.to": 0.70
            },
            "financial": {
                "sec.gov": 1.0,
                "reuters.com": 0.92,
                "ft.com": 0.90,
                "bloomberg.com": 0.90,
                "investopedia.com": 0.80,
                "reddit.com": 0.25
            },
            "academic": {
                "arxiv.org": 0.95,
                "pubmed.ncbi.nlm.nih.gov": 0.99,
                "semanticscholar.org": 0.92,
                "wikipedia.org": 0.65,
                "reddit.com": 0.20
            }
        }

        url = result.get("url", "")
        tavily_score = result.get("tavily_score", 0.5)

        rules = domain_rules.get(domain, domain_rules["general"])
        domain_score = 0.5  # Default score if no specific rule matches
        for pattern, score in rules.items():
            if pattern in url:
                domain_score = score
                break
        
        result["final_credibility_score"] = (tavily_score * 0.6) + (domain_score * 0.4)
        return result