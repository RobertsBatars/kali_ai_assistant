# tools/web_search_tool.py
import requests
import json
from .base_tool import BaseTool
import config # Import from the root directory's config.py

class WebSearchTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="web_search",
            description="Searches the web using Google, Tavily, or Brave Search API."
        )
        self.max_results_per_engine = 3 # Number of results to return

    def _google_search(self, query: str) -> str:
        if not config.GOOGLE_API_KEY or not config.GOOGLE_CSE_ID:
            return "Error: Google API Key or CSE ID is not configured."
        
        url = "https://www.googleapis.com/customsearch/v1"
        params = {
            "key": config.GOOGLE_API_KEY,
            "cx": config.GOOGLE_CSE_ID,
            "q": query,
            "num": self.max_results_per_engine
        }
        try:
            response = requests.get(url, params=params, timeout=10)
            response.raise_for_status()
            search_results = response.json()
            
            output = f"Google Search Results for '{query}':\n"
            if "items" in search_results:
                for i, item in enumerate(search_results["items"][:self.max_results_per_engine]):
                    output += f"{i+1}. Title: {item.get('title', 'N/A')}\n"
                    output += f"   Link: {item.get('link', 'N/A')}\n"
                    output += f"   Snippet: {item.get('snippet', 'N/A')}\n\n"
                return output.strip()
            else:
                return f"No results found from Google for '{query}'."
        except requests.exceptions.RequestException as e:
            return f"Error during Google Search: {e}"
        except Exception as e:
            return f"An unexpected error occurred with Google Search: {e}"

    def _tavily_search(self, query: str) -> str:
        if not config.TAVILY_API_KEY:
            return "Error: Tavily API Key is not configured."
        
        url = "https://api.tavily.com/search"
        payload = {
            "api_key": config.TAVILY_API_KEY,
            "query": query,
            "search_depth": "basic", # or "advanced"
            "max_results": self.max_results_per_engine,
            # "include_domains": [],
            # "exclude_domains": []
        }
        headers = {"Content-Type": "application/json"}
        try:
            response = requests.post(url, json=payload, headers=headers, timeout=10)
            response.raise_for_status()
            search_results = response.json()
            
            output = f"Tavily Search Results for '{query}':\n"
            if "results" in search_results and search_results["results"]:
                for i, item in enumerate(search_results["results"][:self.max_results_per_engine]):
                    output += f"{i+1}. Title: {item.get('title', 'N/A')}\n"
                    output += f"   URL: {item.get('url', 'N/A')}\n"
                    output += f"   Content: {item.get('content', 'N/A')[:250]}...\n\n" # Show a snippet of content
                return output.strip()
            else:
                return f"No results found from Tavily for '{query}'."
        except requests.exceptions.RequestException as e:
            return f"Error during Tavily Search: {e}"
        except Exception as e:
            return f"An unexpected error occurred with Tavily Search: {e}"

    def _brave_search(self, query: str) -> str:
        if not config.BRAVE_SEARCH_API_KEY:
            return "Error: Brave Search API Key is not configured."

        url = "https://api.search.brave.com/res/v1/web/search"
        params = {"q": query, "count": self.max_results_per_engine}
        headers = {
            "Accept": "application/json",
            "X-Subscription-Token": config.BRAVE_SEARCH_API_KEY
        }
        try:
            response = requests.get(url, params=params, headers=headers, timeout=10)
            response.raise_for_status()
            search_results = response.json()

            output = f"Brave Search Results for '{query}':\n"
            # Brave API structure might differ, this is a common pattern
            # Check Brave API documentation for the exact structure of 'web' or 'results'
            results_key = None
            if 'web' in search_results and 'results' in search_results['web']:
                results_key = search_results['web']['results']
            elif 'results' in search_results: # A more generic fallback
                 results_key = search_results['results']


            if results_key:
                for i, item in enumerate(results_key[:self.max_results_per_engine]):
                    output += f"{i+1}. Title: {item.get('title', 'N/A')}\n"
                    output += f"   URL: {item.get('url', 'N/A')}\n"
                    output += f"   Description: {item.get('description', item.get('snippet', 'N/A'))}\n\n"
                return output.strip()
            else:
                return f"No results found from Brave Search for '{query}' or unknown response structure."
        except requests.exceptions.RequestException as e:
            return f"Error during Brave Search: {e}"
        except Exception as e:
            return f"An unexpected error occurred with Brave Search: {e}"

    def execute(self, arguments: dict) -> str:
        """
        Executes a web search.
        Args:
            arguments (dict): Must contain 'query' (str).
                              Optional: 'engine' (str) - "google", "tavily", "brave".
                               Defaults to "google".
        Returns:
            str: Formatted search results or an error message.
        """
        if self.interrupted:
            return "Web search interrupted by user."

        query = arguments.get("query")
        if not query:
            return "Error: 'query' argument is missing for web_search tool."

        engine = arguments.get("engine", "google").lower()
        
        print(f"Performing web search for '{query}' using {engine}...")

        if engine == "google":
            return self._google_search(query)
        elif engine == "tavily":
            return self._tavily_search(query)
        elif engine == "brave":
            return self._brave_search(query)
        else:
            return f"Error: Unknown search engine '{engine}'. Supported engines: google, tavily, brave."

if __name__ == '__main__':
    # Example usage (requires API keys in .env and config.py)
    # Make sure config.py can find your .env file (e.g., place .env in project root)
    search_tool = WebSearchTool()

    test_query = "latest cybersecurity news"
    
    print(f"\n--- Testing Google Search for: {test_query} ---")
    google_results = search_tool.execute({"query": test_query, "engine": "google"})
    print(google_results)

    print(f"\n--- Testing Tavily Search for: {test_query} ---")
    tavily_results = search_tool.execute({"query": test_query, "engine": "tavily"})
    print(tavily_results)
    
    print(f"\n--- Testing Brave Search for: {test_query} ---")
    brave_results = search_tool.execute({"query": test_query, "engine": "brave"})
    print(brave_results)

    print(f"\n--- Testing with missing query ---")
    error_results = search_tool.execute({})
    print(error_results)

    print(f"\n--- Testing with invalid engine ---")
    invalid_engine_results = search_tool.execute({"query": "test", "engine": "duckduckgo"})
    print(invalid_engine_results)
