# tools/cve_search_tool.py
from .base_tool import BaseTool
from .web_search_tool import WebSearchTool # Uses the web search tool

class CVESearchTool(BaseTool):
    def __init__(self):
        super().__init__(
            name="cve_search",
            description="Searches for information about Common Vulnerabilities and Exposures (CVEs)."
        )
        # This tool will delegate to the WebSearchTool for now.
        # It could be expanded to use specific CVE APIs (e.g., NVD, Vulners)
        self.web_search_tool = WebSearchTool()
        self.web_search_tool.max_results_per_engine = 2 # Fewer results for targeted CVE search

    def execute(self, arguments: dict) -> str:
        """
        Searches for CVE information.
        Args:
            arguments (dict): Can contain 'cve_id' (str) or 'query' (str).
                              If 'cve_id' is provided, it takes precedence.
        Returns:
            str: CVE information or an error message.
        """
        if self.interrupted:
            return "CVE search interrupted by user."

        cve_id = arguments.get("cve_id")
        general_query = arguments.get("query")

        if not cve_id and not general_query:
            return "Error: 'cve_id' or 'query' argument must be provided for cve_search tool."

        search_query = ""
        if cve_id:
            # Prioritize official sources for specific CVE IDs
            search_query = f"{cve_id} site:cve.mitre.org OR site:nvd.nist.gov"
            print(f"Performing CVE search for ID: {cve_id} using targeted web search...")
        elif general_query:
            search_query = f"CVE details for {general_query}"
            print(f"Performing general CVE search for query: {general_query} using web search...")
        
        # Delegate to web search tool. You might prefer a specific engine.
        # For now, it will use the web_search_tool's default or what AI passes.
        search_args = {"query": search_query, "engine": "google"} # Default to Google for CVEs
        
        # Pass interruption status to the sub-tool
        self.web_search_tool.set_interrupted(self.interrupted)
        result = self.web_search_tool.execute(search_args)
        
        if "Error:" in result and cve_id: # Fallback for specific CVE ID if targeted search fails
             print(f"Targeted search for {cve_id} yielded an error or no results, trying broader search...")
             search_args = {"query": cve_id, "engine": "google"}
             result = self.web_search_tool.execute(search_args)

        return result

if __name__ == '__main__':
    # Example usage
    cve_tool = CVESearchTool()

    print("\n--- Testing CVE Search by ID ---")
    # A well-known CVE for testing
    results_id = cve_tool.execute({"cve_id": "CVE-2021-44228"}) # Log4Shell
    print(results_id)

    print("\n--- Testing CVE Search by general query ---")
    results_query = cve_tool.execute({"query": "Apache Struts remote code execution vulnerabilities 2023"})
    print(results_query)

    print("\n--- Testing CVE Search with missing arguments ---")
    error_results = cve_tool.execute({})
    print(error_results)
