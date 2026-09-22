from ddgs import DDGS
from smolagents import Tool

MAX_RESULTS = 5

class SearchTool(Tool):
    """DuckDuckGo web search.

    smolagents ships DuckDuckGoSearchTool, but 1.18.0 imports the retired
    `duckduckgo_search` package, which warns on every instantiation. This is the
    same tool built on its replacement, `ddgs`.
    """

    name = "web_search"
    description = "Performs a DuckDuckGo web search for the given query and returns the top results."
    inputs = {
        "query": {
            "type": "string",
            "description": "The search query to perform."
        }
    }
    output_type = "string"

    def __init__(self, max_results: int = MAX_RESULTS):
        super().__init__()
        self.max_results = max_results
        self.ddgs = DDGS()

    def forward(self, query: str) -> str:
        try:
            results = self.ddgs.text(query, max_results=self.max_results)
        except Exception as exc:
            # Returned, not raised: a raising tool ends the agent run, whereas a
            # message lets the agent rephrase or fall back to another tool.
            return f"Search failed for '{query}': {exc}"

        if not results:
            return f"No results found for '{query}'. Try a shorter or less restrictive query."

        return "## Search Results\n\n" + "\n\n".join(
            f"[{r['title']}]({r['href']})\n{r['body']}" for r in results
        )


# Initialize the tool
search_tool = SearchTool()

if __name__ == "__main__":
    print(search_tool.forward("Who's the current President of France?"))
