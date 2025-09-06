
import os
from typing import Literal
from langchain_community.tools.tavily_search import TavilySearchResults
import os
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Callable, List, Optional, cast, Dict, Literal, Union
from pydantic import BaseModel, Field, field_validator
from langchain.tools import BaseTool, Tool


class SearchToolInput(BaseModel):
    query: str = Field(..., description="The search query to look up.")
    max_results: Optional[int] = Field(default=10, description="The maximum number of search results to return.")

# Define the Tool
class TavilySearchTool:
    def __init__(self, max_results: int = 10):
        self.max_results = max_results

    def search(self, query: str) -> Optional[List[Dict[str, Any]]]:
        """
        Perform a web search using the Tavily search engine.
        """
        try:
            # Initialize the Tavily search tool with the configured max_results
            search_tool = TavilySearchResults(max_results=self.max_results, tavily_api_key=os.getenv("TAVILY_API_KEY"))

            # Perform the search (synchronously)
            result = search_tool.invoke({"query": query})

            # Return the search results
            return result
        except Exception as e:
            return {"error": str(e)}

tavily_search_tool = Tool(
    name="Tavily Search",
    func=TavilySearchTool().search,
    description="Performs web searches using the Tavily search engine, providing accurate and trusted results for general queries.",
    args_schema=SearchToolInput
)

base_tools = [tavily_search_tool]

