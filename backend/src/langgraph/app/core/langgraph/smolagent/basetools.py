
import os
from typing import Literal
from langchain_community.tools.tavily_search import TavilySearchResults
from langchain.tools import StructuredTool
import os
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Callable, List, Optional, cast, Dict, Literal, Union
from pydantic import BaseModel, Field, field_validator
from langchain.tools import BaseTool, Tool
from src.langgraph.app.core.langgraph.swarm import create_handoff_tool

# This Pydantic model correctly defines the arguments for the LLM
class SearchToolInput(BaseModel):
    query: str = Field(..., description="The search query to look up for recent and relevant information.")
    max_results: int = Field(5, description="The maximum number of search results to return.")

def tavily_search_func(query: str, max_results: int = 5) -> List[Dict[str, Any]]:
    """
    Performs a web search using the Tavily search engine.
    """
    try:
        # Initialize the Tavily search tool
        search_tool = TavilySearchResults(max_results=max_results)
        # Perform the search and return the result
        return search_tool.invoke(query)
    except Exception as e:
        return [f"An error occurred during search: {e}"]

# 3. Create the tool using StructuredTool.from_function
#    This correctly handles multiple named arguments.
tavily_search_tool = StructuredTool.from_function(
    func=tavily_search_func,
    name="tavily_search",
    description="Performs web searches using the Tavily search engine to find accurate and trusted results for news, facts, and general queries.",
    args_schema=SearchToolInput,
)


transfer_to_researcher_agent = create_handoff_tool(
    agent_name="Deep_Research_Agent",
    description="Transfer the user to the Deep_Research_Agent to perform deep research and implement the solution to the user's request.",
)


transfer_to_tools_agent = create_handoff_tool(
    agent_name="Tools_Agent",
    description="Transfer the user to the Tools_Agent to perform practical tasks that may require specific toolsets like sports, travel, google, weather, or more advanced tools and implement the solution to the user's request.",
) 

# Your list of base tools remains the same
base_tools = [transfer_to_researcher_agent, tavily_search_tool, transfer_to_tools_agent]


