
import os
from typing import Literal, List, Optional, Dict, Any
from langchain.tools import StructuredTool
import os
from dotenv import load_dotenv
load_dotenv()
from typing import Any, Callable, List, Optional, cast, Dict, Literal, Union
from pydantic import BaseModel, Field, field_validator
from langchain.tools import BaseTool, Tool
from src.langgraph.app.core.langgraph.swarm import create_handoff_tool

# This Pydantic model correctly defines the arguments for the LLM
from langchain_tavily import TavilySearch

# Load environment variables from a .env file for local development.
load_dotenv()

# --- Pydantic Input Schema for Robust Validation ---
class TavilySearchInput(BaseModel):
    """Input schema for the Tavily Search tool."""
    query: str = Field(..., description="The search query to look up.")
    max_results: Optional[int] = Field(
        default=5, description="The maximum number of search results to return."
    )
    search_depth: Optional[Literal["basic", "advanced"]] = Field(
        default="advanced", description="The depth of the search: 'basic' or 'advanced'."
    )
    topic: Optional[Literal["general", "news", "finance"]] = Field(
        default="general", description="The topic for the search."
    )
    include_domains: Optional[List[str]] = Field(
        default=None, description="A list of domains to specifically include in the search."
    )
    exclude_domains: Optional[List[str]] = Field(
        default=None, description="A list of domains to specifically exclude from the search."
    )


# --- Production-Ready Tool Class ---
class TavilySearchTool:
    """
    A robust, production-ready tool for performing web searches with Tavily.

    This class encapsulates the logic for the search tool, using Pydantic for
    input validation and providing a secure way to handle API keys for both
    local development and production deployment.
    """
    def __init__(self, api_key: Optional[str] = None):
        """
        Initializes the tool and securely configures the API key.
        """
        self.api_key = api_key or os.getenv("TAVILY_API_KEY")
        if not self.api_key:
            raise ValueError(
                "Tavily API key not provided. Please pass it to the constructor "
                "or set the TAVILY_API_KEY environment variable."
            )
        # Instantiate the TavilySearch tool from the correct package once.
        self.tool = TavilySearch(tavily_api_key=self.api_key)


    def run(self, **kwargs) -> List[Dict[str, Any]]:
        """
        Executes the Tavily search with validated input.

        This method is designed to be wrapped by a LangChain StructuredTool.
        It takes keyword arguments that are validated by the Pydantic schema.
        """
        try:
            # Validate the input using the Pydantic model
            validated_args = TavilySearchInput(**kwargs)

            # Convert the Pydantic model to a dictionary for invocation.
            # exclude_none=True ensures we don't pass optional args if they weren't provided.
            invoke_args = validated_args.model_dump(exclude_none=True)

            # Perform the search using the validated arguments
            result = self.tool.invoke(invoke_args)
            return result
        except Exception as e:
            # Return a structured error message if something goes wrong
            return [{"error": f"An error occurred during the search: {e}"}]

# --- Create a default instance and a StructuredTool ---

# 1. Instantiate our production-ready class.
default_tavily_instance = TavilySearchTool()

# 2. Create a StructuredTool from the class method.
tavily_search_tool = StructuredTool.from_function(
    name="tavily_web_search",
    func=default_tavily_instance.run,
    description=(
        "A search engine optimized for comprehensive, accurate, and trusted results. "
        "Use this for any general web search, research, or to find current events."
    ),
    args_schema=TavilySearchInput
)


transfer_to_Deep_Research_Agent = create_handoff_tool(
    agent_name="Deep_Research_Agent",
    description="Transfer the user to the Deep_Research_Agent to perform deep research and implement the solution to the user's request.",
)


transfer_to_Tools_Agent = create_handoff_tool(
    agent_name="Tools_Agent",
    description="Transfer the user to the Tools_Agent to perform practical tasks that may require specific toolsets like sports, travel, google, weather, or more advanced tools and implement the solution to the user's request.",
) 

# Your list of base tools remains the same
base_tools = [transfer_to_Deep_Research_Agent, tavily_search_tool, transfer_to_Tools_Agent]


