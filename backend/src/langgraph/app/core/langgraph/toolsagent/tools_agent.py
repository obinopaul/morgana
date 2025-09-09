import asyncio
import os
import logging
from typing import List, Any, Optional, Sequence, TypedDict, Annotated
from dotenv import load_dotenv

# LangChain and LangGraph core imports
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps

# Local application imports
from src.langgraph.app.core.langgraph.agents import create_agent
from src.langgraph.app.core.langgraph.toolsagent.prompts import TOOLS_AGENT_PROMPT
from src.langgraph.app.core.langgraph.toolsagent.tools import base_tools
from langchain_mcp_adapters.client import MultiServerMCPClient

# Load environment variables from .env file
load_dotenv()

# --- Production-Ready Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Agent State & Configuration Schemas ---
class AgentState(TypedDict):
    """
    Defines the state of the agent, which flows through the graph.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: RemainingSteps

class AgentConfig(TypedDict, total=False):
    """
    A schema for configuring the agent's compiled graph, allowing for interrupts.
    """
    interrupt_before: List[str]
    interrupt_after: List[str]


# --- Refactored Agent Factory ---
class ToolsAgent:
    """
    An advanced factory for creating an autonomous agent that combines local
    tools with tools from a specific MCP server.
    """

    def __init__(self,
                 llm: BaseLanguageModel,
                 max_steps: int = 15,
                 checkpointer: Optional[BaseCheckpointSaver] = None):
        """
        Initializes the agent factory's configuration.

        Args:
            llm: An initialized language model instance (e.g., ChatOpenAI).
            max_steps: The maximum number of LLM calls before stopping.
            checkpointer: An optional LangGraph checkpointer for state persistence.
        """
        self.llm = llm
        self.max_steps = max_steps
        self.checkpointer = checkpointer
        logger.info("ToolsAgent factory initialized.")

    async def _load_all_tools(self, max_retries: int = 3, delay: int = 5) -> List[Any]:
        """
        Asynchronously loads tools from local sources and a dedicated MCP server.
        Includes a retry mechanism for resilience.
        """
        logger.info("Initializing tools from local files and dedicated MCP server...")
        all_tools = list(base_tools)
        mcp_tools = []

        microsoft_server_url = os.getenv("MICROSOFT_MCP_SERVER_URL")

        if not microsoft_server_url:
            logger.warning("MICROSOFT_MCP_SERVER_URL not set. Proceeding with local tools only.")
        else:
            logger.info(f"Connecting to Microsoft MCP server at {microsoft_server_url}...")
            mcp_configs = {"microsoft": {"url": microsoft_server_url, "transport": "streamable_http"}}
            client = MultiServerMCPClient(mcp_configs)

            for attempt in range(max_retries):
                try:
                    mcp_tools = await client.get_tools()
                    logger.info(f"Successfully loaded {len(mcp_tools)} tools from the MCP server.")
                    break
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1}/{max_retries} failed to connect to MCP server: {e}")
                    if attempt + 1 == max_retries:
                        logger.error("All retry attempts failed. Proceeding with local tools only.")
                        break
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)

        all_tools.extend(mcp_tools)

        if not all_tools:
            logger.warning("Warning: No tools were loaded at all (neither local nor MCP).")
        else:
            tool_names = [tool.name for tool in all_tools]
            logger.info(f"Total tools available: {len(all_tools)}. Names: {tool_names}")

        return all_tools

    async def build(self, config: Optional[AgentConfig] = None) -> CompiledStateGraph:
        """
        Asynchronously builds and compiles the agent graph executor.

        This method first loads all required tools (local and remote) and then
        constructs the agent using LangGraph's `create_agent` factory.

        Args:
            config: Optional configuration for setting graph interrupts.

        Returns:
            A compiled LangGraph state graph (executor) ready for execution.
        """
        logger.info("Building and compiling the ToolsAgent executor...")
        
        # 1. Asynchronously load all tools before building the agent
        loaded_tools = await self._load_all_tools()
        
        # 2. Use the factory function to create the agent graph
        agent_executor = create_agent(
            model=self.llm,
            tools=loaded_tools,
            prompt=TOOLS_AGENT_PROMPT,
            state_schema=AgentState,
            checkpointer=self.checkpointer,
            interrupt_before=config.get("interrupt_before") if config else None,
            interrupt_after=config.get("interrupt_after") if config else None
        )
        
        logger.info("ToolsAgent executor compiled successfully.")
        return agent_executor


async def main():
    """Main function to demonstrate instantiating and running the ToolsAgent."""
    from langgraph.checkpoint.memory import MemorySaver

    # Verify that necessary environment variables are set
    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set in the .env file.")
    if not os.getenv("MICROSOFT_MCP_SERVER_URL"):
        logger.warning("MICROSOFT_MCP_SERVER_URL is not set; the agent will only use its local tools.")

    # --- Agent Setup ---
    # 1. Initialize the language model
    llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)
    
    # 2. Setup persistence (in-memory for this example)
    memory = MemorySaver()

    # 3. Instantiate the agent factory with the LLM and checkpointer
    agent_factory = ToolsAgent(llm=llm, checkpointer=memory)

    # 4. Define an interrupt configuration to pause execution before the tool node
    interrupt_config: AgentConfig = {"interrupt_before": ["tools"]}
    
    # 5. Asynchronously build the agent executor
    #    The `build` call now handles the async tool loading internally.
    agent_executor = await agent_factory.build(config=interrupt_config)

    # --- Agent Execution ---
    thread_id = "sports_convo_thread_002"
    run_config = {"configurable": {"thread_id": thread_id}}

    query = "Who was the NBA champion in 2022, and which country won the world cup in 2018?"
    initial_input = {
        "messages": [HumanMessage(content=query)],
        "remaining_steps": agent_factory.max_steps,
    }

    logger.info(f"--- Running Agent for Thread '{thread_id}' with Query: '{query}' ---")
    
    try:
        # Stream the agent's execution steps using the executor directly
        async for chunk in agent_executor.astream(initial_input, config=run_config, recursion_limit=150):
            for key, value in chunk.items():
                if key == "agent" and value.get('messages'):
                    ai_msg = value['messages'][-1]
                    if ai_msg.tool_calls:
                        tool_names = ", ".join([call['name'] for call in ai_msg.tool_calls])
                        logger.info(f"Agent requesting tool(s): {tool_names}")
                    else:
                        logger.info(f"\n--- Final Answer ---\n{ai_msg.content}")
                elif key == "tools" and value.get('messages'):
                    tool_msg = value['messages'][-1]
                    logger.info(f"Tool executed. Result: {str(tool_msg.content)[:300]}...")
    except Exception as e:
        logger.error(f"An error occurred during agent execution: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())