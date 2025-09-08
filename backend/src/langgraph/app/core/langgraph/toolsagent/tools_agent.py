import asyncio
import os
import logging
from typing import List, Sequence, TypedDict, Annotated, Literal, Optional, Dict, Any
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, HumanMessage
from src.langgraph.app.core.langgraph.agents import create_agent
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph.state import CompiledStateGraph
from langgraph.graph.message import add_messages
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.managed import RemainingSteps 

# Import local tools and the system prompt
from src.langgraph.app.core.langgraph.toolsagent.tools import base_tools
from src.langgraph.app.core.langgraph.toolsagent.prompts import TOOLS_AGENT_PROMPT

# Load environment variables from .env file
load_dotenv()

# --- Production-Ready Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Enhanced Agent Configuration & State ---
class AgentState(TypedDict):
    """
    Defines the state of the agent. This is the central data structure that flows
    through the graph. Using LangGraph's `RemainingSteps` provides robust,
    built-in loop protection.

    Attributes:
        messages: The history of messages in the conversation.
        remaining_steps: The number of steps left before execution is halted.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: RemainingSteps

class AgentConfig(TypedDict, total=False):
    """
    A schema for configuring the agent's compiled graph, allowing for
    interrupts before or after specific nodes.
    """
    interrupt_before: List[str]
    interrupt_after: List[str]


class ToolsAgent:
    """
    An advanced, asynchronous, and robust autonomous agent that combines local
    tools with tools from a specific list of MCP servers defined in environment variables.
    """

    def __init__(self, model_name: str = "gpt-4o", max_steps: int = 15, checkpointer: Optional[BaseCheckpointSaver] = None):
        """
        Initializes the agent's configuration.

        Args:
            model_name: The specific OpenAI model name to use (e.g., "gpt-4o").
            max_steps: The maximum number of LLM calls before forcing a stop.
            checkpointer: An optional LangGraph checkpointer for state persistence and memory.
        """
        self.model_name = model_name
        self.max_steps = max_steps
        self.checkpointer = checkpointer
        self.tools: list = []
        self._initialized_tools = False
        self.executor: Optional[CompiledStateGraph] = None

    async def _load_mcp_tools(self, max_retries: int = 3, delay: int = 5):
        """
        Asynchronously connects to the dedicated Microsoft MCP server URL defined
        in the environment and combines its tools with local tools. Includes a retry
        mechanism for resilience.
        """
        if self._initialized_tools:
            return

        logger.info("Initializing tools from local files and dedicated MCP server...")

        mcp_tools = []
        microsoft_server_url = os.getenv("MICROSOFT_MCP_SERVER_URL")

        if not microsoft_server_url:
            logger.warning("Environment variable 'MICROSOFT_MCP_SERVER_URL' is not set. Proceeding with local tools only.")
        else:
            logger.info(f"Found Microsoft MCP server URL: {microsoft_server_url}")
            mcp_configs = {"microsoft": {"url": microsoft_server_url, "transport": "streamable_http"}}
            client = MultiServerMCPClient(mcp_configs)
            
            for attempt in range(max_retries):
                try:
                    mcp_tools = await client.get_tools()
                    logger.info(f"Successfully loaded {len(mcp_tools)} tools from the Microsoft MCP server.")
                    break  # Success, exit retry loop
                except Exception as e:
                    logger.error(f"Attempt {attempt + 1}/{max_retries} failed to connect to MCP server: {e}")
                    if attempt + 1 == max_retries:
                        logger.error("All retry attempts failed. Proceeding with local tools only.")
                        break
                    logger.info(f"Retrying in {delay} seconds...")
                    await asyncio.sleep(delay)

        self.tools = base_tools + mcp_tools
        
        if not self.tools:
             logger.warning("No tools were loaded at all (neither local nor MCP). The agent will have limited capabilities.")
        else:
            logger.info(f"Total tools available: {len(self.tools)}. Names: {[tool.name for tool in self.tools]}")
        
        self._initialized_tools = True

    async def _build_executor(self, config: Optional[AgentConfig] = None):
        """
        Builds and compiles the agent graph using LangChain's create_agent factory.
        This method is called lazily to ensure tools are loaded before compilation.
        """
        if self.executor:
            return

        logger.info("Building and compiling the sports agent executor...")
        
        # Ensure tools are loaded before building the agent
        await self._load_mcp_tools()
        
        # Instantiate the language model
        llm = ChatOpenAI(model=self.model_name, temperature=0, streaming=True)
        
        # Use LangChain's factory to create the standard ReAct agent graph
        self.executor = create_agent(
            model=llm,
            tools=self.tools,
            prompt=TOOLS_AGENT_PROMPT,
            state_schema=AgentState,
            checkpointer=self.checkpointer,
            interrupt_before=config.get("interrupt_before") if config else None,
            interrupt_after=config.get("interrupt_after") if config else None
        )
        logger.info("Sports agent executor compiled successfully.")



    async def ainvoke(self, query: str, thread_id: str) -> Dict[str, Any]:
        """
        Asynchronously invokes the agent to get the final result in a single call.
        This is ideal for multi-agent systems where one agent's complete output is
        the input for another.

        Args:
            query: The user's query for the agent to process.
            thread_id: A unique identifier for the conversation thread for memory.

        Returns:
            A dictionary representing the final state of the agent's execution.
        """
        await self._build_executor()

        run_config = {"configurable": {"thread_id": thread_id}}
        initial_input = {
            "messages": [HumanMessage(content=query)],
            "remaining_steps": self.max_steps,
        }

        logger.info(f"--- Invoking Agent for Thread '{thread_id}' with Query: '{query}' ---")
        
        final_state = await self.executor.ainvoke(initial_input, config=run_config)
        
        logger.info(f"\n--- Final Answer ---\n{final_state['messages'][-1].content}")
        return final_state
    
    
    async def arun(self, query: str, thread_id: str, config: Optional[AgentConfig] = None):
        """
        Asynchronously runs the agent with a given query and conversation thread ID.

        Args:
            query: The user's query for the agent to process.
            thread_id: A unique identifier for the conversation thread for memory.
            config: Optional configuration for setting interrupts.
        """
        # Build the executor on the first run
        await self._build_executor(config)

        # Define the per-run configuration, including the thread_id for memory
        run_config = {"configurable": {"thread_id": thread_id}}

        # Prepare the initial input for the agent graph
        initial_input = {
            "messages": [HumanMessage(content=query)],
            "remaining_steps": self.max_steps,
        }

        logger.info(f"--- Running Agent for Thread '{thread_id}' with Query: '{query}' ---")
        
        # Stream the agent's execution steps for real-time logging
        try:
            async for chunk in self.executor.astream(initial_input, config=run_config, recursion_limit=150):
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


async def main():
    """Main function to instantiate and run the agent with advanced features."""
    from langgraph.checkpoint.memory import MemorySaver

    if not os.getenv("OPENAI_API_KEY"):
        raise ValueError("OPENAI_API_KEY must be set in the .env file.")
    if not os.getenv("SPORTS_MCP_SERVERS"):
        raise ValueError("SPORTS_MCP_SERVERS must be set in the .env file (e.g., 'NBA,SOCCER').")

    # 1. Setup persistence: MemorySaver keeps the state of each conversation in memory.
    #    For production, you might use RedisSaver, PostgresSaver, etc.
    memory = MemorySaver()

    # 2. Instantiate the agent with the checkpointer for memory.
    agent = ToolsAgent(llm_provider=ChatOpenAI, model_name="gpt-4o", checkpointer=memory)
    
    # 3. Define an interrupt configuration to pause execution before the tool node.
    #    This is useful for debugging or adding human-in-the-loop validation.
    interrupt_config: AgentConfig = {"interrupt_before": ["tools"]}
    
    # 4. Define a unique ID for the conversation thread.
    thread_id = "sports_convo_thread_001"

    query = "Who was the NBA champion in 2022, and which country won the world cup in 2018?"
    
    # 5. Run the agent.
    await agent.arun(query, thread_id=thread_id, config=interrupt_config)


if __name__ == "__main__":
    # To run this code, you need to have your environment variables set up.
    # Create a .env file with:
    # OPENAI_API_KEY="your_openai_api_key"
    # SPORTS_MCP_SERVERS="NBA,SOCCER"
    # MCP_NBA_SERVER_URL="http://localhost:8001"
    # MCP_SOCCER_SERVER_URL="http://localhost:8002"
    # ...and ensure your MCP servers are running.
    asyncio.run(main())
    
    

