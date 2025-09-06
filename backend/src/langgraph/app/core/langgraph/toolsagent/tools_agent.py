import asyncio
import os
import logging
from typing import List, Sequence, TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langchain_mcp_adapters.client import MultiServerMCPClient
from langgraph.graph import StateGraph, END, CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointer

# Import local tools and the system prompt
from tools import local_tools
from prompts import TOOLS_AGENT_PROMPT

# Load environment variables from .env file
load_dotenv()

# --- Production-Ready Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Enhanced Agent Configuration & State ---
class AgentState(TypedDict):
    """
    Represents the state of our agent, including robust loop protection.

    Attributes:
        messages: The history of messages in the conversation.
        remaining_steps: The number of steps left before execution is halted.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: int

class AgentConfig(TypedDict, total=False):
    """A schema for configuring the agent's compiled graph."""
    interrupt_before: List[str]
    interrupt_after: List[str]


class ToolsAgent:
    """
    An advanced, asynchronous, and robust autonomous agent that combines local
    tools with tools from a specific list of MCP servers defined in environment variables.
    """

    def __init__(self, llm_provider: type[BaseChatModel], model_name: str, max_steps: int = 15, checkpointer: Optional[BaseCheckpointer] = None):
        """
        Initializes the agent's configuration.

        Args:
            llm_provider: The class of the language model to use (e.g., ChatOpenAI).
            model_name: The specific model name (e.g., "gpt-4o").
            max_steps: The maximum number of LLM calls before forcing a stop.
            checkpointer: An optional LangGraph checkpointer for state persistence.
        """
        self.llm_provider = llm_provider
        self.model_name = model_name
        self.max_steps = max_steps
        self.checkpointer = checkpointer
        self.tools: list = []
        self.tool_node: ToolNode | None = None
        self._initialized_tools = False

    async def _load_and_configure_tools_from_env(self, max_retries: int = 3, delay: int = 5):
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

        self.tools = local_tools + mcp_tools
        
        if not self.tools:
             logger.warning("No tools were loaded at all (neither local nor MCP). The agent will have limited capabilities.")
        else:
            self.tool_node = ToolNode(self.tools)
            logger.info(f"Total tools available: {len(self.tools)}. Names: {[tool.name for tool in self.tools]}")
        
        self._initialized_tools = True

    def _should_continue(self, state: AgentState) -> Literal["tools", "__end__"]:
        """
        Determines the next step for the agent based on the last message and step count.
        """
        last_message = state["messages"][-1]
        
        if not isinstance(last_message, AIMessage) or not last_message.tool_calls:
            logger.info("--- Agent decided to end execution (no tool calls). ---")
            return END
        
        if state["remaining_steps"] <= 0:
            logger.warning("--- Agent has reached the maximum step limit. Halting execution. ---")
            return END
            
        return "tools"

    async def _call_model(self, state: AgentState) -> dict:
        """
        Invokes the language model and decrements the step counter.
        """
        logger.info(f"--- AGENT (Remaining Steps: {state['remaining_steps']}): Pondering next move... ---")
        
        remaining_steps = state["remaining_steps"] - 1
        
        llm = self.llm_provider(model=self.model_name)
        llm_with_tools = llm.bind_tools(self.tools)
        response = await llm_with_tools.ainvoke(state["messages"])
        
        return {"messages": [response], "remaining_steps": remaining_steps}

    async def _call_tools(self, state: AgentState) -> dict:
        """
        Executes tool calls with robust error handling.
        """
        if not self.tool_node:
            return {"messages": []}

        logger.info("--- AGENT: Executing tool call(s)... ---")
        try:
            return await self.tool_node.ainvoke(state)
        except Exception as e:
            logger.error(f"--- ERROR: Tool execution failed: {e} ---", exc_info=True)
            error_messages = []
            last_message = state["messages"][-1]
            if isinstance(last_message, AIMessage) and last_message.tool_calls:
                for tool_call in last_message.tool_calls:
                    error_messages.append(
                        ToolMessage(
                            content=f"Error executing tool '{tool_call['name']}': {e}",
                            tool_call_id=tool_call["id"],
                        )
                    )
            return {"messages": error_messages}

    def build_graph(self) -> StateGraph:
        """
        Builds the uncompiled LangGraph StateGraph. This allows the agent
        to be integrated as a node in a larger multi-agent system without being
        pre-compiled.
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._call_model)
        if self.tool_node:
            workflow.add_node("tools", self._call_tools)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._should_continue)
        
        if self.tool_node:
            workflow.add_edge("tools", "agent")

        return workflow

    def get_executor(self, config: Optional[AgentConfig] = None) -> CompiledGraph:
        """
        Builds the graph and then compiles it into a runnable executor,
        applying any provided configurations like checkpointers or interrupts.
        
        Args:
            config: An optional dictionary to configure compilation, including
                    'interrupt_before' and 'interrupt_after' lists.

        Returns:
            A compiled LangGraph executor.
        """
        logger.info("Building and compiling the agent executor...")
        agent_graph = self.build_graph()
        
        config = config or {}
        
        agent_executor = agent_graph.compile(
            checkpointer=self.checkpointer,
            interrupt_before=config.get("interrupt_before", []),
            interrupt_after=config.get("interrupt_after", [])
        )
        logger.info("Agent executor compiled successfully.")
        return agent_executor

    async def arun(self, query: str, thread_id: str, config: Optional[AgentConfig] = None):
        """
        Asynchronously runs the agent with a given query and configuration.
        """
        await self._load_and_configure_tools_from_env()
        
        agent_executor = self.get_executor(config)

        # Config for the stream/invoke call, including the thread_id for the checkpointer
        run_config = {"configurable": {"thread_id": thread_id}}

        initial_input = {
            "messages": [
                HumanMessage(content=TOOLS_AGENT_PROMPT),
                HumanMessage(content=query)
            ],
            "remaining_steps": self.max_steps,
        }

        logger.info(f"--- Running Agent for Thread '{thread_id}' with Query: '{query}' ---")
        
        events = agent_executor.astream(initial_input, config=run_config, recursion_limit=150)
        
        try:
            async for chunk in events:
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
    
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("TAVILY_API_KEY")):
        raise ValueError("API keys like OPENAI_API_KEY and TAVILY_API_KEY must be set in the .env file.")

    # 1. Setup persistence (optional, but demonstrates robustness)
    memory = MemorySaver()

    # 2. Instantiate the agent with the checkpointer
    agent = ToolsAgent(llm_provider=ChatOpenAI, model_name="gpt-4o", checkpointer=memory)
    
    # 3. Define an interrupt configuration (optional guardrail)
    # This will cause the graph to pause before the 'tools' node is executed.
    interrupt_config: AgentConfig = {"interrupt_before": ["tools"]}

    # 4. Define a unique ID for the conversation thread
    thread_id = "user_session_123"

    query = "Using my Microsoft account, what are the subjects of my last 2 emails?"
    await agent.arun(query, thread_id=thread_id, config=interrupt_config)


if __name__ == "__main__":
    asyncio.run(main())

