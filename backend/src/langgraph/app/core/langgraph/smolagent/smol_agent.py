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

# Import local base tools and the system prompt
from src.langgraph.app.core.langgraph.smolagent.basetools import base_tools
from src.langgraph.app.core.langgraph.smolagent.prompts import SMOL_AGENT_PROMPT

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


class SMOLAgent:
    """
    A robust, simplified autonomous agent that uses a predefined set of local,
    base tools. It does not connect to any external MCP servers.
    """

    def __init__(self, model_name: str = "gpt-4o", max_steps: int = 15, checkpointer: Optional[BaseCheckpointSaver] = None, tools = None):
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
        self.tools: list = tools if tools is not None else base_tools
        self.executor: Optional[CompiledStateGraph] = None # Compiled agent graph executor

    async def _build_executor(self, config: Optional[AgentConfig] = None):
        """
        Builds and compiles the agent graph using LangChain's create_agent factory.
        This method is called lazily to ensure tools are loaded before compilation.
        """
        if self.executor:
            return

        logger.info("Building and compiling the sports agent executor...")
        
        # Instantiate the language model
        llm = ChatOpenAI(model=self.model_name, temperature=0, streaming=True)
        
        # Use LangChain's factory to create the standard ReAct agent graph
        self.executor = create_agent(
            model=llm,
            tools=self.tools,
            prompt=SMOL_AGENT_PROMPT,
            state_schema=AgentState,
            checkpointer=self.checkpointer,
            interrupt_before=config.get("interrupt_before") if config else None,
            interrupt_after=config.get("interrupt_after") if config else None
        )
        logger.info("Sports agent executor compiled successfully.")


    async def ainvoke(self, messages: Sequence[BaseMessage], thread_id: str) -> Dict[str, Any]:
        """
        Asynchronously invokes the agent to get the final result in a single call.
        This is ideal for multi-agent systems where one agent's complete output is
        the input for another.

        Args:
            messages: The list of BaseMessage objects representing the conversation history.
            thread_id: A unique identifier for the conversation thread for memory.

        Returns:
            A dictionary representing the final state of the agent's execution.
        """
        await self._build_executor()

        run_config = {"configurable": {"thread_id": thread_id}}
        initial_input = {
            "messages": messages,
            "remaining_steps": self.max_steps,
        }

        logger.info(f"--- Invoking Agent for Thread '{thread_id}' with Messages: {messages} ---")
        
        final_state = await self.executor.ainvoke(initial_input, config=run_config)
        
        logger.info(f"\n--- Final Answer ---\n{final_state['messages'][-1].content}")
        return final_state
    
    
    async def arun(self, messages: Sequence[BaseMessage], thread_id: str, config: Optional[AgentConfig] = None):
        """
        Asynchronously runs the agent with a given list of messages and conversation thread ID.

        Args:
            messages: The list of BaseMessage objects representing the conversation history.
            thread_id: A unique identifier for the conversation thread for memory.
            config: Optional configuration for setting interrupts.
        """
        # Build the executor on the first run
        await self._build_executor(config)

        # Define the per-run configuration, including the thread_id for memory
        run_config = {"configurable": {"thread_id": thread_id}}

        # Prepare the initial input for the agent graph
        initial_input = {
            "messages": messages,
            "remaining_steps": self.max_steps,
        }

        logger.info(f"--- Running Agent for Thread '{thread_id}' with Messages: {messages} ---")
        
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
    """Main function to instantiate and run the SMOLAgent."""
    from langgraph.checkpoint.memory import MemorySaver

    if not (os.getenv("OPENAI_API_KEY") and os.getenv("TAVILY_API_KEY")):
        raise ValueError("API keys for OpenAI and Tavily must be set in the .env file.")

    memory = MemorySaver()
    agent = SMOLAgent(llm_provider=ChatOpenAI, model_name="gpt-4o", checkpointer=memory, tools=base_tools)
    thread_id = "smol_convo_789"

    query = "What is the current time and what are the top 3 news headlines in Oklahoma City today?"
    await agent.arun(query, thread_id=thread_id)


if __name__ == "__main__":
    asyncio.run(main())
