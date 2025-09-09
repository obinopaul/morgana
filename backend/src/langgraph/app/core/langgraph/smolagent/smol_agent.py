import asyncio
import os
import logging
from typing import List, Sequence, TypedDict, Annotated, Optional, Dict, Any

from dotenv import load_dotenv
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps
from src.langgraph.app.core.langgraph.swarm import create_handoff_tool
# Local application imports
from src.langgraph.app.core.langgraph.agents import create_agent
from src.langgraph.app.core.langgraph.smolagent.basetools import base_tools
from src.langgraph.app.core.langgraph.smolagent.prompts import SMOL_AGENT_PROMPT

# Load environment variables from a .env file
load_dotenv()

# --- Production-Ready Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)


# --- Agent State & Configuration Schemas ---
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


# --- Refactored Agent Factory ---
class SMOLAgent:
    """
    A factory class for creating a robust, simplified autonomous agent.

    This class acts as a wrapper around LangGraph's `create_agent` function,
    simplifying the configuration and instantiation of a ReAct agent graph.
    It bundles the necessary components (LLM, tools, prompt, state) and
    provides a single method to build the compiled agent executor.
    """

    def __init__(self,
                 llm: BaseLanguageModel,
                 tools: Optional[List] = None,
                 max_steps: int = 15,
                 checkpointer: Optional[BaseCheckpointSaver] = None):
        """
        Initializes the agent's configuration.

        Args:
            llm: An initialized language model instance (e.g., ChatOpenAI).
            tools: A list of tools for the agent to use. Defaults to base_tools.
            max_steps: The maximum number of LLM calls before stopping.
            checkpointer: An optional LangGraph checkpointer for state persistence.
        """
        self.llm = llm
        self.tools = tools if tools is not None else base_tools
        self.max_steps = max_steps
        self.checkpointer = checkpointer
        logger.info("SMOLAgent factory initialized.")


    async def build(self, config: Optional[AgentConfig] = None) -> CompiledStateGraph:
        """
        Builds and compiles the agent graph executor.

        This method uses the configuration provided during initialization to
        construct the agent using LangGraph's `create_agent` factory.

        Args:
            config: Optional configuration for setting graph interrupts.

        Returns:
            A compiled LangGraph state graph (executor) ready for execution.
        """
        logger.info("Building and compiling the agent executor...")

        agent_executor = create_agent(
            model=self.llm,
            tools=self.tools,
            prompt=SMOL_AGENT_PROMPT,
            state_schema=AgentState,
            checkpointer=self.checkpointer,
            interrupt_before=config.get("interrupt_before") if config else None,
            interrupt_after=config.get("interrupt_after") if config else None
        )
        logger.info("Agent executor compiled successfully.")
        return agent_executor
    
    
async def main():
    """Main function to demonstrate instantiating and running the agent."""
    from langgraph.checkpoint.memory import MemorySaver

    # Ensure necessary API keys are set
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("TAVILY_API_KEY")):
        raise ValueError("API keys for OpenAI and Tavily must be set in the .env file.")

    # 1. Initialize the language model
    llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

    # 2. Instantiate the agent factory with the LLM and other settings
    memory = MemorySaver()
    agent_factory = SMOLAgent(llm=llm, checkpointer=memory, tools=base_tools)

    # 3. Build the agent executor by calling the build() method
    agent_executor = agent_factory.build()
    
    # --- Running the Agent ---
    thread_id = "smol_convo_101"
    run_config = {"configurable": {"thread_id": thread_id}}
    
    query = "What is the current time and what are the top 3 news headlines in Oklahoma City today?"
    initial_input = {
        "messages": [HumanMessage(content=query)],
        "remaining_steps": agent_factory.max_steps, # Use max_steps from the factory
    }

    logger.info(f"--- Running Agent for Thread '{thread_id}' with Input: {initial_input['messages']} ---")

    # 6. Stream the agent's execution steps using the executor directly
    try:
        # The agent_executor object has the `astream` method built-in
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