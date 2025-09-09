import asyncio
import os
import logging
from typing import List, Sequence, TypedDict, Annotated, Optional

from dotenv import load_dotenv
from langchain_core.language_models.base import BaseLanguageModel
from langchain_core.messages import BaseMessage, HumanMessage
from langchain_openai import ChatOpenAI
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph.message import add_messages
from langgraph.graph.state import CompiledStateGraph
from langgraph.managed import RemainingSteps

# --- Local application imports (assuming these exist in your project structure) ---
# Note: These are placeholders. You'll need the actual implementations for this to run.
# Local application imports
from src.langgraph.app.core.langgraph.agents import create_agent
from src.langgraph.app.core.langgraph.smolagent import SMOLAgent
from src.langgraph.app.core.langgraph.toolsagent import ToolsAgent
from src.langgraph.app.core.langgraph.deepagents import DeepResearchAgent
from src.langgraph.app.core.langgraph.swarm import SwarmState, create_handoff_tool, create_swarm


# --- Production-Ready Logging ---
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# --- Agent State & Configuration Schemas ---
class AgentState(TypedDict):
    """
    Defines the state of an individual agent.
    """
    messages: Annotated[Sequence[BaseMessage], add_messages]
    remaining_steps: RemainingSteps


class AgentConfig(TypedDict, total=False):
    """
    A schema for configuring the agent's compiled graph for interrupts.
    """
    interrupt_before: List[str]
    interrupt_after: List[str]


# --- Refactored Agent Swarm Factory ---
class MORGANA:
    """
    A factory class for creating a robust, multi-agent autonomous system (swarm).
    """

    def __init__(self,
                 llm: BaseLanguageModel,
                 checkpointer: Optional[BaseCheckpointSaver] = None):
        """
        Initializes the agent swarm's configuration.

        Args:
            llm: An initialized language model instance (e.g., ChatOpenAI).
            checkpointer: An optional LangGraph checkpointer for state persistence.
        """
        self.llm = llm
        self.checkpointer = checkpointer
        logger.info("MORGANA swarm factory initialized.")

    async def build(self) -> CompiledStateGraph:
        """
        Builds and compiles the multi-agent swarm graph executor.
        """
        logger.info("Building and compiling the agent swarm executor...")

        # Smol Agent for planning and simple tasks
        smol_agent_factory = SMOLAgent(llm=self.llm, checkpointer=self.checkpointer)
        smol_agent = await smol_agent_factory.build()
        smol_agent.name = "Smol_Agent"

        # Deep Research Agent for in-depth research tasks
        deep_agent_factory = DeepResearchAgent(llm=self.llm, checkpointer=self.checkpointer)
        deep_agent = await deep_agent_factory.build()
        deep_agent.name = "Deep_Research_Agent"

        # Tools Agent for executing tools
        tools_agent_factory = ToolsAgent(llm=self.llm, checkpointer=self.checkpointer)
        tools_agent = await tools_agent_factory.build()
        tools_agent.name = "Tools_Agent"

        # Create the swarm by defining the graph and transitions
        builder = create_swarm(
            [smol_agent, deep_agent, tools_agent],
            default_active_agent="Smol_Agent"
        )

        agent_executor = builder.compile(checkpointer=self.checkpointer)
        logger.info("Agent swarm executor compiled successfully.")
        return agent_executor


async def main():
    """Main function to demonstrate instantiating and running the agent swarm."""
    load_dotenv()
    if not (os.getenv("OPENAI_API_KEY") and os.getenv("TAVILY_API_KEY")):
        raise ValueError("API keys for OpenAI and Tavily must be set in the .env file.")

    # 1. Initialize the language model
    llm = ChatOpenAI(model="gpt-4o", temperature=0, streaming=True)

    # 2. Instantiate the agent factory with the LLM and a memory checkpointer
    memory = MemorySaver()
    agent_factory = MORGANA(llm=llm, checkpointer=memory)

    # 3. Build the agent executor by calling the build() method
    agent_executor = await agent_factory.build()

    # --- Running the Agent ---
    thread_id = "multi_agent_convo_1"
    run_config = {"configurable": {"thread_id": thread_id}}

    query = "Research the main causes and consequences of the 2008 financial crisis, then write a short summary report in Spanish."
    initial_input = SwarmState(
        messages=[HumanMessage(content=query)],
        # The 'team' state can be defined in your SwarmState TypedDict
        team={
            "active_agent": "Planner_Agent",
            "agents": ["Planner_Agent", "Deep_Research_Agent", "Tools_Agent"],
        }
    )


    logger.info(f"--- Running Agent Swarm for Thread '{thread_id}' with Query: '{query}' ---")

    try:
        async for chunk in agent_executor.astream(initial_input, config=run_config, recursion_limit=150):
            for key, value in chunk.items():
                if messages := value.get("messages"):
                    # The final message is usually the one we want to display
                    ai_msg = messages[-1]
                    if ai_msg.content:
                        logger.info(f"--- Agent Output ---\n{ai_msg.content}")

    except Exception as e:
        logger.error(f"An error occurred during agent execution: {e}", exc_info=True)


if __name__ == "__main__":
    asyncio.run(main())

