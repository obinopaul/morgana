import asyncio
import os
import logging
from typing import List, Sequence, TypedDict, Annotated, Literal, Optional
from dotenv import load_dotenv

from langchain_core.messages import BaseMessage, ToolMessage, AIMessage, HumanMessage, SystemMessage
from langchain_core.language_models.chat_models import BaseChatModel
from langchain_openai import ChatOpenAI
from langgraph.graph import StateGraph, END, CompiledGraph
from langgraph.graph.message import add_messages
from langgraph.prebuilt import ToolNode
from langgraph.checkpoint.base import BaseCheckpointer

# Import local base tools and the system prompt
from basetools import base_tools
from prompts import SMOL_AGENT_PROMPT

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


class SMOLAgent:
    """
    A robust, simplified autonomous agent that uses a predefined set of local,
    base tools. It does not connect to any external MCP servers.
    """

    def __init__(self, llm_provider: type[BaseChatModel], model_name: str, max_steps: int = 10, checkpointer: Optional[BaseCheckpointer] = None):
        """
        Initializes the agent, its configuration, and its local tools.

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
        
        # Directly load local tools upon initialization
        self.tools = base_tools
        self.tool_node = ToolNode(self.tools)
        logger.info(f"SMOLAgent initialized with {len(self.tools)} local tools: {[tool.name for tool in self.tools]}")

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
        Builds the uncompiled LangGraph StateGraph, ready for compilation.
        """
        workflow = StateGraph(AgentState)
        workflow.add_node("agent", self._call_model)
        workflow.add_node("tools", self._call_tools)

        workflow.set_entry_point("agent")
        workflow.add_conditional_edges("agent", self._should_continue)
        workflow.add_edge("tools", "agent")

        return workflow

    def get_executor(self, config: Optional[AgentConfig] = None) -> CompiledGraph:
        """
        Builds and compiles the graph into a runnable executor, applying configurations.
        """
        logger.info("Building and compiling the SMOLAgent executor...")
        agent_graph = self.build_graph()
        
        config = config or {}
        
        agent_executor = agent_graph.compile(
            checkpointer=self.checkpointer,
            interrupt_before=config.get("interrupt_before", []),
            interrupt_after=config.get("interrupt_after", [])
        )
        logger.info("SMOLAgent executor compiled successfully.")
        return agent_executor

    async def arun(self, query: str, thread_id: str, config: Optional[AgentConfig] = None):
        """
        Asynchronously runs the agent with a query, thread_id, and optional config.
        """
        agent_executor = self.get_executor(config)

        run_config = {"configurable": {"thread_id": thread_id}}

        initial_input = {
            "messages": [
                SystemMessage(content=SMOL_AGENT_PROMPT),
                HumanMessage(content=query)
            ],
            "remaining_steps": self.max_steps,
        }

        logger.info(f"--- Running Agent for Thread '{thread_id}' with Query: '{query}' ---")
        
        final_state = None
        async for chunk in agent_executor.astream(initial_input, config=run_config, recursion_limit=150):
            final_state = chunk
            for key, value in chunk.items():
                if key == "agent" and value.get('messages'):
                    ai_msg = value['messages'][-1]
                    if ai_msg.tool_calls:
                        tool_names = ", ".join([call['name'] for call in ai_msg.tool_calls])
                        logger.info(f"Agent requesting tool(s): {tool_names}")
                elif key == "tools" and value.get('messages'):
                    tool_msg = value['messages'][-1]
                    logger.info(f"Tool executed. Result: {str(tool_msg.content)[:100]}...")

        if final_state:
            final_message = final_state.get("agent", {}).get("messages", [])[-1]
            if not final_message.tool_calls:
                 logger.info(f"\n--- Final Answer ---\n{final_message.content}")


async def main():
    """Main function to instantiate and run the SMOLAgent."""
    from langgraph.checkpoint.memory import MemorySaver

    if not (os.getenv("OPENAI_API_KEY") and os.getenv("TAVILY_API_KEY")):
        raise ValueError("API keys for OpenAI and Tavily must be set in the .env file.")

    memory = MemorySaver()
    agent = SMOLAgent(llm_provider=ChatOpenAI, model_name="gpt-4o", checkpointer=memory)
    thread_id = "smol_convo_789"

    query = "What is the current time and what are the top 3 news headlines in Oklahoma City today?"
    await agent.arun(query, thread_id=thread_id)


if __name__ == "__main__":
    asyncio.run(main())
