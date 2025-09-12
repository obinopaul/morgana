"""This file contains the LangGraph Agent/workflow and interactions with the LLM."""

from typing import (
    Any,
    AsyncGenerator,
    Dict,
    Literal,
    Optional,
)

from asgiref.sync import sync_to_async
from langchain_core.messages import (
    BaseMessage,
    ToolMessage,
    convert_to_openai_messages,
)
from datetime import datetime # <<< IMPORTED
import copy # <<< IMPORTED
from langchain_openai import ChatOpenAI
from langfuse.langchain import CallbackHandler
from langgraph.checkpoint.postgres.aio import AsyncPostgresSaver
from langgraph.graph import StateGraph, END
from langchain_core.messages import BaseMessage, HumanMessage
from langgraph.graph.state import CompiledStateGraph
from langgraph.types import StateSnapshot
from openai import OpenAIError
from psycopg_pool import AsyncConnectionPool

from src.langgraph.app.core.config import (
    Environment,
    settings,
)
from src.langgraph.app.core.langgraph import MORGANA
from src.langgraph.app.core.logging import logger
from src.langgraph.app.core.metrics import llm_inference_duration_seconds
# from src.langgraph.app.core.prompts import SYSTEM_PROMPT
from src.langgraph.app.schemas import (
    GraphState,
    Message,
)
from src.langgraph.app.utils import (
    dump_messages,
    prepare_messages,
)



class LangGraphAgent:
    """Manages the MORGANA multi-agent system and its interactions with the application.

    This class handles the creation and management of the MORGANA workflow,
    including LLM interactions, database connections, and response processing.
    """

    def __init__(self):
        """Initialize the LangGraph Agent with necessary components."""
        # Use environment-specific LLM model
        self.llm = ChatOpenAI(
            model=settings.LLM_MODEL,
            temperature=settings.DEFAULT_LLM_TEMPERATURE,
            api_key=settings.LLM_API_KEY,
            max_tokens=settings.MAX_TOKENS,
            streaming=True, # Ensure streaming is enabled for MORGANA
            **self._get_model_kwargs(),
        )
        # self.tools_by_name = {tool.name: tool for tool in tools}
        self._connection_pool: Optional[AsyncConnectionPool] = None
        self._agent_executor: Optional[CompiledStateGraph] = None

        logger.info("llm_initialized_for_morgana", model=settings.LLM_MODEL, environment=settings.ENVIRONMENT.value)

    def _get_model_kwargs(self) -> Dict[str, Any]:
        """Get environment-specific model kwargs.

        Returns:
            Dict[str, Any]: Additional model arguments based on environment
        """
        model_kwargs = {}

        # Development - we can use lower speeds for cost savings
        if settings.ENVIRONMENT == Environment.DEVELOPMENT:
            model_kwargs["top_p"] = 0.8

        # Production - use higher quality settings
        elif settings.ENVIRONMENT == Environment.PRODUCTION:
            model_kwargs["top_p"] = 0.95
            model_kwargs["presence_penalty"] = 0.1
            model_kwargs["frequency_penalty"] = 0.1

        return model_kwargs

    async def _get_connection_pool(self) -> AsyncConnectionPool:
        """Get a PostgreSQL connection pool using environment-specific settings.

        Returns:
            AsyncConnectionPool: A connection pool for PostgreSQL database.
        """
        if self._connection_pool is None:
            try:
                # Configure pool size based on environment
                max_size = settings.POSTGRES_POOL_SIZE

                self._connection_pool = AsyncConnectionPool(
                    settings.POSTGRES_URL,
                    open=False,
                    max_size=max_size,
                    kwargs={
                        "autocommit": True,
                        "connect_timeout": 5,
                        "prepare_threshold": None,
                    },
                )
                await self._connection_pool.open()
                logger.info("connection_pool_created", max_size=max_size, environment=settings.ENVIRONMENT.value)
            except Exception as e:
                logger.error("connection_pool_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                # In production, we might want to degrade gracefully
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_connection_pool", environment=settings.ENVIRONMENT.value)
                    return None
                raise e
        return self._connection_pool
    
    async def _get_or_create_agent_executor(self) -> Optional[CompiledStateGraph]:
        """Create and compile the MORGANA agent executor if it doesn't exist.

        Returns:
            Optional[CompiledStateGraph]: The compiled MORGANA agent or None if initialization fails.
        """
        if self._agent_executor is None:
            try:
                connection_pool = await self._get_connection_pool()
                checkpointer = None
                if connection_pool:
                    checkpointer = AsyncPostgresSaver(connection_pool)
                    await checkpointer.setup()
                elif settings.ENVIRONMENT != Environment.PRODUCTION:
                    raise Exception("Connection pool initialization failed in a non-production environment.")

                # Instantiate the MORGANA factory with the LLM and checkpointer
                morgana_factory = MORGANA(llm=self.llm, checkpointer=checkpointer)

                # Build and compile the agent executor
                self._agent_executor = await morgana_factory.build()

                logger.info(
                    "morgana_agent_executor_created",
                    graph_name="MORGANA Multi-Agent Swarm",
                    environment=settings.ENVIRONMENT.value,
                    has_checkpointer=checkpointer is not None,
                )
            except Exception as e:
                logger.error("morgana_agent_creation_failed", error=str(e), environment=settings.ENVIRONMENT.value)
                if settings.ENVIRONMENT == Environment.PRODUCTION:
                    logger.warning("continuing_without_agent_executor")
                    return None
                raise e
        return self._agent_executor

    def _prepare_input_with_timestamp(self, messages: list[Message]) -> list[Message]:
        """Appends the current timestamp to the last human message."""
        if not messages or messages[-1].role != "user":
            return messages

        # Use deepcopy to avoid modifying the original messages list
        messages_with_timestamp = copy.deepcopy(messages)
        
        current_time = datetime.now().strftime("%Y-%m-%d %H:%M:%S CDT")
        timestamp_info = f"\n\n[Current Time: {current_time}]"
        
        # Append to the content of the last message
        messages_with_timestamp[-1].content += timestamp_info
        return messages_with_timestamp
    
    async def get_response(
        self,
        messages: list[Message],
        session_id: str,
        user_id: Optional[str] = None,
    ) -> list[dict]:
        """Get a response from the MORGANA agent.

        Args:
            messages (list[Message]): The messages to send to the agent.
            session_id (str): The session ID for conversation tracking.
            user_id (Optional[str]): The user ID for Langfuse tracking.

        Returns:
            list[dict]: The final response from the agent.
        """
        agent_executor = await self._get_or_create_agent_executor()
        if not agent_executor:
            raise RuntimeError("Agent executor could not be initialized.")

        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [CallbackHandler()],
            "metadata": {
                "user_id": user_id,
                "session_id": session_id,
                "environment": settings.ENVIRONMENT.value,
                "debug": False,
            },
            "recursion_limit": 150, # Set a robust recursion limit
        }

        # MORGANA expects a specific input structure, typically a list of BaseMessages.
        # The last message from the user is the primary input.
        # initial_input = {"messages": [HumanMessage(content=messages[-1].content)]}
        final_messages = self._prepare_input_with_timestamp(messages)
        initial_input = {"messages": dump_messages(final_messages)}

        try:
            with llm_inference_duration_seconds.labels(model=self.llm.model_name).time():
                 response = await agent_executor.ainvoke(initial_input, config)

            # Extract the final state from the last active agent's output
            # final_agent_output = list(response.values())[0]
            # return self.__process_messages(final_agent_output["messages"])
            return self.__process_messages(response["messages"])
        except Exception as e:
            logger.error(f"Error getting response from MORGANA: {str(e)}", session_id=session_id)
            raise e

    async def get_stream_response(
        self, messages: list[Message], session_id: str, user_id: Optional[str] = None
    ) -> AsyncGenerator[str, None]:
        """Get a streaming response from the MORGANA agent.

        Args:
            messages (list[Message]): The messages to send to the agent.
            session_id (str): The session ID for the conversation.
            user_id (Optional[str]): The user ID for the conversation.

        Yields:
            str: Tokens of the LLM response from the agent swarm.
        """
        agent_executor = await self._get_or_create_agent_executor()
        if not agent_executor:
            raise RuntimeError("Agent executor could not be initialized.")

        config = {
            "configurable": {"thread_id": session_id},
            "callbacks": [
                CallbackHandler(
                    environment=settings.ENVIRONMENT.value, debug=False, user_id=user_id, session_id=session_id
                )
            ],
            "recursion_limit": 150,
        }
        # initial_input = {"messages": [HumanMessage(content=messages[-1].content)]}
        final_messages = self._prepare_input_with_timestamp(messages)
        initial_input = {"messages": dump_messages(final_messages)}

        try:
            # The MORGANA swarm outputs dictionary chunks. We need to parse them.
            async for chunk in agent_executor.astream(initial_input, config):
                for agent_name, agent_output in chunk.items():
                    if output_messages := agent_output.get("messages"):
                        # Yield the content of the last message in the chunk
                        last_message = output_messages[-1]
                        if last_message and last_message.content and isinstance(last_message.content, str):
                            yield last_message.content
        except Exception as stream_error:
            logger.error("Error in MORGANA stream processing", error=str(stream_error), session_id=session_id)
            raise stream_error

    async def get_chat_history(self, session_id: str) -> list[Message]:
        """Get the chat history for a given thread ID from the checkpointer.

        Args:
            session_id (str): The session ID for the conversation.

        Returns:
            list[Message]: The chat history.
        """
        agent_executor = await self._get_or_create_agent_executor()
        if not agent_executor:
            logger.warning("Cannot get chat history, agent executor not initialized.")
            return []

        try:
            state: StateSnapshot = await sync_to_async(agent_executor.get_state)(
                config={"configurable": {"thread_id": session_id}}
            )
            return self.__process_messages(state.values.get("messages", [])) if state and state.values else []
        except Exception as e:
            logger.error("Failed to retrieve chat history", error=str(e), session_id=session_id)
            return []

    def __process_messages(self, messages: list[BaseMessage]) -> list[Message]:
        """Converts BaseMessages to a serializable format, filtering for user/assistant roles."""
        if not messages:
            return []
        openai_style_messages = convert_to_openai_messages(messages)
        return [
            Message(**message)
            for message in openai_style_messages
            if message["role"] in ["assistant", "user"] and message["content"]
        ]

    async def clear_chat_history(self, session_id: str) -> None:
        """Clear all chat history for a given thread ID from the database.

        Args:
            session_id: The ID of the session to clear history for.
        """
        try:
            conn_pool = await self._get_connection_pool()
            if not conn_pool:
                logger.warning("Cannot clear history, no connection pool available.")
                return

            async with conn_pool.connection() as conn:
                for table in settings.CHECKPOINT_TABLES:
                    try:
                        await conn.execute(f'DELETE FROM "{table}" WHERE thread_id = %s', (session_id,))
                        logger.info(f"Cleared {table} for session {session_id}")
                    except Exception as e:
                        logger.error(f"Error clearing {table} for session {session_id}", error=str(e))
                        # Continue to try clearing other tables
        except Exception as e:
            logger.error("Failed to clear chat history", error=str(e), session_id=session_id)
            raise