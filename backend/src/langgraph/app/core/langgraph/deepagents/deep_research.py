from typing import Literal
import asyncio
import os
import logging
from typing import Literal, Optional, List, Dict, Any
from pydantic import BaseModel, Field
from langchain.tools import StructuredTool
from dotenv import load_dotenv
from tavily import TavilyClient
from langchain_core.messages import HumanMessage
from langchain_core.runnables import Runnable
from langgraph.checkpoint.base import BaseCheckpointSaver
from langgraph.graph.state import CompiledStateGraph
from langchain_core.language_models.base import BaseLanguageModel
from langgraph.checkpoint.memory import MemorySaver

from src.langgraph.app.core.langgraph.deepagents import create_deep_agent
from src.langgraph.app.core.langgraph.swarm import create_handoff_tool
from .sub_agent import SubAgent

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(name)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

load_dotenv()  # Load environment variables from a .env file if present

# It's best practice to initialize the client once and reuse it.
tavily_client = TavilyClient(api_key=os.environ["TAVILY_API_KEY"])


transfer_to_Smol_Agent = create_handoff_tool(
    agent_name="Smol_Agent",
    description="Transfer the user to the Smol_Agent to answer basic questions and implement the solution to the user's request.",
)


transfer_to_Tools_Agent = create_handoff_tool(
    agent_name="Tools_Agent",
    description="Transfer the user to the Tools_Agent to perform practical tasks that may require specific toolsets like sports, travel, google, weather, or more advanced tools and implement the solution to the user's request.",
) 


from langchain_tavily import TavilySearch

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


# Search tool to use to do research
def internet_search(
    query: str,
    max_results: int = 5,
    topic: Literal["general", "news", "finance"] = "general",
    include_raw_content: bool = False,
):
    """Run a web search"""
    search_docs = tavily_client.search(
        query,
        max_results=max_results,
        include_raw_content=include_raw_content,
        topic=topic,
    )
    return search_docs

base_tools = [internet_search, tavily_search_tool, transfer_to_Smol_Agent, transfer_to_Tools_Agent]

SUB_RESEARCH_PROMPT = """You are a dedicated researcher. Your job is to conduct research based on the users questions.

Conduct thorough research and then reply to the user with a detailed answer to their question

only your FINAL answer will be passed on to the user. They will have NO knowledge of anything except your final message, so your final report should be your final message!"""

RESEARCH_SUB_AGENT = {
    "name": "research-agent",
    "description": "Used to research more in depth questions. Only give this researcher one topic at a time. Do not pass multiple sub questions to this researcher. Instead, you should break down a large topic into the necessary components, and then call multiple research agents in parallel, one for each sub question.",
    "prompt": SUB_RESEARCH_PROMPT,
    "tools": ["internet_search"],
}

SUB_CRITIQUE_PROMPT = """You are a dedicated editor. You are being tasked to critique a report.

You can find the report at `final_report.md`.

You can find the question/topic for this report at `question.txt`.

The user may ask for specific areas to critique the report in. Respond to the user with a detailed critique of the report. Things that could be improved.

You can use the search tool to search for information, if that will help you critique the report

Do not write to the `final_report.md` yourself.

Things to check:
- Check that each section is appropriately named
- Check that the report is written as you would find in an essay or a textbook - it should be text heavy, do not let it just be a list of bullet points!
- Check that the report is comprehensive. If any paragraphs or sections are short, or missing important details, point it out.
- Check that the article covers key areas of the industry, ensures overall understanding, and does not omit important parts.
- Check that the article deeply analyzes causes, impacts, and trends, providing valuable insights
- Check that the article closely follows the research topic and directly answers questions
- Check that the article has a clear structure, fluent language, and is easy to understand.
"""

CRITIQUE_SUB_AGENT = {
    "name": "critique-agent",
    "description": "Used to critique the final report. Give this agent some information about how you want it to critique the report.",
    "prompt": SUB_CRITIQUE_PROMPT,
}


# Prompt prefix to steer the agent to be an expert researcher
RESEARCH_INSTRUCTIONS = """You are an expert researcher. Your job is to conduct thorough research, and then write a polished report.

The first thing you should do is to write the original user question to `question.txt` so you have a record of it.

Use the research-agent to conduct deep research. It will respond to your questions/topics with a detailed answer.

When you think you enough information to write a final report, write it to `final_report.md`

You can call the critique-agent to get a critique of the final report. After that (if needed) you can do more research and edit the `final_report.md`
You can do this however many times you want until are you satisfied with the result.

Only edit the file once at a time (if you call this tool in parallel, there may be conflicts).

Here are instructions for writing the final report:

<report_instructions>

CRITICAL: Make sure the answer is written in the same language as the human messages! If you make a todo plan - you should note in the plan what language the report should be in so you dont forget!
Note: the language the report should be in is the language the QUESTION is in, not the language/country that the question is ABOUT.

Please create a detailed answer to the overall research brief that:
1. Is well-organized with proper headings (# for title, ## for sections, ### for subsections)
2. Includes specific facts and insights from the research
3. References relevant sources using [Title](URL) format
4. Provides a balanced, thorough analysis. Be as comprehensive as possible, and include all information that is relevant to the overall research question. People are using you for deep research and will expect detailed, comprehensive answers.
5. Includes a "Sources" section at the end with all referenced links

You can structure your report in a number of different ways. Here are some examples:

To answer a question that asks you to compare two things, you might structure your report like this:
1/ intro
2/ overview of topic A
3/ overview of topic B
4/ comparison between A and B
5/ conclusion

To answer a question that asks you to return a list of things, you might only need a single section which is the entire list.
1/ list of things or table of things
Or, you could choose to make each item in the list a separate section in the report. When asked for lists, you don't need an introduction or conclusion.
1/ item 1
2/ item 2
3/ item 3

To answer a question that asks you to summarize a topic, give a report, or give an overview, you might structure your report like this:
1/ overview of topic
2/ concept 1
3/ concept 2
4/ concept 3
5/ conclusion

If you think you can answer the question with a single section, you can do that too!
1/ answer

REMEMBER: Section is a VERY fluid and loose concept. You can structure your report however you think is best, including in ways that are not listed above!
Make sure that your sections are cohesive, and make sense for the reader.

For each section of the report, do the following:
- Use simple, clear language
- Use ## for section title (Markdown format) for each section of the report
- Do NOT ever refer to yourself as the writer of the report. This should be a professional report without any self-referential language. 
- Do not say what you are doing in the report. Just write the report without any commentary from yourself.
- Each section should be as long as necessary to deeply answer the question with the information you have gathered. It is expected that sections will be fairly long and verbose. You are writing a deep research report, and users will expect a thorough answer.
- Use bullet points to list out information when appropriate, but by default, write in paragraph form.

REMEMBER:
The brief and research may be in English, but you need to translate this information to the right language when writing the final answer.
Make sure the final answer report is in the SAME language as the human messages in the message history.

Format the report in clear markdown with proper structure and include source references where appropriate.

<Citation Rules>
- Assign each unique URL a single citation number in your text
- End with ### Sources that lists each source with corresponding numbers
- IMPORTANT: Number sources sequentially without gaps (1,2,3,4...) in the final list regardless of which sources you choose
- Each source should be a separate line item in a list, so that in markdown it is rendered as a list.
- Example format:
  [1] Source Title: URL
  [2] Source Title: URL
- Citations are extremely important. Make sure to include these, and pay a lot of attention to getting these right. Users will often use these citations to look into more information.
</Citation Rules>
</report_instructions>

You have access to a few tools.

## `internet_search`

Use this to run an internet search for a given query. You can specify the number of results, the topic, and whether raw content should be included.
"""

# # Create the agent
# agent = create_deep_agent(
#     [internet_search],
#     research_instructions,
#     subagents=[critique_sub_agent, research_sub_agent],
#     checkpointer=MemorySaver(),
# ).with_config({"recursion_limit": 1000})



# --- Refactored Agent Factory ---
class DeepResearchAgent:
    """
    A factory class for creating a multi-agent deep research system.

    This class encapsulates the configuration for a complex research agent,
    which uses sub-agents for research and critique to produce a polished,
    comprehensive report.
    """

    def __init__(self, llm: Optional[BaseLanguageModel] = None, checkpointer: Optional[BaseCheckpointSaver] = None, tools: Optional[List] = None, sub_agents: Optional[List[SubAgent]] = None):
        """
        Initializes the deep research agent factory.

        Args:
            model: The language model to use for the agent.
            checkpointer: An optional LangGraph checkpointer for state persistence.
                          If None, a new in-memory saver will be used.
        """
        self.model = llm
        self.checkpointer = checkpointer if checkpointer is not None else MemorySaver()
        self.tools = base_tools if tools is None else tools
        self.sub_agents = [CRITIQUE_SUB_AGENT, RESEARCH_SUB_AGENT] if sub_agents is None else sub_agents
        logger.info("DeepResearchAgent factory initialized.")

    async def build(self) -> CompiledStateGraph:
        """
        Builds and compiles the deep research agent graph.

        Returns:
            A compiled LangGraph runnable (agent executor) ready for execution.
        """
        logger.info("Building the deep research agent executor...")

        agent_executor = create_deep_agent(
            tools=self.tools,
            model=self.model if self.model else None,
            instructions=RESEARCH_INSTRUCTIONS,
            subagents=self.sub_agents,
            checkpointer=self.checkpointer,
        ).with_config({"recursion_limit": 1000})

        # agent_executor.name = "Deep_Research_Agent"
        logger.info("Deep research agent executor built successfully.")
        return agent_executor


async def main():
    """Main function to demonstrate the DeepResearchAgent factory."""
    if not os.getenv("TAVILY_API_KEY") or not os.getenv("OPENAI_API_KEY"):
        raise ValueError("TAVILY_API_KEY and OPENAI_API_KEY must be set in the .env file.")

    # 1. Set up persistence for the conversation
    memory = MemorySaver()

    # 2. Instantiate the agent factory with the checkpointer
    agent_factory = DeepResearchAgent(checkpointer=memory)

    # 3. Build the agent executor
    agent_executor = agent_factory.build()

    # 4. Define the research task and run the agent
    thread_config = {"configurable": {"thread_id": "deep-research-thread-2"}}
    query = "What were the main causes and consequences of the 2008 financial crisis? Write the report in Spanish."

    initial_input = [HumanMessage(content=query)]

    logger.info(f"--- Running Deep Research Agent for query: '{query}' ---")

    # Use astream_events to get a detailed stream of the agent's actions
    async for event in agent_executor.astream_events(initial_input, config=thread_config, version="v1"):
        kind = event["event"]
        if kind == "on_chat_model_stream":
            content = event["data"]["chunk"].content
            if content:
                # Print the LLM's thinking and output as it's generated
                print(content, end="", flush=True)
        elif kind == "on_tool_end":
            tool_name = event['name']
            tool_output = event['data'].get('output')
            print(f"\n\n--- Finished Tool Call: {tool_name} ---")
            # You can optionally print the full tool output for debugging
            # print(tool_output)
            print("---")


if __name__ == "__main__":
    asyncio.run(main())