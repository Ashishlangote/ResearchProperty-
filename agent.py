from this import s
import uuid
import os
from dotenv import load_dotenv
from typing import Annotated
from typing_extensions import TypedDict
from langchain_core.prompts import ChatPromptTemplate
from langchain_core.messages import ToolMessage, AIMessage
from langchain_core.runnables import RunnableLambda, Runnable
from langchain_core.tools import tool
from langchain_groq import ChatGroq
from langgraph.checkpoint.memory import MemorySaver
from langgraph.graph import StateGraph, START
from langgraph.graph.message import AnyMessage, add_messages
from langgraph.prebuilt import ToolNode, tools_condition
from firecrawl import FirecrawlApp

load_dotenv()

app = FirecrawlApp(api_key=os.getenv("FIRECRAWL_API_KEY"))


@tool
def housing_search(city: str):
    """
    Search top residential projects in the given city from Housing.com.

    Args:
        city (str): The city to search for (e.g., "Pune").

    Returns:
        A list of dictionaries. Each dictionary includes:
        - project: Project name
        - location: Area/locality in the city
        - bhk_details: List of BHK type with price (e.g. 2 BHK - ₹1.2 Cr)
        - price_range: Overall min-max price range (optional)
        - link: Link to the full project page
    """
    try:
        city = city.lower().strip()
        url=f"https://housing.com/in/buy/{city}/{city}"
        result_markdown = app.scrape(url, formats=['markdown'])
        if result_markdown is not None:
            return str(result_markdown.markdown).strip().replace("\n\n", " ")[:9000]
    except Exception as e:
        return f"Failed to fetch listings for {city}. Error: {str(e)}"

def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls

    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease check the city name or try again later.",
                tool_call_id=tc["id"],
            )
            for tc in tool_calls
        ]
    }


class State(TypedDict):
    messages: Annotated[list[AnyMessage], add_messages]


class Assistant:
    def __init__(self, runnable: Runnable):
        self.runnable = runnable

    def __call__(self, state: State):
        while True:
            result = self.runnable.invoke(state)

            if not result.tool_calls and (
                not result.content
                or isinstance(result.content, list)
                and not result.content[0].get("text")
            ):
                messages = state["messages"] + [("user", "Respond with a real output.")]
                state = {**state, "messages": messages}
            else:
                break

        return {"messages": result}


class RealEstateAssistant:
    def __init__(self, thread_id: str = ""):
        load_dotenv()
        self.api_key = os.getenv("GOOGLE")
        self.llm = ChatGroq(model="llama-3.3-70b-versatile", temperature=0.7)
        self.tools = [housing_search]
        self.prompt = self._build_prompt()
        self.assistant = Assistant(self.prompt | self.llm.bind_tools(self.tools))
        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.thread_id = str(uuid.uuid4()) if thread_id is None else thread_id

    def _build_prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    '''You are a smart real estate assistant.
                    If a user asks about buying flats in a city, call the `housing_search` tool with that city name.
                    Only use the tool if the user explicitly mentions a city.
                    Once you get the data, show the top 3-5 listings in a user-friendly format including:
                    - For each project, display:
                      - Project name
                      - Location
                      - BHK types and prices
                      - Total price range
                      - Link to full listing

                    Note - Never mention where this data is sourced from or refer to external websites.
                    '''
                ),
                ("placeholder", "{messages}"),
            ]
        )

    def _build_graph(self):
        builder = StateGraph(State)
        builder.add_node("assistant", self.assistant)
        builder.add_node("tools", self._create_tool_node_with_fallback())
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        return builder.compile(checkpointer=self.memory)

    def _create_tool_node_with_fallback(self):
        return ToolNode(self.tools).with_fallbacks(
            [RunnableLambda(handle_tool_error)],
            exception_key="error"
        )

    def chat(self, message: str) -> str:
        config = {"configurable": {"thread_id": self.thread_id}}
        state = {"messages": [("user", message)]}
        events = self.graph.stream(state, config, stream_mode="messages")

        for event in events:
            msg = event[0]
            if isinstance(msg, AIMessage) and msg.content:
                yield msg.content


# assistant = RealEstateAssistant()
# print("💬 Real Estate Assistant Bot (type 'exit' to quit)\n")

# while True:
#     user_input = input("You: ")

#     if user_input.strip().lower() in {"exit", "quit"}:
#         print("👋 Goodbye!")
#         break

#     try:
#         response = assistant.chat(user_input)
#         print("Assistant:", response, "\n")
#     except Exception as e:
#         print("⚠️ Error:", e)
