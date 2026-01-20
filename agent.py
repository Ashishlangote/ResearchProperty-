import uuid
import os
from dotenv import load_dotenv
from typing import Annotated, Optional, List
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

from utils import price_to_number
from utils import get_mongo_collection

load_dotenv()

@tool
def mongo_housing_search(
    city: str,
    locality: Optional[str] = None,
    bhk: Optional[str] = None,
    min_budget_lakh: Optional[float] = None,
    max_budget_lakh: Optional[float] = None,
    project_status: Optional[str] = None,
    amenities: Optional[List[str]] = None,
    developer: Optional[str] = None,
    rera_only: Optional[bool] = False
):
    """
    Search residential projects from MongoDB.
    """
    print("🏘️ Mongo Housing Search Called")

    collection = get_mongo_collection()

    # ---------------------------
    # BASE QUERY
    # ---------------------------
    query = {
        "location.city": {"$regex": f"^{city}$", "$options": "i"}
    }

    if locality:
        query["location.locality"] = {"$regex": locality, "$options": "i"}

    if project_status:
        query["project_status"] = {"$regex": project_status, "$options": "i"}

    if amenities:
        query["amenities"] = {"$all": amenities}

    if developer:
        query["developer"] = {"$regex": developer, "$options": "i"}

    if rera_only:
        query["rera_id"] = {"$exists": True, "$ne": ""}

    projects = list(collection.find(query))

    results = []

    for project in projects:

        # 🔹 Budget filter (PROJECT LEVEL)
        try:
            min_price = price_to_number(
                project["price_details"]["price_range"]["min_all_inclusive"]
            )
            max_price = price_to_number(
                project["price_details"]["price_range"]["max_all_inclusive"]
            )
        except Exception:
            continue

        if min_budget_lakh and max_price < min_budget_lakh:
            continue
        if max_budget_lakh and min_price > max_budget_lakh:
            continue

        # 🔹 BHK filter (CONFIG LEVEL)
        bhk_details = []

        for config in project.get("configuration", []):
            if bhk and bhk.lower() not in config.get("type", "").lower():
                continue

            bhk_details.append({
                "type": config["type"],
                "carpet_area_sqft": config.get("carpet_area_sqft")
            })

        if bhk and not bhk_details:
            continue

        results.append({
            "project_name": project["project_name"],
            "developer": project.get("developer"),
            "locality": project["location"].get("locality"),
            "city": project["location"].get("city"),
            "bhk_available": bhk_details,
            "price_range": project["price_details"]["price_range"],
            "project_status": project.get("project_status"),
            "possession": project.get("possession", {}).get("start"),
            "amenities": project.get("amenities", [])[:5],
            "rera_id": project.get("rera_id")
        })

    if not results:
        return "No matching projects found."

    return results[:5]


def handle_tool_error(state) -> dict:
    error = state.get("error")
    tool_calls = state["messages"][-1].tool_calls

    return {
        "messages": [
            ToolMessage(
                content=f"Error: {repr(error)}\nPlease try again with valid details.",
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
                state["messages"].append(("user", "Please give a valid response."))
            else:
                break

        return {"messages": result}


class RealEstateAssistant:
    def __init__(self, thread_id: Optional[str] = None):
        load_dotenv()

        self.llm = ChatGroq(
            model="openai/gpt-oss-20b",
            temperature=0
        )

        self.tools = [mongo_housing_search]
        self.prompt = self._build_prompt()
        self.assistant = Assistant(self.prompt | self.llm.bind_tools(self.tools))

        self.memory = MemorySaver()
        self.graph = self._build_graph()
        self.thread_id = thread_id or str(uuid.uuid4())

    def _build_prompt(self):
        return ChatPromptTemplate.from_messages(
            [
                (
                    "system",
                    """
                    You are a smart Indian real estate assistant.

                    CRITICAL TOOL RULES (FOLLOW STRICTLY):
                    - Tool arguments MUST match expected data types.
                    - min_budget_lakh and max_budget_lakh MUST be NUMBERS (float).
                    - NEVER pass strings like "1 crore", "80 lakh", "₹1 Cr".

                    BUDGET CONVERSION RULES:
                    - 1 crore = 100
                    - 1.2 crore = 120
                    - 85 lakh = 85

                    EXAMPLES:
                    - "under 1 crore" → max_budget_lakh = 100
                    - "below 85 lakh" → max_budget_lakh = 85
                    - "80L to 1.2Cr" → min_budget_lakh = 80, max_budget_lakh = 120

                    If you cannot confidently convert a budget to a NUMBER,
                    DO NOT pass that argument to the tool.

                    If user intent is about buying/searching flats,
                    YOU MUST call `mongo_housing_search`.

                    Never mention databases or data sources.
                    """
                ),
                ("placeholder", "{messages}"),
            ]
        )

    def _build_graph(self):
        builder = StateGraph(State)
        builder.add_node("assistant", self.assistant)
        builder.add_node("tools", self._create_tool_node())
        builder.add_edge(START, "assistant")
        builder.add_conditional_edges("assistant", tools_condition)
        builder.add_edge("tools", "assistant")
        return builder.compile(checkpointer=self.memory)

    def _create_tool_node(self):
        return ToolNode(self.tools).with_fallbacks(
            [RunnableLambda(handle_tool_error)],
            exception_key="error"
        )

    def chat(self, message: str):
        config = {"configurable": {"thread_id": self.thread_id}}
        state = {"messages": [("user", message)]}

        events = self.graph.stream(
            state,
            config,
            stream_mode="messages"
        )

        for event in events:
            msg = event[0]
            if isinstance(msg, AIMessage) and msg.content:
                yield msg.content


# assistant = RealEstateAssistant()
# for reply in assistant.chat("2 BHK flats in Wakad under 1 crore with swimming pool"):
#     print(reply)
