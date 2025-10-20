from typing import List, Annotated, TypedDict
from langchain_core.messages import BaseMessage
from langgraph.graph import StateGraph, END
from langgraph.checkpoint.memory import MemorySaver
from agents import create_router_agent, create_gemini_agent, html_demo_agent_node
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_community.chat_models import ChatOllama

class AgentState(TypedDict):
    messages: Annotated[List[BaseMessage], lambda x, y: x + y]

# --- Decision logic for the router ---
def route_logic(state: AgentState) -> str:
    """
    Determines the next node based on the router's decision.
    """
    last_message = state["messages"][-1]
    # If the last message is a function call, route to the corresponding agent.
    if hasattr(last_message, "additional_kwargs") and "function_call" in last_message.additional_kwargs:
        function_name = last_message.additional_kwargs["function_call"]["name"]
        return function_name
    # Otherwise, terminate the graph.
    return END

# --- Graph-Creation ---
def build_graph():
    """
    Create and compile the LangGraph-graph
    """
    # llm = ChatOllama(model="llama3", temperature=0)
    llm_gemini = ChatGoogleGenerativeAI(model="gemini-2.5-flash-lite", temperature=0.8, convert_system_message_to_human=True)
    
    # Init agents
    router_agent = create_router_agent(llm_gemini)
    gemini_agent = create_gemini_agent(llm_gemini)

    # Graph definition
    workflow = StateGraph(AgentState)

    workflow.add_node("router", router_agent)
    workflow.add_node("gemini_agent", gemini_agent)
    workflow.add_node("html_demo_agent", html_demo_agent_node)

    workflow.set_entry_point("router")
    
    # A conditional edge that decides where to go based on the `route_logic`.
    workflow.add_conditional_edges(
        "router",
        route_logic,
        {
            "gemini_agent": "gemini_agent",
            "html_demo_agent": "html_demo_agent",
            END: END
        }
    )
    
    workflow.add_edge("gemini_agent", END)
    workflow.add_edge("html_demo_agent", END)

    # Compile the graph and add memory
    memory = MemorySaver()
    graph = workflow.compile(checkpointer=memory)
    
    return graph
