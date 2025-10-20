import streamlit as st
from langchain_core.messages import HumanMessage, AIMessage
from graph_builder import build_graph
from agent_tools import render_agent_output
import os
from dotenv import load_dotenv

load_dotenv()

st.set_page_config(
    page_title="Multi-Agent Chat",
    page_icon="🤖",
    layout="wide"
)

st.title("🤖 Multi-Agenten Chat")

if "messages" not in st.session_state:
    st.session_state.messages = [AIMessage(content="Hallo! How can I help you?")]

# Show chat history
for msg in st.session_state.messages:
    if isinstance(msg, AIMessage):
        st.chat_message("assistant").write(msg.content)
    elif isinstance(msg, HumanMessage):
        st.chat_message("user").write(msg.content)

# Handle user input
if prompt := st.chat_input("Please type in your question..."):
    st.session_state.messages.append(HumanMessage(content=prompt))
    st.chat_message("user").write(prompt)

    try:
        graph = build_graph()
        
        inputs = {"messages": [HumanMessage(content=prompt)]}
        
        final_response_placeholder = st.chat_message("assistant").empty()
        final_response = ""

        with st.spinner("The agents are preparing your response..."):
            # Execute the graph and stream the events
            for output in graph.stream(inputs, stream_mode="values"):
                agent_name = list(output.keys())[0]
                agent_output = output[agent_name]
                
                # Render the agent output
                render_agent_output(agent_name, agent_output)

                # Collect final answer
                if "messages" in agent_output and isinstance(agent_output["messages"][-1], AIMessage):
                    final_response = agent_output["messages"][-1].content

        # Show the final answer in the chat
        if final_response:
             final_response_placeholder.markdown(final_response)
             st.session_state.messages.append(AIMessage(content=final_response))
        else:
            final_response_placeholder.error("No final answer could be generated.")

    except Exception as e:
        st.error(f"Ein Fehler ist aufgetreten: {e}")
        st.exception(e)
