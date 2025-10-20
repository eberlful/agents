import streamlit as st
from langchain_core.messages import AIMessage

def render_agent_output(agent_name: str, agent_output: dict):
    """
    Renders an agent's output in a collapsible Streamlit container.
    This function serves as a unified interface for display.
    
    Args:
        agent_name (str): The name of the agent.
        agent_output (dict): The output of the agent.
    """
    
    with st.expander(f"Agent: `{agent_name}`", expanded=True):
        
        # Extract the last message from the agent
        if "messages" in agent_output and agent_output["messages"]:
            last_message = agent_output["messages"][-1]
            
            # Render the text from the last message
            if last_message.content:
                st.write(last_message.content)

            # --- HTML-output ---
            # Check if a html-key is provided
            if isinstance(last_message, AIMessage) and "html" in last_message.additional_kwargs:
                html_content = last_message.additional_kwargs["html"]
                if html_content:
                    st.markdown("---")
                    st.subheader("HTML-Ergebnis")
                    st.markdown(html_content, unsafe_allow_html=True)
            
            # Shows which agent comes after
            if isinstance(last_message, AIMessage) and "function_call" in last_message.additional_kwargs:
                function_call = last_message.additional_kwargs["function_call"]
                st.info(f"**Action:** Call `{function_call['name']}` with arguments: `{function_call['arguments']}`")
        else:
            # Fallback, if the output has not the right/propter format
            st.write("Could not parse the agent output.")
            st.json(agent_output)