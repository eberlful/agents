from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.output_parsers.openai_functions import JsonOutputFunctionsParser
from langchain_core.messages import AIMessage, BaseMessage, HumanMessage, ToolMessage
from typing import Literal

# --- Definition of agents (Tools) ---
functions = [
    {
        "name": "gemini_agent",
        "description": "Ein allgemeiner Konversations-Agent. Wird verwendet, wenn keine spezifische Aufgabe angefordert wird.",
        "parameters": {
            "type": "object",
            "properties": {
                "query": {
                    "type": "string",
                    "description": "Die Frage des Benutzers, die an den Gemini-Agenten weitergeleitet werden soll.",
                }
            },
            "required": ["query"],
        },
    },
    {
        "name": "html_demo_agent",
        "description": "Ein Demo-Agent, der eine HTML-Tabelle generiert. Benutze diesen, wenn der Benutzer explizit nach HTML fragt.",
        "parameters": {
            "type": "object",
            "properties": {
                "topic": {
                    "type": "string",
                    "description": "Das Thema, für das eine HTML-Tabelle erstellt werden soll.",
                }
            },
            "required": ["topic"],
        },
    },
]

# --- Router agent ---
# Decides which agent will be called next.
def create_router_agent(llm):
    """Erstellt den Router-Agenten, der den nächsten Schritt bestimmt."""
    system_prompt = (
        "Du bist ein intelligenter Router, der Benutzeranfragen an den am besten geeigneten Agenten weiterleitet. "
        "Du rufst IMMER eine der bereitgestellten Funktionen auf. Leite die Anfrage des Benutzers unverändert weiter."
    )
    
    prompt = ChatPromptTemplate.from_messages(
        [
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="messages"),
        ]
    )
    
    router = prompt | llm.bind_functions(functions=functions)
    return router

# --- Gemini agent ---
# A simple agent for general conversations.
def create_gemini_agent(llm):
    """Erstellt den Agenten für allgemeine Konversationen."""
    system_prompt = "Du bist ein hilfreicher KI-Assistent. Antworte dem Benutzer freundlich und informativ."
    prompt = ChatPromptTemplate.from_messages([
        ("system", system_prompt),
        MessagesPlaceholder(variable_name="messages"),
    ])
    agent = prompt | llm
    return agent

# --- HTML demo ggent ---
# This agent intentionally returns a mix of text and HTML to demonstrate the render function.
def html_demo_agent_node(state):
    """
    Ein Knoten, der eine formatierte HTML-Ausgabe simuliert.
    Dies ist ein Beispiel dafür, wie ein Agent strukturierte Daten zurückgeben kann.
    """
    topic = state["messages"][-1].additional_kwargs["function_call"]["arguments"]
    
    # Erstellen der HTML-Tabelle
    html_content = f"""
    <h4>Ergebnis-Tabelle für '{topic.get('topic', 'Unbekannt')}'</h4>
    <table border="1" style="width:100%; border-collapse: collapse; border-radius: 8px; overflow: hidden;">
      <tr style="background-color: #f2f2f2;">
        <th style="padding: 8px;">Spalte 1</th>
        <th style="padding: 8px;">Spalte 2</th>
      </tr>
      <tr>
        <td style="padding: 8px;">Daten A</td>
        <td style="padding: 8px;">123</td>
      </tr>
      <tr>
        <td style="padding: 8px;">Daten B</td>
        <td style="padding: 8px;">456</td>
      </tr>
    </table>
    """
    
    text_content = f"Hier ist die angeforderte HTML-Tabelle zum Thema '{topic.get('topic', 'Unbekannt')}'."
    
    return {
        "messages": [
            AIMessage(
                content=text_content, 
                additional_kwargs={"html": html_content}
            )
        ]
    }
