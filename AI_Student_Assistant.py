import streamlit as st
from langchain_openai import ChatOpenAI
from langchain.memory import ConversationBufferMemory
from langchain.tools import tool
from langchain.agents import AgentExecutor, Tool
from langchain_core.prompts import ChatPromptTemplate
from langgraph.graph import StateGraph, END, START
from langchain_community.tools.tavily_search import TavilySearchResults
from typing_extensions import TypedDict
from dotenv import load_dotenv
import os
# Load environment variables and set OpenAI API key
load_dotenv()
api_key = os.environ["OPENAI_API_KEY"] 
os.environ["TAVILY_API_KEY"] = os.getenv("TAVILY_API_KEY")

# Setup LLM
llm = ChatOpenAI(model="gpt-4o-mini", api_key=api_key)

# Memory setup
memory = ConversationBufferMemory(return_messages=True)

# Define the state structure
class State(TypedDict):
    query: str
    category: str
    response: str

# Web Search Tool

@tool
def web_search(state: State) -> str:
    """Perform a real-time web search using the Tavily API and fetch accurate information."""
    search_query = state.get("query", "").strip()
    if not search_query:
        return "Please provide a valid search query."
    
    # Initialize Tavily Search
    tavily_search = TavilySearchResults(max_results=5)
    try:
        # Perform the search
        search_results = tavily_search.invoke(search_query)
        
        if not search_results:
            return "No relevant results were found. Please try rephrasing your query."
        
        # Format and return results
        formatted_results = "\n".join([f"{i+1}. {result}" for i, result in enumerate(search_results)])
        response = f"Here are the most relevant results for your query:\n{formatted_results}"
    except Exception as e:
        response = f"An error occurred while performing the web search: {str(e)}"
    
    # Save response to state and memory
    state["response"] = response
    memory.save_context({"Human": search_query}, {"AI": response})
    return response

@tool
def provide_motivational_quote(state: State) -> str:
    """Provide a motivational quote to encourage the student."""
    motivational_quote = "The only way to do great work is to love what you do. – Steve Jobs"
    state["response"] = motivational_quote
    memory.save_context({"Human": state["query"]}, {"AI": motivational_quote})
    return motivational_quote



# Tools
tools = [
    Tool(name="web_search", func=web_search, description="Search the web for information using Tavily API."),
    Tool(name="provide_motivational_quote", func=provide_motivational_quote, description="Provide motivational quotes."),
    
    
]

# Agent Executor
agent = AgentExecutor(agent=llm, tools=tools, verbose=True)

def use_tool(state: State) -> State:
    """Use LangChain tools to handle specific tasks."""
    query = state["query"]
    try:
        response = agent.run(query)
        state["response"] = response
    except Exception as e:
        state["response"] = f"An error occurred: {str(e)}"
    return state

# Categorize queries
def categorize_query(state: State) -> State:
    """Categorize the query into relevant categories."""
    prompt = ChatPromptTemplate.from_template(
        """
        Categorize the following query into one of these categories:
        'Study Plan', 'Study Tips', 'Complaints', 'General Advice', 'Web Search', or 'Use Tools'.
        Query: {query}

        Conversation History:
        {history}
        """
    )
    chain = prompt | llm
    category = chain.invoke({"query": state["query"], "history": memory.buffer}).content
    state["category"] = category
    memory.save_context({"Human": state["query"]}, {"AI": category})
    return state


# Routing logic
def route_query(state: State) -> str:
    """Route the query based on its category."""
    category = state.get("category", "")
    if category == "Study Plan":
        return "handle_study_plan"
    elif category == "Study Tips":
        return "handle_study_tips"
    elif category == "Complaints":
        return "handle_complaints"
    elif category == "Use Tools":
        return "use_tool"
    
    else:
        return "handle_general_advice"

# Handle study tips
def handle_study_tips(state: State) -> State:
    """Provide study tips based on the user's query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide study tips or techniques for the following query: {query}"
    )
    chain = prompt | llm
    study_tips = chain.invoke({"query": state["query"], "history": memory.buffer}).content
    state["response"] = study_tips
    memory.save_context({"Human": state["query"]}, {"AI": study_tips})
    return state

# Handle study plans
def handle_study_plan(state: State) -> State:
    """Provide advice for creating a study plan."""
    prompt = ChatPromptTemplate.from_template(
        "Help the user create a study plan based on the following query and study plan table: {query}\n\nConversation History:\n{history}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "history": memory.buffer}).content
    state["response"] = response
    memory.save_context({"Human": state["query"]}, {"AI": response})
    return state

# Handle complaints
def handle_complaints(state: State) -> State:
    """Help the user submit complaints."""
    prompt = ChatPromptTemplate.from_template(
        "Help the user frame and submit a formal complaint for the following query: {query}\n\nConversation History:\n{history}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "history": memory.buffer}).content
    state["response"] = response
    memory.save_context({"Human": state["query"]}, {"AI": response})
    return state

# Handle general advice
def handle_general_advice(state: State) -> State:
    """Provide general advice for the user's query."""
    prompt = ChatPromptTemplate.from_template(
        "Provide general advice or suggestions for this query: {query}\n\nConversation History:\n{history}"
    )
    chain = prompt | llm
    response = chain.invoke({"query": state["query"], "history": memory.buffer}).content
    state["response"] = response
    memory.save_context({"Human": state["query"]}, {"AI": response})
    return state



# Workflow setup
workflow = StateGraph(State)
workflow.add_node("categorize_query", categorize_query)
workflow.add_node("handle_study_plan", handle_study_plan)
workflow.add_node("handle_study_tips", handle_study_tips)
workflow.add_node("handle_general_advice", handle_general_advice)
workflow.add_node("handle_complaints", handle_complaints)
workflow.add_node("use_tool", use_tool)
workflow.add_edge(START, "categorize_query")
workflow.add_conditional_edges(
    "categorize_query", route_query,
    [ "use_tool", "handle_study_plan", "handle_general_advice", "handle_study_tips", "handle_complaints"], END
)

app = workflow.compile()

# Streamlit UI
st.set_page_config(page_title="Student Assistant Chatbot", page_icon=":books:", layout="wide")
st.title("🎓 Student Assistant Chatbot")
st.write("Welcome! I'm here to help you with your study needs, motivational quotes, web search, and more. Just type your query below.")
# Streamlit Sidebar
st.sidebar.title("📚 Assistant Menu")
# User Navigation Options
st.sidebar.header("🔍 Navigation")
st.sidebar.markdown("""
Use the options below to explore:
- **Start New Chat**: Clear your current conversation.
- **View Tools**: Learn about the capabilities of this assistant.
- **Settings**: Customize your experience.
""")
# Descriptions
st.sidebar.header("🛠️ Tools")
st.sidebar.markdown("""
- **Web Search**: Find real-time information.
- **Motivational Quotes**: Get inspired with a quote.
- **Study Tips**: Learn effective study techniques.
- **Study Plans**: Create tailored study schedules.
- **Complaints**: Frame and submit formal complaints.
""")
# Settings Section
st.sidebar.header("⚙️ Settings")
def update_sidebar_theme():
    st.session_state.sidebar_dark_mode = not st.session_state.get("sidebar_dark_mode", False)
# Initialize the sidebar dark mode state
if "sidebar_dark_mode" not in st.session_state:
    st.session_state.sidebar_dark_mode = False  # Default to light mode
# Sidebar toggle for Dark Mode
sidebar_dark_mode = st.sidebar.checkbox(
    "🌙 Dark Mode",
    value=st.session_state.sidebar_dark_mode,
    on_change=update_sidebar_theme,
)
# Apply styles dynamically to the sidebar based on the selected theme
if st.session_state.sidebar_dark_mode:
    st.markdown(
        """
        <style>
        .stSidebar {
            background-color: #2e3b4e;  /* Darker background */
            color: #ffffff;             /* Light text */
        }
        .stSidebar button {
            background-color: #4a6073;  /* Subtle button color */
            color: #ffffff;
        }
        .stSidebar button:hover {
            background-color: #657d91; /* Hover effect */
        }
        .stSidebar .st-checkbox > div:first-child {
            background-color: #657d91;  /* Checkbox background in dark mode */
        }
        .stCheckbox{
            color: #ffffff;  /* Change label text to white in dark mode */
        }

         
        </style>
        """,
        unsafe_allow_html=True,
    )
else:
    st.markdown(
        """
        <style>
        .stSidebar {
            background-color: #f8f9fa;  /* Light background */
            color: #000000;             /* Dark text */
        }
        .stSidebar button {
            background-color: #007bff;  /* Bright button color */
            color: #ffffff;
        }
        .stSidebar button:hover {
            background-color: #0056b3; /* Hover effect */
        }
        .stSidebar h1, .stSidebar h2, .stSidebar h3, .stSidebar h4, .stSidebar h5, .stSidebar h6 {
            color: #333333;             /* Darker headings */
        }
        
        .stSidebar .st-checkbox > div:first-child {
            background-color: #ffffff;  /* Checkbox background in light mode */
        }
        .stCheckbox {
            color: #000000;  /* Change label text to black in light mode */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )


st.sidebar.markdown("""
Adjust the theme, reset the conversation, or change notification preferences.
""")
# Initialize conversation history
if "history" not in st.session_state:
    st.session_state.history = []

# Function to display chat history
def display_chat():
    # Clear chat UI before redisplaying
    for idx, message in enumerate(st.session_state.history):
        role = message["role"]
        if role == "user":
            st.chat_message("user").markdown(message["content"])
        elif role == "assistant":
            st.chat_message("assistant").markdown(message["content"])

# Reset Conversation
if st.sidebar.button("🔄 Reset Conversation"):
    st.session_state.history.clear()
    st.sidebar.success("Conversation reset successfully!")

# Chat Input Handler
query = st.chat_input("💬 Type your question here...")
if query:
    # Append the user's input to the history
    st.session_state.history.append({"role": "user", "content": query})
    
    # Generate response
    state = {"query": query, "category": "", "response": ""}
    result = app.invoke(state)
    # Append AI's response to history
    st.session_state.history.append({"role": "assistant", "content": result["response"]})

# Display chat history after both messages are added
display_chat()

