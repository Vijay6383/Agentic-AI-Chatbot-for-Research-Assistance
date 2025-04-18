import streamlit as st
import os
from langchain_community.tools.tavily_search import TavilySearchResults
from langgraph.checkpoint.memory import MemorySaver
from langchain.agents import Tool, AgentExecutor, create_tool_calling_agent
from langchain_experimental.utilities import PythonREPL
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_groq import ChatGroq
from matplotlib import pyplot as plt
from dotenv import load_dotenv
import io

# load env variable
load_dotenv()


# Set up the search tool and python repl tool
search_tool = TavilySearchResults(max_results=1, api_key=os.getenv("TAVILY_API_KEY"))
python_repl = PythonREPL()
repl_tool = Tool(
    name="python_repl",
    description="Executes Python code and returns the result.",
    func=python_repl.run,
)

# Initialize the LLM agent
llm = ChatGroq(
    model="llama-3.3-70b-versatile",
    temperature=0.7,
    max_tokens=1024,
    max_retries=2,
    api_key=os.getenv("GROQ_API_KEY"),
)

# Create the chat prompt template that includes chat history
prompt_template = ChatPromptTemplate.from_messages(
    [
        ("system", "You are a helpful assistant that can use tools."),
        MessagesPlaceholder(
            variable_name="chat_history"
        ),  # Add chat history placeholder
        ("human", "{input}"),
        MessagesPlaceholder("agent_scratchpad"),
    ]
)

# Define the available tools
tools = [search_tool, repl_tool]

# Create a unique session ID if it doesn't exist
if "session_id" not in st.session_state:
    st.session_state.session_id = f"session_{id(st.session_state)}"

# Create the agent
agent = create_tool_calling_agent(llm, tools, prompt_template)

# Initialize memory and agent executor if not in session state
if "memory" not in st.session_state:
    st.session_state.memory = MemorySaver()

if "agent_executor" not in st.session_state:
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, tools=tools, checkpointer=st.session_state.memory
    )

# Initialize conversation history using session state
if "messages" not in st.session_state:
    st.session_state.messages = []

# Also keep a formatted chat history for the LLM
if "chat_history" not in st.session_state:
    st.session_state.chat_history = []

# Add figure history to store plots with messages
if "figure_history" not in st.session_state:
    st.session_state.figure_history = []


##############################################
# HELPER FUNCTIONS FOR FIGURE PERSISTENCE
##############################################
def save_figure_to_session(fig, message_index):
    """Save a matplotlib figure to session state with reference to message index"""
    # Convert figure to bytes (PNG format)
    buf = io.BytesIO()
    fig.savefig(buf, format="png")
    buf.seek(0)
    # Store the figure data with message index
    st.session_state.figure_history.append(
        {"message_index": message_index, "image_data": buf.getvalue()}
    )


def display_figure_from_data(fig_data):
    """Display a figure from its saved binary data"""
    if "image_data" in fig_data:
        st.image(fig_data["image_data"])


##############################################
# RESET FUNCTION
##############################################
def reset_conversation():
    """Clear all chat history and figures"""
    st.session_state.messages = []
    st.session_state.chat_history = []
    st.session_state.figure_history = []
    # Generate a new session ID
    st.session_state.session_id = f"session_{id(st.session_state)}"
    # Reset memory by creating a new memory saver instance
    st.session_state.memory = MemorySaver()
    # Recreate the agent executor with the new memory
    st.session_state.agent_executor = AgentExecutor(
        agent=agent, tools=tools, checkpointer=st.session_state.memory
    )


##############################################
# STREAMLIT CHAT APP SETUP
##############################################

# Create a header with title on left and reset button on right
col1, col2 = st.columns([7, 1])  # Adjust ratio as needed
with col1:
    st.title("🐱‍👤AI Research Assistant")
    st.markdown("Llama 3.3  🛠 Web Search  🛠 Python Execution")
with col2:
    # Create a reset button with broom emoji aligned to the right
    if st.button("🧹", help="Clear chat history"):
        reset_conversation()
        st.success("Conversation has been reset!")

# Display previous conversation history with their figures
for i, message in enumerate(st.session_state.messages):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        # Display any figures associated with this message
        for fig_data in st.session_state.figure_history:
            if fig_data["message_index"] == i:
                display_figure_from_data(fig_data)

# Accept new user input
user_input = st.chat_input("Ask anything (you can generate python visuals) Eg: show me a pie chart of Top 5 populated country...")

st.html(
    """
<style>
    .stChatInput div {
        min-height: 60px
    }
</style>
    """
)

if user_input:
    # Append user input to history and display it
    st.session_state.messages.append({"role": "user", "content": user_input})
    # Also add to chat history for the LLM in the correct format
    st.session_state.chat_history.append(("human", user_input))

    with st.chat_message("user"):
        st.markdown(user_input)

    # Process the query with proper error handling
    with st.chat_message("assistant"):
        # Create a message container for streaming
        message_placeholder = st.empty()
        # Container for figures that might be generated during this response
        figure_container = st.container()

        max_attempts = 2
        attempts = 0
        success = False
        full_response = ""  # To accumulate the streamed response

        while attempts < max_attempts and not success:
            try:
                # Pass the chat history to the agent
                for step in st.session_state.agent_executor.stream(
                    {
                        "input": user_input,
                        "chat_history": st.session_state.chat_history,
                    },
                    {"configurable": {"thread_id": st.session_state.session_id}},
                ):
                    if "output" in step:
                        # Append the new chunk to the accumulated response
                        full_response += step["output"]
                        # Update the display with the accumulated content
                        message_placeholder.markdown(full_response)

                        # Check if any matplotlib figures were generated in this step
                        if plt.get_fignums():
                            with figure_container:
                                fig = plt.gcf()
                                st.pyplot(fig)
                                # Save the figure with current message index
                                save_figure_to_session(
                                    fig, len(st.session_state.messages)
                                )
                                plt.close(fig)
                success = True
            except Exception as error:
                if "Failed to call a function" in str(error):
                    attempts += 1
                    full_response = ""
                    continue
                else:
                    st.error(f"An unexpected error occurred: {error}")
                    break

        if not success:
            st.error(
                "Error persists after retries. Please adjust your prompt and try again."
            )

        # Append the final agent response to both history formats
        st.session_state.messages.append(
            {"role": "assistant", "content": full_response}
        )
        st.session_state.chat_history.append(("ai", full_response))

