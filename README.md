
# 🤖 Agentic AI Chatbot for Research Assistance

An interactive, autonomous chatbot powered by LLaMA 3.3 (via Groq API), LangGraph, and Streamlit. This project helps users with research tasks by enabling live web search, Python code execution, and memory-aware conversations — all within a sleek, chat-based UI.

---

## 🚀 Features

- 🔍 **Web Search Integration** via Tavily API  
- 🧮 **Python REPL Execution** for code-based queries, data analysis, and visualization  
- 🧠 **Memory-aware Conversations** using LangGraph’s `MemorySaver`  
- 📊 **Dynamic Matplotlib Plot Rendering** inside the chat  
- 🪄 **Tool-Calling Agent** built with LangChain’s `create_tool_calling_agent()`  
- 🧹 **Resettable Sessions** with persistent session ID and chat memory  

---

## 🛠️ Tech Stack

- **[Streamlit](https://streamlit.io/)** – UI Framework  
- **[LangChain](https://www.langchain.com/)** – LLM Orchestration  
- **[Groq API](https://console.groq.com/)** – LLaMA 3.3 70B LLM  
- **[Tavily](https://www.tavily.com/)** – Web Search API  
- **[Matplotlib](https://matplotlib.org/)** – Visualization  
- **[LangGraph](https://github.com/langchain-ai/langgraph)** – Memory Checkpointing  
- **Python REPL** – Live Python Code Execution  

---

## 📸 Screenshots

<img src="docs/chat_interface.png" alt="Chat UI with Plots" width="500">
<img src="docs/tool_calling.png" alt="Agent using Python and Search Tools" width="500">

---

## 📦 Installation

1. **Clone the repository**
   ```bash
   git clone https://github.com/your-username/agentic-ai-chatbot.git
   cd agentic-ai-chatbot

2. **Create a virtual environment**
    ```bash
    python -m venv venv
    source venv/bin/activate  # On Windows: venv\Scripts\activate
    ```
3. **Set up environment variables**
    ```bash
    TAVILY_API_KEY="your_tavily_key"
    GROQ_API_KEY="your_groq_key"
    ```
4. **Run the app**
    ```bash
    streamlit run app.py
    ```

## 💡 How It Works

- User sends a query via chat.

- Agent determines if tools like web search or Python REPL are needed.

- Based on the response type, it may fetch web results, run Python code, or both.

- Responses are streamed to the user in real-time.

- Any generated plots are captured and persisted using in-session memory.

## 🧪 Example Prompts
- ```"Plot a sine wave using Python."```

- ```"Search for the latest research on quantum computing."```

- ```"What does the ROC curve represent in ML?"```

- ```"Run some Python code to calculate the mean of a list."```

- ```"Show me pie chart top 5 most used programming laguages in 2025"``` 
