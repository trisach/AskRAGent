# RAG-Powered Agent Q&A Knowledge Assistant

A Streamlit-based intelligent document assistant that combines Retrieval-Augmented Generation (RAG) with an agentic workflow to provide accurate answers from documents while handling computational and dictionary requests via specialized tools.

## Architecture

```
┌─────────────────────────┐     ┌──────────────────────┐     ┌───────────────────┐
│      User Interface     │     │    Agent System      │     │  Data Processing  │
│     (Streamlit App)     │     │                      │     │                   │
│                         │     │  ┌────────────────┐  │     │  ┌─────────────┐  │
│  ┌─────────────────┐    │     │  │  StateGraph    │  │     │  │PDF Loading  │  │
│  │ Query Input     │    │     │  │  Orchestration │  │     │  │& Chunking   │  │
│  └─────────────────┘    │     │  └────────────────┘  │     │  └─────────────┘  │
│          │              │     │          │           │     │         │         │
│  ┌─────────────────┐    │     │  ┌───────┴───────┐   │     │  ┌─────────────┐  │
│  │ Response Display│◄───┼─────┼──┤ LLM (Groq)    │   │     │  │ Cohere      │  │
│  └─────────────────┘    │     │  └───────────────┘   │     │  │ Embeddings   │  │
│          ▲              │     │          ▲           │     │  └─────────────┘  │
│  ┌─────────────────┐    │     │  ┌───────┴───────┐   │     │         │         │
│  │ Decision Logs   │◄───┼─────┼──┤   Tools       │   │     │  ┌─────────────┐  │
│  └─────────────────┘    │     │  │ Condition     │   │     │  │ FAISS       │  │
│          ▲              │     │  └───────┬───────┘   │     │  │ Vector Store │  │
│  ┌─────────────────┐    │     │          │           │     │  └─────────────┘  │
│  │ Context Display │◄───┼─────┼──────────┘           │     │                   │
│  └─────────────────┘    │     │                      │     │                   │
└─────────────────────────┘     └──────────────────────┘     └───────────────────┘
                                          │
                                          ▼
                               ┌─────────────────────────┐
                               │     Custom Tools        │
                               │                         │
                               │  ┌──────────────────┐   │
                               │  │ Math Operations  │   │
                               │  │ (add, subtract,  │   │
                               │  │ multiply, etc.)  │   │
                               │  └──────────────────┘   │
                               │                         │
                               │  ┌──────────────────┐   │
                               │  │ Dictionary API   │   │
                               │  │ Lookup Tool      │   │
                               │  └──────────────────┘   │
                               │                         │
                               │  ┌──────────────────┐   │
                               │  │ RAG with Sources │   │
                               │  │ Tool            │   │
                               │  └──────────────────┘   │
                               └─────────────────────────┘
```

### Core Components

1. **Data Processing Pipeline** (`preproc.py`)
   - Document loading via PyPDFLoader
   - Chunking with RecursiveCharacterTextSplitter
   - Embedding generation with Cohere's `embed-english-v3.0` model
   - FAISS vector store for efficient similarity search

2. **Agent System** (`main.py`)
   - LangGraph-based agent orchestration
   - Conditional routing between RAG and tool execution
   - Groq's qwen-qwq-32b LLM integration for high-quality responses
   - Decision logging and transparent execution tracking

3. **Custom Tools** (`customTools.py`)
   - Mathematical operations (add, subtract, multiply, divide, modulus, power)
   - Dictionary lookups via external API
   - Tool selection based on query intent

4. **Web Interface** (`app.py`)
   - Streamlit-based UI with PDF upload capability
   - Interactive chat interface
   - Decision logs visualization
   - Context exploration for transparency

## Design Choices

1. **LangGraph for Orchestration**: Uses LanGraph's StateGraph for flexible and transparent agent workflow management.

2. **Tool-First Approach**: Routes calculation and definition requests to specialized tools rather than relying on the LLM's inherent capabilities, ensuring accuracy.

3. **Transparent RAG**: Returns and displays source context alongside answers, maintaining provenance and allowing users to verify information.

4. **Decision Logging**: Tracks each step of the agent's reasoning process, providing insights into how answers are derived.

5. **Modular Architecture**: Separates concerns between data preprocessing, agent logic, tool implementation, and UI for maintainability.

6. **Efficient Embedding**: Uses Cohere's state-of-the-art embedding model for high-quality semantic search.

## Running the Application

### Prerequisites

- Python 3.8+
- API keys for:
  - Groq (LLM provider)
  - Cohere (Embedding service)

### Setup

1. Clone the repository:
   ```bash
   git clone https://github.com/yourusername/rag-agent-assistant.git
   cd rag-agent-assistant
   ```

2. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

3. Create a `.streamlit/secrets.toml` file with your API keys:
   ```toml
   GROQ_API_KEY = "your-groq-api-key"
   COHERE_API_KEY = "your-cohere-api-key"
   ```

### Running the App

Start the Streamlit app:
```bash
streamlit run app.py
```

### Usage

1. Upload a PDF document using the file uploader
2. Ask questions in the chat interface:
   - For calculations: "Calculate 24 * 7" or "What is 144 divided by 12?"
   - For definitions: "Define quantum computing" or "What does serendipity mean?"
   - For document questions: Ask anything related to the uploaded document
