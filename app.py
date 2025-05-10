import streamlit as st
import os
if not os.path.exists("data"):
    os.makedirs("data")
from langchain_core.messages import HumanMessage, AIMessage, ToolMessage
from main import build_graph

class DecisionLogger:
    def __init__(self):
        self.steps = []
        self.tools_used = set()
        self.rag_contexts = []
        
    def add_step(self, description, tool=None):
        step_num = len(self.steps) + 1
        self.steps.append(f"Step {step_num}: {description}")
        if tool:
            self.tools_used.add(tool)
    
    def add_rag_context(self, context):
        self.rag_contexts.append(context)
    
    def get_summary(self):
        tools_list = list(self.tools_used)
        summary = [
            "  ├── Steps: "
        ]
        
        for i, step in enumerate(self.steps):
            if i == len(self.steps) - 1:
                summary.append(f"  │      └── {step}")
            else:
                summary.append(f"  │      ├── {step}")
                
        summary.extend([
            f"  ├── Number of steps: {len(self.steps)}",
            "    ├── Tools:"
        ])
        
        for i, tool in enumerate(tools_list):
            if i == len(tools_list) - 1:
                summary.append(f"  │      └── {i+1}. {tool}")
            else:
                summary.append(f"  │      ├── {i+1}. {tool}")
                
        summary.append(f"  └── Number of tools: {len(tools_list)}")
        
        return "\n".join(summary)

def extract_tool_name(message):
    if hasattr(message, "tool_call_id") and message.tool_call_id:
        return message.name
    elif hasattr(message, "tool_calls") and message.tool_calls:
        return message.tool_calls[0]["name"]
    return None

def detect_rag_usage(messages):
    for message in messages:
        tool_name = extract_tool_name(message)
        if tool_name == "RAG_with_sources":
            return True
    return False

def extract_rag_context(messages):
    contexts = []
    for i, message in enumerate(messages):
        if (isinstance(message, ToolMessage) and 
            hasattr(message, "name") and 
            message.name == "RAG_with_sources"):
            # Get the context from the tool response
            contexts.append(message.content)
    return contexts

def main():
    st.set_page_config(page_title="RAG Agent Dashboard", layout="wide")
    st.title("RAG-Powered Agent Q&A Knowledge Assistant")
    uploaded_file = st.file_uploader("Upload a PDF document", type="pdf")
    fname = uploaded_file.name if uploaded_file is not None else None

# Logic to save the uploaded file
    if uploaded_file is not None:
        file_path = f"./data/{fname}"
        with open(file_path, "wb") as f:
            f.write(uploaded_file.getbuffer())
        st.success(f"File uploaded and saved as: {fname}")
    # Initialize the graph once
        if "graph" not in st.session_state or st.session_state.get("last_file") != fname:
            with st.spinner("Initializing system..."):
                st.session_state.graph = build_graph(fname)
                st.session_state.messages = []  
                st.session_state.logger = DecisionLogger()
                st.session_state.last_file = fname
    else:
        st.info("Please upload a PDF to begin")
    
    # Create tabs for main interaction and logs
    tab1, tab2 = st.tabs(["Main", "Decision Logs"])
    
    with tab1:
        # Initialize message history in session state if not already present
        chat_container = st.container()
    
    # Second: Add the chat input AFTER declaring the container
        user_input = st.chat_input(
            "Ask a question",
            disabled=(uploaded_file is None)
        )
    
    # Third: Fill the container with chat history
    with chat_container:
        # Initialize message history in session state if not already present
        if "messages" not in st.session_state:
            st.session_state.messages = []
            
        # Display message history
        for message in st.session_state.messages:
            if isinstance(message, HumanMessage):
                with st.chat_message("user"):
                    st.markdown(message.content)
            elif isinstance(message, AIMessage):
                with st.chat_message("assistant"):
                    st.markdown(message.content)
        
        # Process user input
        if user_input:
            # Add user message to history
            user_message = HumanMessage(content=user_input)
            st.session_state.messages.append(user_message)
            
            # Display user message
            with st.chat_message("user"):
                st.markdown(user_input)
            
            # Create a new logger for this query
            logger = DecisionLogger()
            logger.add_step(f"Received query: '{user_input}'")
            
            # Process message with agent
            with st.spinner("Thinking..."):
                logger.add_step("Analyzing query and determining approach")
                state = st.session_state.graph.invoke({"messages": [user_message]})
                response = state["messages"][-1].content
                
                # Detect which tools were used
                tools_used = []
                for message in state["messages"]:
                    tool_name = extract_tool_name(message)
                    if tool_name:
                        tools_used.append(tool_name)
                        logger.add_step(f"Using tool: {tool_name}", tool_name)
                
                # Check if RAG tool was used
                rag_used = detect_rag_usage(state["messages"])
                if rag_used:
                    rag_contexts = extract_rag_context(state["messages"])
                    for ctx in rag_contexts:
                        logger.add_rag_context(ctx)
                
                logger.add_step("Generating final response")
            
                
                # Store logger in session state
                st.session_state.logger = logger
            
            # Display AI response
            ai_message = AIMessage(content=response)
            st.session_state.messages.append(ai_message)
            with st.chat_message("assistant"):
                st.markdown(response)
            
            # Display tools used
            if tools_used:
                st.subheader("Tools Used")
                st.write(", ".join(set(tools_used)))
            
            # Display RAG contexts if applicable
            if rag_used:
                st.subheader("Retrieved Context Snippets")
                with st.expander("Show Context", expanded=True):
                    for i, context in enumerate(logger.rag_contexts):
                        st.text_area(f"Context {i+1}", context, height=200)
    
    with tab2:
        if hasattr(st.session_state, "logger"):
            st.text(st.session_state.logger.get_summary())

if __name__ == "__main__":
    main()