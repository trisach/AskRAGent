from langchain.chains import RetrievalQA
from preproc import Datapreproc
from langchain.prompts import PromptTemplate
from langchain.tools import Tool
from customTools import all_tools
from langchain_groq.chat_models import ChatGroq
from langgraph.graph import START, StateGraph, MessagesState
from langgraph.prebuilt import ToolNode, tools_condition
# from dotenv import load_dotenv
import streamlit as st
import os
# load_dotenv()
class Agent:
    def __init__(self,vector_store,llm):
        self.vector_store = vector_store
        # Set the Groq API key from Streamlit secrets
        os.environ["GROQ_API_KEY"] = st.secrets["GROQ_API_KEY"]
        self.llm = ChatGroq(model="qwen-qwq-32b", temperature=0)
        self.retriever = self.vector_store.as_retriever(search_type = "similarity",search_kwargs = {"k":3})
        prompt_template = """You are a helpful assisstant.You are here to help the user with your assisstance.Use the following pieces of context to answer the question at the end. Please follow the following rules:
        1. If you don't know the answer, don't try to make up an answer. Just say "I can't find the final answer."
        2. If you find the answer, write the answer in a concise way with five sentences maximum.
        3.Always cite the context in FULL to the user. In showing the context , do NOT use ellipses or summarization.
        {context}

        Question: {question}

        Helpful Answer:
        """
        PROMPT = PromptTemplate(
            template=prompt_template, input_variables=["context", "question"]
        )

        self.rag_chain = RetrievalQA.from_chain_type(llm = self.llm,
        chain_type = "stuff",
        retriever = self.retriever,
        return_source_documents = True,
        chain_type_kwargs = {"prompt":PROMPT})

        def rag_with_sources(query: str) -> str:
            """This is a RAG tool that also returns source documents."""
            result = self.rag_chain.invoke({'query': query})
            docs = result['source_documents']

            # Format the retrieved chunks
            chunks = []
            for i, doc in enumerate(docs, 1):
                # meta_src = doc.metadata.get('source', f'chunk_{i}')
                snippet = doc.page_content.strip()
                chunks.append(f"Citation {i}:\n{snippet}")
            full_context = "\n\n".join(chunks)

            # Use the system prompt to format the output
            formatted_output = PROMPT.format(
                context=full_context,
                question=query
            )
            def print_retrieved_chunks(self, chunks):
                """Print the retrieved chunks for transparency"""
                print("\n" + "="*50)
                print("="*50)
                print(f"Retrieved {len(chunks)} relevant chunks:")
                print("-"*50)
                for i, chunk in enumerate(chunks, 1):
                    print(f"CHUNK {i}:")
                    print(chunk.page_content.strip())
                    print(f"Source: {chunk.metadata.get('source', 'Unknown')}")
                    print("-"*50)
                print("="*50 + "\n")
            print(print_retrieved_chunks)
            return formatted_output

        self.RAG_tool = Tool(
            name="RAG_with_sources",
            func=rag_with_sources,
            description="Retrieve top-k chunks and answer using RAG, returning both.",
        )

    
def build_graph(uploaded_file):
    """Build the graph"""
    preproc = Datapreproc("data/")
    print("Loading and splitting documents...")
    # preproc.load_docs()
    preproc.load_docs(f"data/{uploaded_file}")
    splits = preproc.splits()
    print("Creating vector store...")
    vector_store = preproc.create_vector_store(splits)
    print("Vector store created.")
    print("Initializing agent system...")
    agent_system = Agent(vector_store, llm=None)
    print("System initialized.")


    # Add RAG tools
    tools = all_tools + [agent_system.RAG_tool]
    print("RAG-Powered Multi-Agent Q&A Assistant")
    print("Type 'exit' to quit.\n")

    # Bind tools to the LLM
    llm_with_tools = agent_system.llm.bind_tools(tools)

    def assistant(state: MessagesState):
        system_prompt = """You are an advanced AI assistant. Use the available tools to answer user questions. 
        Refer to previous parts of our conversation when appropriate.
        If you don't know the answer, say "I can't find the final answer." 
        For calculations and math operations:
        - ALWAYS use the appropriate tool (add, subtract, multiply, divide) rather than calculating yourself
        - Show your reasoning with the tool calls
        - Present the final result clearly
        Never attempt to perform calculations directly. Always use the provided tools.
        Be concise and helpful. 
        When using tools, explain your reasoning briefly."""
        
        # Create a new messages array with the system message at the beginning
        messages_with_system = [{"role": "system", "content": system_prompt}]
        
        # Add all the existing messages
        if isinstance(state["messages"], str):
            # If it's just a string (first message)
            messages_with_system.append({"role": "user", "content": state["messages"]})
        else:
            # If it's already a list of messages
            messages_with_system.extend(state["messages"])
        
        # Get response with the system prompt included
        result = llm_with_tools.invoke(messages_with_system)
        
        return {"messages": state["messages"] + [result]}

    # Create the graph
    builder = StateGraph(MessagesState)
    builder.add_node("assistant", assistant)
    builder.add_node("tools",ToolNode(tools))
    builder.add_edge(START, "assistant")

    # Add conditional edge: if tools_condition, go to tools, else finish
    builder.add_conditional_edges(
        "assistant",
        tools_condition
    )
    # After tools, always return to assistant
    builder.add_edge("tools", "assistant")

    return builder.compile()


if __name__ == "__main__":
    graph = build_graph("./data/")
    system_prompt = """You are an advanced AI assistant. Use the available tools to answer user questions. 
    Refer to previous parts of our conversation when appropriate.
    If you don't know the answer, say "I can't find the final answer." 
    For calculations and math operations:
    - ALWAYS use the appropriate tool (add, subtract, multiply, divide) rather than calculating yourself
    - Show your reasoning with the tool calls
    - Present the final result clearly
    Never attempt to perform calculations directly. Always use the provided tools.
    Be concise and helpful. 
    When using tools, explain your reasoning briefly."""
    while True:
        user_input = input("You: ")
        if user_input.lower() in ["exit", "quit"]:
            print("Goodbye!")
            break
        state = graph.invoke({"messages": user_input})
        response = state["messages"][-1]
        print(f"Assistant: {response.content}")
