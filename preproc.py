from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.document_loaders import PyPDFLoader
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.embeddings import HuggingFaceInferenceAPIEmbeddings
from langchain_community.embeddings import CohereEmbeddings
import os
# from dotenv import load_dotenv
import streamlit as st
# load_dotenv()
class Datapreproc:
    def __init__(self,filepath  , chunk_size=700 , overlap= 50, emb_model="embed-english-v3.0"):
        self.filepath = filepath
        self.chunk_size = chunk_size
        self.overlap = overlap
        self.emb_model = emb_model
    
    def load_docs(self,file_path):
        """load all docs from src/data"""
        # loader = PyPDFLoader("./data/acsbr-015.pdf")
        loader = PyPDFLoader(file_path)
        self.docs = loader.load()
        return self.docs
    
    def splits(self):
        """splits the documents into sizeable chunks"""
        text_splitter = RecursiveCharacterTextSplitter(chunk_size = self.chunk_size, chunk_overlap = self.overlap)
        self.splits = text_splitter.split_documents(self.docs)
        return self.splits
    
    
    def create_vector_store(self, splits):
        """Creates a vector store using Cohere embeddings API"""
        os.environ["COHERE_API_KEY"] = st.secrets["COHERE_API_KEY"]
        # Initialize Cohere embeddings
        embeddings = CohereEmbeddings(model=self.emb_model,user_agent="langchain")
        
        # Create and return the vector store
        vector_store = FAISS.from_documents(splits, embeddings)
        return vector_store
    
    def forward(self):
        """loads filepath->loads_docs->splits into chunks->creates a vector store"""
        self.load_docs()
        self.splits()
        vector_store = self.create_vector_store(self.splits)
        return vector_store
    




