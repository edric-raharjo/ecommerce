# Imports
import os
import pickle
import torch
import faiss
from uuid import uuid4
from dotenv import load_dotenv

# Langchain imports
from langchain_community.document_loaders import TextLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import RetrievalQA
from langchain.document_loaders import PyPDFLoader
from langchain_community.docstore.in_memory import InMemoryDocstore
from langchain_community.vectorstores import FAISS
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain.chains import create_retrieval_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain import hub

# WatsonxLLM
from langchain_ibm import WatsonxLLM
from langchain_huggingface import HuggingFaceEmbeddings

# load .env file
load_dotenv()

## Get API KEY
os.environ["WATSONX_APIKEY"] = os.getenv('API_KEY')

# Set LLM Parameters
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 300,
    "temperature": 0.95
}

# Create LLM instance
llm = WatsonxLLM(
    model_id = "meta-llama/llama-2-70b-chat",
    url = os.getenv('URL_IBM'),
    project_id = os.getenv('PROJECT_ID'),
    params=parameters
)

"""RAG PART BELOW"""
# Load Embeddings 
embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')

# Doc to Vector function 
def load_doc(doc_loc, vector_store_loc, filetype='txt', method='reg'):
    # Define a path to save/load the vector index
    vector_store_file = vector_store_loc

    # Check if the index already exists
    if os.path.exists(vector_store_file):
        # Load the saved index
        try:
            with open(vector_store_file, 'rb') as f:
                vector_store = pickle.load(f)
            return vector_store
        except PermissionError:
            # When Permission Error it means it's a FAISS database, use the load_local to get the FAISS DB
            global embeddings
            vector_store = FAISS.load_local(vector_store_loc, embeddings, allow_dangerous_deserialization=True)
    else:
        # Open the txt file
        if filetype == 'txt':
            with open(doc_loc, 'r', encoding='utf-8') as file:
                doc = [Document(page_content=file.read())]

            # Create vector database from the doc using HF embeddings
            vector_store = VectorstoreIndexCreator(
                embedding=embeddings,
                text_splitter=RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
            ).from_documents(doc)

            # Save the index for future use
            with open(vector_store_file, 'wb') as f:
                pickle.dump(index, f)

        # Open the pdf file
        elif filetype == 'pdf':
            loader = PyPDFLoader(doc_loc)

            # regular vector store
            if method == 'reg':
              # Change loader to list
              loader = [loader]

              # Create vector database from the loader using HF embeddings
              vector_store = VectorstoreIndexCreator(
                  embedding=embeddings,
                  text_splitter=RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
              ).from_loaders(loader)

              # Save the index for future use
              with open(vector_store_file, 'wb') as f:
                  pickle.dump(index, f)

            # FAISS 
            elif method == 'FAISS':
              # Get page_content 
              pages = loader.load_and_split()

              # Create the Vector Store
              # index = faiss.IndexFlatL2(len(embeddings.embed_query("hello world")))
              vector_store = FAISS.from_documents(pages, embeddings)

              # Save the FAISS
              vector_store.save_local(vector_store_loc)

    # Return the vectorDB
    return vector_store

# Function to load the saved index
def load_saved_index(index_file):
    # Check if the index file exists
    if os.path.exists(index_file):
        # Load the saved index
        with open(index_file, 'rb') as f:
            index = pickle.load(f)
        return index
    else:
        raise FileNotFoundError(f"Index file {index_file} not found. Please create the index first.")

# get document location
doc_loc = os.getenv('DOC_LOC')

# create the vectorDB
# index = load_saved_index(doc_loc)
index = load_doc(doc_loc,'DB/ecommerce_300_index','pdf','FAISS')

# load the retriever
retriever = index.as_retriever(
    search_type="similarity",
    search_kwargs={"k": 1},
)

# Load the Prompt Template
retrieval_qa_chat_prompt = ChatPromptTemplate.from_template(
    """
    Answer the following question based only on the provided context. 
    Think step by step before providing a detailed answer. 
    <context>
    {context}
    </context>
    Question: {input}
    """
)

# Create the chain
combine_docs_chain = create_stuff_documents_chain(
    llm, retrieval_qa_chat_prompt 
)

# Add the retriever
retrieval_chain = create_retrieval_chain(retriever, combine_docs_chain)

# TODO Fix the stupid answers

# Ask a response
# response = retrieval_chain.invoke({"input": "What is Lift Percentage?"})
# Print it
# print(response)