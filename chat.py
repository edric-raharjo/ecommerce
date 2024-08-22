# Imports
import os
import pickle
from dotenv import load_dotenv

# Langchain imports
from langchain.document_loaders import PyPDFLoader
from langchain.indexes import VectorstoreIndexCreator
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain.chains import create_retrieval_chain
from langchain.chains.combine_documents import create_stuff_documents_chain
from langchain_core.prompts import ChatPromptTemplate
from langchain_huggingface import HuggingFaceEmbeddings

# WatsonxLLM 
from langchain_ibm import WatsonxLLM

# load .env file
load_dotenv()

## Get API KEY
os.environ["WATSONX_APIKEY"] = os.getenv('API_KEY')

# Set LLM Parameters
parameters = {
    "decoding_method": "sample",
    "max_new_tokens": 200,
    "temperature": 0.5
}

# Create LLM instance
llm = WatsonxLLM(
    model_id = "meta-llama/llama-2-13b-chat",
    url = os.getenv('URL_IBM'),
    project_id = os.getenv('PROJECT_ID'),
    params=parameters
)

"""RAG PART BELOW"""

# Doc to Vector function [TODO BROKEN]
def load_doc(doc_loc, index_loc, filetype='txt'):
    # Define a path to save/load the vector index
    index_file = index_loc

    # Check if the index already exists
    if os.path.exists(index_file):
        # Load the saved index
        with open(index_file, 'rb') as f:
            index = pickle.load(f)
    else:
        # Open the txt file
        if filetype == 'txt':
            with open(doc_loc, 'r', encoding='utf-8') as file:
                doc = [Document(page_content=file.read())]

            # Create vector database from the doc using HF embeddings
            index = VectorstoreIndexCreator(
                embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
                text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
            ).from_documents(doc)

        # Open the pdf file
        elif filetype == 'pdf':
            loader = [PyPDFLoader(doc_loc)]

            # Create vector database from the loader using HF embeddings
            index = VectorstoreIndexCreator(
                embedding=HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2'),
                text_splitter=RecursiveCharacterTextSplitter(chunk_size=100, chunk_overlap=0)
            ).from_loaders(loader)
        
        # Save the index for future use
        with open(index_file, 'wb') as f:
            pickle.dump(index, f)

    # Return the vectorDB
    return index

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
index = load_doc(doc_loc)

retriever = index.vectorstore.as_retriever()

system_prompt = (
    "Use the given context to answer the question. "
    "If you don't know the answer, say you don't know. "
    "Use three sentence maximum and keep the answer concise. "
    "Context: {context}"
)
prompt = ChatPromptTemplate.from_messages(
    [
        ("system", system_prompt),
        ("human", "{input}"),
    ]
)
question_answer_chain = create_stuff_documents_chain(llm, prompt)
chain = create_retrieval_chain(retriever, question_answer_chain)

