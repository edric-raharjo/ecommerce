# Imports
import os
import pickle
import torch
import faiss
from uuid import uuid4
from dotenv import load_dotenv
import random

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

class WatsonxLLMHandler:
    def __init__(self):
        # Load environment variables
        load_dotenv()
        self.api_key = os.getenv('API_KEY')
        self.url = os.getenv('URL_IBM')
        self.project_id = os.getenv('PROJECT_ID')
        self.doc_loc = os.getenv('DOC_LOC')

        # Set LLM parameters
        self.parameters = {
            "decoding_method": "sample",
            "max_new_tokens": 500,
            "temperature": 0.95
        }

        # Create LLM instance
        self.llm = WatsonxLLM(
            model_id="meta-llama/llama-3-1-70b-instruct",
            url=self.url,
            project_id=self.project_id,
            params=self.parameters
        )

        # Load Embeddings
        self.embeddings = HuggingFaceEmbeddings(model_name='all-MiniLM-L12-v2')

    def load_doc(self, doc_loc, vector_store_loc, filetype='txt', method='reg'):
        vector_store_file = vector_store_loc

        # Check if the index already exists
        if os.path.exists(vector_store_file):
            try:
                with open(vector_store_file, 'rb') as f:
                    vector_store = pickle.load(f)
                return vector_store
            except PermissionError:
                vector_store = FAISS.load_local(vector_store_loc, self.embeddings, allow_dangerous_deserialization=True)
        else:
            if filetype == 'txt':
                with open(doc_loc, 'r', encoding='utf-8') as file:
                    doc = [Document(page_content=file.read())]

                vector_store = VectorstoreIndexCreator(
                    embedding=self.embeddings,
                    text_splitter=RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
                ).from_documents(doc)

                with open(vector_store_file, 'wb') as f:
                    pickle.dump(vector_store, f)

            elif filetype == 'pdf':
                loader = PyPDFLoader(doc_loc)

                if method == 'reg':
                    loader = [loader]

                    vector_store = VectorstoreIndexCreator(
                        embedding=self.embeddings,
                        text_splitter=RecursiveCharacterTextSplitter(chunk_size=350, chunk_overlap=50)
                    ).from_loaders(loader)

                    with open(vector_store_file, 'wb') as f:
                        pickle.dump(vector_store, f)

                elif method == 'FAISS':
                    pages = loader.load_and_split()
                    vector_store = FAISS.from_documents(pages, self.embeddings)
                    vector_store.save_local(vector_store_loc)

        return vector_store

    def load_saved_index(self, index_file):
        if os.path.exists(index_file):
            with open(index_file, 'rb') as f:
                index = pickle.load(f)
            return index
        else:
            raise FileNotFoundError(f"Index file {index_file} not found. Please create the index first.")

    def create_retrieval_chain(self, vector_store_loc, filetype='pdf', method='FAISS'):
        index = self.load_doc(self.doc_loc, vector_store_loc, filetype, method)

        retriever = index.as_retriever(
            search_type="similarity",
            search_kwargs={"k": 3},
        )

        retrieval_qa_chat_prompt = ChatPromptTemplate.from_template(
            """
            Answer the following question based only on the provided context. 
            Think step by step before providing a detailed answer. 
            DO NOT WRITE A QUESTION, JUST GIVE THE ANSWER IN A DETAILED MANNER
            <context>
            {context}
            </context>
            Question: {input}
            """
        )

        combine_docs_chain = create_stuff_documents_chain(self.llm, retrieval_qa_chat_prompt)
        return create_retrieval_chain(retriever, combine_docs_chain)

# Ask a response
# response = retrieval_chain.invoke({"input": "What is Lift Percentage?"})
# Print it
# print(response)