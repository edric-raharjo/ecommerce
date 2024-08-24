import streamlit as st
import os

from chat import WatsonxLLMHandler 

# Initialize the WatsonxLLMHandler class
llm_handler = WatsonxLLMHandler()

# Setup the app title
st.title('IISMA E-Commerce Website')

# Setup a prompt display to display the queries
prompt = st.chat_input("Drop your questions regarding e-commerce here")

# Store past messages 
if 'messages' not in st.session_state:
    st.session_state.messages = []

# Display past messages
for message in st.session_state.messages:
    st.chat_message(message['role']).markdown(message['content'])

# Show the prompt if the user asks it
if prompt:
    # Display prompt
    st.chat_message('user').markdown(prompt)
    # Store user prompt
    st.session_state.messages.append({'role': 'user', 'content': prompt})

    # Get the retrieval chain
    vector_store_loc = 'DB/ecommerce_300_index'  # Set the location of your vector store
    retrieval_chain = llm_handler.create_retrieval_chain(vector_store_loc, filetype='pdf', method='FAISS')

    # Get reply
    response = retrieval_chain.invoke({"input": prompt})['answer']

    # Display reply
    st.chat_message('assistant').markdown(response)
    # Store replies
    st.session_state.messages.append({'role': 'assistant', 'content': response})
