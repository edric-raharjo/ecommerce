import streamlit as st
import pandas as pd
import numpy as np
import os

from chat import retrieval_chain

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
    st.session_state.messages.append({'role':'user','content':prompt})

    # Get reply
    response = retrieval_chain.invoke({"input": prompt})['answer']
    # Display reply
    st.chat_message('assistant').markdown(response)
    # Store replies
    st.session_state.messages.append({'role':'assistant','content':response})

