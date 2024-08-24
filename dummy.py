import streamlit as st
import random
import time

# Assuming DummyRetriever is defined in the same file or imported from another file
class DummyRetriever:
    def __init__(self):
        self.choices = [
            "This is a brief reply, exactly ten words long.",
            """This is a slightly longer response, containing exactly fifty words. It is meant
              to provide a bit more detail while still being concise and to the point. This type of response is ideal when you 
              need more context or clarity, but do not require an extensive explanation.""",
            """This is a much more detailed response, containing exactly one hundred words. It provides a 
            comprehensive overview of the subject at hand, ensuring that all relevant points are covered. 
            This type of response is ideal for situations where a thorough explanation is necessary. 
            It balances detail with readability, making it suitable for in-depth understanding without being overwhelming."""
        ]
    
    def invoke(self, data: dict) -> dict:
        if "input" in data:
            return {"answer": random.choice(self.choices)}
        else:
            return {"answer": "No input provided"}

# Initialize the DummyRetriever
dummy_retriever = DummyRetriever()

# Change the tab title 
st.set_page_config(
    page_title="E-comms",
    page_icon="🤑",
    layout="wide",
    initial_sidebar_state="expanded"
)
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

    # Get reply using DummyRetriever
    response = dummy_retriever.invoke({"input": prompt})['answer']
    time.sleep(random.choice([0.15,0.2,0.25]))
    # Display reply
    st.chat_message('assistant').markdown(response)
    # Store replies
    st.session_state.messages.append({'role': 'assistant', 'content': response})
