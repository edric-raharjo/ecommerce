import streamlit as st
import random
import time
import math

# Assuming DummyRetriever is defined in the same file or imported from another file
class DummyRetriever:
    def __init__(self):
        self.choices = [
            "This is just a dummy answer, nothing more to it.",
            "This is just a dummy answer, created to meet the word requirement. It provides no real information, simply serving as a placeholder in this example.",
            "This is just a dummy answer, carefully crafted to reach exactly fifty words. It contains no meaningful content or value, only fulfilling the word count. Please disregard any attempt to find substance in this placeholder, as it merely exists to demonstrate a specific format in this list.",
            "This is just a dummy answer, structured to meet the one hundred word count. It does not contain any meaningful or valuable information, instead serving solely as a placeholder within this list. The purpose of this response is purely illustrative, showing how one might construct a response that is exactly one hundred words long. Despite the length, it lacks any depth or substance, making it clear that it is intended only to demonstrate the required word count. Please disregard the content, as it does not aim to provide any real insight or understanding of a particular topic."
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

# CSS
st.markdown("""
    <style>
    div.stSpinner > div {
        text-align:center;
        align-items: center;
        justify-content: center;
    }
    </style>""", unsafe_allow_html=True)

# Setup the app title
st.title('IISMA E-Commerce Website')

with st.sidebar:
    st.header("Welcome to the E-Comm Chatbot")
    st.write("Go ahead and ask some questions, or try one these below")
    init_prompt_1 = st.button("What is 10+9?")
    init_prompt_2 = st.button("What's updog?")
    init_prompt_3 = st.button("Can I be your boyfriend?")


# Setup a prompt display to display the queries
if init_prompt_1:
    prompt = "What is 10+9?"
elif init_prompt_2 :
    prompt = "What's updog?"
elif init_prompt_3 :
    prompt = "Can I be your boyfriend?"
else:
    prompt = st.chat_input('Enter your text')

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
    
    with st.spinner("Loading"):
        time.sleep(random.choice([0.2,0.3]))

    # Display reply
    st.chat_message('assistant').markdown(response)
    # Store replies
    st.session_state.messages.append({'role': 'assistant', 'content': response})

    if init_prompt_1 or init_prompt_2 or init_prompt_3:
        prompt = st.chat_input('Enter your text')
