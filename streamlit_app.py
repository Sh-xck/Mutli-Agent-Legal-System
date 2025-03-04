import streamlit as st
from langchain.schema import HumanMessage, AIMessage
from agent import agent
import os
from transformers import pipeline

GROQ_API_KEY="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"
HUGGINGFACE_API_KEY="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"


os.environ["GROQ_API_KEY"] = GROQ_API_KEY
os.environ["HUGGINGFACE_API_KEY"] = HUGGINGFACE_API_KEY

def initialize_session_state():
    if "messages" not in st.session_state:
        st.session_state.messages = []

def main():
    st.title("Legal Assistant Chat")
    
    # Initialize session state for message history
    initialize_session_state()

    # Display chat messages from history
    for message in st.session_state.messages:
        if isinstance(message, HumanMessage):
            st.chat_message("user").write(message.content)
        else:
            st.chat_message("assistant").write(message.content)

    # Chat input
    if prompt := st.chat_input("What would you like to know about Indian law?"):
        # Add user message to chat history and display it immediately
        st.session_state.messages.append(HumanMessage(content=prompt))
        st.chat_message("user").write(prompt)
        
        # Get AI response
        with st.spinner("Thinking..."):
            response = agent(prompt)
            # Add AI message to chat history
            st.session_state.messages.append(AIMessage(content=response))
        
        # Check if user wants detailed response
        if "in detail" in prompt.lower():
            # Display the full response
            st.chat_message("assistant").write(response)
        else:
            # Display the response and generate summary
            st.chat_message("assistant").write(response)
            with st.spinner("Generating summary..."):
                conversation = "\n".join([
                    f"{'User' if isinstance(msg, HumanMessage) else 'Assistant'}: {msg.content}"
                    for msg in st.session_state.messages
                ])
                
                summarizer = pipeline("summarization", model="facebook/bart-large-cnn")
                summary = summarizer(conversation, max_length=250, min_length=80)[0]['summary_text']
                
                st.info("Summary:\n" + summary)

if __name__ == "__main__":
    main() 