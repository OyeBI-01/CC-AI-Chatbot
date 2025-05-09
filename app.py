import streamlit as st
import os
from dotenv import load_dotenv
from components.chat_interface import ChatInterface
from services.bot_service import BotService
from utils.config import setup_page_config

# Load environment variables
load_dotenv()

# FastAPI backend URL (optional)
FASTAPI_URL = "http://localhost:8000"

def main():
    # Setup page configuration
    setup_page_config()
    
    # Initialize session state
    if "messages" not in st.session_state:
        st.session_state.messages = []
    
    if "uploaded_doc" not in st.session_state:
        st.session_state.uploaded_doc = None
    
    if "bot_service" not in st.session_state:
        st.session_state.bot_service = BotService(FASTAPI_URL)

    # Display header
    st.title("Mark Musk - CreditChek API Assistant")
        

    st.markdown("""
    ### Welcome to Mark Musk
    Ask about CreditChek API integration! Try these:
    - How do I set up webhooks for CreditChek transaction updates?
    - How do I configure the RecovaPRO SDK?
    - Show me a Python example for identity verification.
    """)

    # Initialize chat interface with shared BotService
    chat_interface = ChatInterface(fastapi_url=FASTAPI_URL, bot_service=st.session_state.bot_service)
    
    # Render chat interface
    chat_interface.render()

if __name__ == "__main__":
    main()