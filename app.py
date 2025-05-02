# app.py
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
    
    # Sidebar for documentation
    with st.sidebar:
        st.header("Documentation Sources")
        
        # GitHub Repository
        st.subheader("GitHub Repository")
        github_repo = st.text_input(
            "GitHub Repository URL",
            value="git@github.com:ORIBData/docusaurus-.git",  
            placeholder="e.g., git@github.com:ORIBData/docusaurus-.git or https://github.com/ORIBData/docusaurus-"
        )
        branch = st.text_input("Branch", value="master")
        
        if st.button("Process GitHub Repository"):
            if github_repo:
                with st.spinner("Processing GitHub repository..."):
                    try:
                        st.session_state.bot_service.process_github_repo(github_repo, branch)
                        st.success("GitHub repository processed successfully!")
                    except ValueError as ve:
                        st.error(f"Error: {str(ve)}")
                    except Exception as e:
                        st.error(f"Unexpected error processing repository: {str(e)}")
            else:
                st.warning("Please enter a GitHub repository URL")
        
        st.divider()
        
        # File Upload
        st.subheader("Document Upload")
        st.write("Upload CreditChek documentation to improve responses.")
        
        uploaded_file = st.file_uploader("Upload documentation (PDF/TXT)", type=["pdf", "txt"])
        if uploaded_file is not None and (st.session_state.uploaded_doc != uploaded_file.name):
            st.session_state.uploaded_doc = uploaded_file.name
            
            if uploaded_file.type == "text/plain":
                file_content = uploaded_file.read().decode("utf-8", errors="ignore")
                st.write("**File Preview**: (First 500 characters)")
                st.text(file_content[:500])
                uploaded_file.seek(0)
            else:
                st.write("**File Preview**: PDF preview not available")
            
            with st.spinner("Processing documentation..."):
                progress = st.progress(0)
                try:
                    st.session_state.bot_service.process_uploaded_doc(uploaded_file)
                    progress.progress(100)
                    st.success(f"Documentation processed: {uploaded_file.name}")
                except Exception as e:
                    st.error(f"Error processing document: {str(e)}")
    
    st.markdown("""
    ### Welcome to Mark Musk
    Ask about CreditChek API integration! Try these:
    - How do I set up webhooks for CreditChek transaction updates?
    - How do I configure the RecovaPRO SDK?
    - Show me a Python example for identity verification.
    """)

    # Initialize chat interface
    chat_interface = ChatInterface(FASTAPI_URL)
    
    # Display chat messages
    chat_interface.display_chat_history()

    # Get user input
    if prompt := st.chat_input("Ask me about CreditChek API..."):
        chat_interface.handle_user_input(prompt)

if __name__ == "__main__":
    main()