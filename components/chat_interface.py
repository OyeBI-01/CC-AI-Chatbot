import streamlit as st
from datetime import datetime
from services.bot_service import BotService
import logging
import json
import re

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class ChatInterface:
    def __init__(self, fastapi_url: str, bot_service: BotService = None):
        logger.info(f"Initializing ChatInterface with fastapi_url: {fastapi_url}")
        self.bot_service = bot_service if bot_service else BotService(fastapi_url)
        self.fastapi_url = fastapi_url
        self.initialize_session_state()

    def initialize_session_state(self):
        logger.debug("Initializing session state")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "preferred_language" not in st.session_state:
            st.session_state.preferred_language = "Python"

    def display_chat_history(self):
        logger.debug("Displaying chat history")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_input(self):
        logger.debug("Checking for user input")
        # Language selector
        with st.sidebar:
            st.header("Preferences")
            st.session_state.preferred_language = st.selectbox(
                "Select Programming Language",
                ["Python", "NodeJS", "PHP Laravel", "GoLang"],
                index=["Python", "NodeJS", "PHP Laravel", "GoLang"].index(st.session_state.preferred_language)
            )
            logger.debug(f"Selected language: {st.session_state.preferred_language}")
        
        if prompt := st.chat_input("Ask about CreditChek API or SDK..."):
            logger.info(f"Received user prompt: {prompt}")
            # Append preferred language to query if not specified
            query = prompt
            if not any(lang.lower() in prompt.lower() for lang in ["python", "nodejs", "php laravel", "golang"]):
                query = f"{prompt} in {st.session_state.preferred_language}"
            logger.debug(f"Modified query: {query}")
            
            # Add user message
            st.session_state.messages.append({"role": "user", "content": prompt})
            with st.chat_message("user"):
                st.markdown(prompt)
            
            # Generate and display response
            with st.chat_message("assistant"):
                try:
                    logger.info(f"Generating response for query: {query}")
                    response = self.bot_service.generate_response(query)
                    logger.debug(f"Response generated: {json.dumps(response, indent=2)}")
                    
                    # Display response text
                    if response["response"]["text"]:
                        text = response["response"]["text"]
                        code_snippets = response["response"].get("code_snippets", {})
                        for lang, code in code_snippets.items():
                            code_block = f"```{lang.lower()}\n{code}\n```"
                            if code_block in text:
                                logger.debug(f"Code snippet for {lang} already in text, skipping separate rendering")
                                code_snippets.pop(lang, None)
                        st.markdown(text)
                    else:
                        logger.warning("Response text is empty")
                        st.markdown("No response text available.")
                    
                    # Display code snippets if not already in text
                    if code_snippets:
                        for lang, code in code_snippets.items():
                            logger.info(f"Rendering code snippet for language: {lang}")
                            st.code(code, language=lang.lower())
                    else:
                        logger.debug("No additional code snippets to render")
                        for result in response.get("tool_results", []):
                            if result.get("tool") == "code_generator" and result.get("output"):
                                logger.info(f"Rendering code from tool_results for language: {st.session_state.preferred_language}")
                                st.code(result["output"], language=st.session_state.preferred_language.lower())
                    
                    # Save response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["response"]["text"] or "No response text available."
                    })
                    logger.debug(f"Session state messages: {st.session_state.messages}")
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}. Please check logs and ensure GOOGLE_API_KEY is valid."
                    logger.error(error_msg)
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    def render(self):
        logger.debug("Rendering ChatInterface")
        self.display_chat_history()
        self.handle_user_input()
        