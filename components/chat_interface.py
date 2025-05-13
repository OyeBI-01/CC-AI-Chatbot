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
        
        # Expanded language options with common aliases
        self.supported_languages = [
            "Python", "JavaScript", "NodeJS", "TypeScript",
            "Java", "C#", "Go", "Golang", "PHP", "Laravel",
            "Ruby", "Rust", "Swift", "Kotlin", "Dart",
            "Scala", "R", "Perl", "Bash", "Shell"
        ]

    def initialize_session_state(self):
        logger.debug("Initializing session state")
        if "messages" not in st.session_state:
            st.session_state.messages = []
        if "preferred_language" not in st.session_state:
            st.session_state.preferred_language = "Python"
        if "custom_language" not in st.session_state:
            st.session_state.custom_language = ""

    def display_chat_history(self):
        logger.debug("Displaying chat history")
        for message in st.session_state.messages:
            with st.chat_message(message["role"]):
                st.markdown(message["content"])

    def handle_user_input(self):
        logger.debug("Checking for user input")
        with st.sidebar:
            st.header("Preferences")
            
            # Language selection with custom option - FIXED SYNTAX
            # language_options = self.supported_languages + ["Other..."]
            # language_options = self.supported_languages + ["Other..."]
            language_options = self.supported_languages
            selected_language = st.selectbox(
                "Select Programming Language",
                options=language_options,
                index=language_options.index(st.session_state.preferred_language)
            )
            
            # Handle custom language input - FIXED IF BLOCK
            if selected_language == "Other...":
                custom_lang = st.text_input(
                    "Specify Language", 
                    value=st.session_state.custom_language,
                    placeholder="e.g., Rust, Kotlin, etc."
                )
                if custom_lang:
                    st.session_state.preferred_language = custom_lang.strip()
                    st.session_state.custom_language = custom_lang.strip()
            else:
                st.session_state.preferred_language = selected_language
            
            logger.debug(f"Selected language: {st.session_state.preferred_language}")

        if prompt := st.chat_input("Ask about CreditChek API or SDK..."):
            logger.info(f"Received user prompt: {prompt}")
            
            # Set language preference in bot service
            self.bot_service.set_language_preference(st.session_state.preferred_language)
            
            # Process query without modifying it
            query = prompt
            
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
                        
                        # Check for code snippets already in text
                        for lang, code in list(code_snippets.items()):
                            code_block = f"```{lang.lower()}\n{code}\n```"
                            if code_block in text:
                                logger.debug(f"Code snippet for {lang} already in text, skipping separate rendering")
                                code_snippets.pop(lang, None)
                        
                        st.markdown(text)
                    else:
                        logger.warning("Response text is empty")
                        st.markdown("No response text available.")
                    
                    # Display additional code snippets
                    if code_snippets:
                        for lang, code in code_snippets.items():
                            logger.info(f"Rendering code snippet for language: {lang}")
                            st.code(code, language=lang.lower())
                    
                    # Fallback for code in tool results
                    for result in response.get("tool_results", []):
                        if result.get("tool") == "code_generator" and result.get("output"):
                            detected_lang = self._detect_language_from_code(result["output"])
                            logger.info(f"Rendering code from tool_results for language: {detected_lang}")
                            st.code(result["output"], language=detected_lang.lower())
                    
                    # Save response
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": response["response"]["text"] or "No response text available."
                    })
                    
                except Exception as e:
                    error_msg = f"Error generating response: {str(e)}. Please check logs and ensure GOOGLE_API_KEY is valid."
                    logger.error(error_msg)
                    st.error(error_msg)
                    st.session_state.messages.append({
                        "role": "assistant",
                        "content": error_msg
                    })

    def _detect_language_from_code(self, code: str) -> str:
        """Heuristically detect language from code snippet"""
        code_lower = code.lower()
        language_map = {
            "import java": "Java",
            "package main": "Go",
            "<?php": "PHP",
            "def ": "Python",
            "function ": "JavaScript",
            "fn ": "Rust",
            "func ": "Go",
            "class ": "Python"  # Could be Python, Java, C#, etc.
        }
        
        for pattern, lang in language_map.items():
            if pattern in code_lower:
                return lang
        
        # Fallback to preferred language if detection fails
        return st.session_state.preferred_language

    def render(self):
        logger.debug("Rendering ChatInterface")
        self.display_chat_history()
        self.handle_user_input()  # FIXED: Proper line separation