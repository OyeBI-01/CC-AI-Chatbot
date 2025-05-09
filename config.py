# utils/config.py
import os
from dotenv import load_dotenv
import logging
import streamlit as st

logger = logging.getLogger(__name__)
load_dotenv()

def get_pinecone_config():
    config = {
        "api_key": os.getenv("PINECONE_API_KEY"),
        "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT", "us-west-2"),
        "PINECONE_INDEX": os.getenv("PINECONE_INDEX"),
        "namespace": os.getenv("PINECONE_NAMESPACE", None)
    }
    logger.info(f"get_pinecone_config output: {config | {'api_key': '***'}}")
    return config

def get_openai_config():
    return {
        "api_key": os.getenv("OPENAI_API_KEY"),
        "model": "gpt-3.5-turbo",
        "temperature": 0.7
    }

def get_app_config():
    return {
        "supported_languages": ["Python", "NodeJS", "PHP Laravel", "GoLang", "Ruby", "ASP.NET", "Java"]
    }

def setup_page_config():
    st.set_page_config(
        page_title="Mark Musk - CreditChek API Assistant",
        page_icon=":robot:",
        layout="wide"
    )