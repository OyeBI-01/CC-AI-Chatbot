import os
from dotenv import load_dotenv
import logging
import streamlit as st

logger = logging.getLogger(__name__)
load_dotenv()

def get_pinecone_config():
    config = {
        "api_key": os.getenv("PINECONE_API_KEY"),
        "PINECONE_ENVIRONMENT": os.getenv("PINECONE_ENVIRONMENT", "us-east1-aws"),
        "PINECONE_INDEX": os.getenv("PINECONE_INDEX"),
        "namespace": os.getenv("PINECONE_NAMESPACE", None)
    }
    logger.info(f"get_pinecone_config output: {config | {'api_key': '***'}}")
    return config

def get_openai_config():
    return {
        "api_key": os.getenv("GOOGLE_API_KEY"),
        "model": "gemini-2.0-flash",
        "temperature": 0.7
    }

def get_app_config():
    return {
        "doc_url": "https://docs.creditchek.africa",
        "supported_languages": ["Python", "NodeJS", "PHP Laravel", "GoLang"],
        "chunk_size": 1000,
        "chunk_overlap": 200
    }

def setup_page_config():
    st.set_page_config(
        page_title="Mark Musk - CreditChek API Assistant",
        page_icon=":robot:",
        layout="wide"
    )