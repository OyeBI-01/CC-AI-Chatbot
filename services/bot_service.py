from typing import Dict, Any, List
import os
import logging
import tempfile
import shutil
import subprocess
import sys
import time
from dotenv import load_dotenv
import pinecone
from pinecone import ServerlessSpec
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_pinecone import PineconeVectorStore
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain.memory import ConversationBufferMemory
from langchain.chains import LLMChain
from langchain.prompts import PromptTemplate
from langchain.docstore.document import Document
from pydantic import BaseModel
from PyPDF2 import PdfReader
import io
from concurrent.futures import ThreadPoolExecutor, as_completed

# Set custom temp directory
tempfile.tempdir = "C:\\Users\\User\\Desktop\\Mark Musk\\temp"

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Load environment variables
load_dotenv()

from utils.config import get_pinecone_config, get_app_config

class BotService:
    def __init__(self, fastapi_url: str):
        self.fastapi_url = fastapi_url

        # Load configurations
        try:
            self.pinecone_config = get_pinecone_config()
            self.app_config = get_app_config()
        except Exception as e:
            logger.error(f"Failed to load configurations: {e}")
            raise

        # Apply Pinecone fallbacks
        self.pinecone_config["api_key"] = self.pinecone_config.get("api_key", os.getenv("PINECONE_API_KEY"))
        self.pinecone_config["PINECONE_INDEX"] = self.pinecone_config.get("PINECONE_INDEX", os.getenv("PINECONE_INDEX"))
        self.pinecone_config["namespace"] = self.pinecone_config.get("namespace", os.getenv("PINECONE_NAMESPACE", None))

        # Validate Pinecone config
        required_pinecone_keys = ["api_key", "PINECONE_INDEX"]
        logger.info(f"Pinecone config: {self.pinecone_config | {'api_key': '***'}}")
        for key in required_pinecone_keys:
            if not self.pinecone_config.get(key):
                logger.error(f"Missing Pinecone configuration key: {key}")
                raise ValueError(f"Missing required Pinecone configuration key: {key}")

        # Set Google API key
        os.environ["GOOGLE_API_KEY"] = os.getenv("GOOGLE_API_KEY")

        # Initialize embeddings
        try:
            self.embeddings = GoogleGenerativeAIEmbeddings(
                model="models/text-embedding-004",
                google_api_key=os.getenv("GOOGLE_API_KEY")
            )
            logger.info(f"Using embedding model: text-embedding-004 (768 dimensions)")
        except Exception as e:
            logger.error(f"Failed to initialize Gemini embeddings: {e}")
            raise

        # Initialize Pinecone
        try:
            self.pinecone_instance = pinecone.Pinecone(api_key=self.pinecone_config["api_key"])
        except Exception as e:
            logger.error(f"Failed to initialize Pinecone: {e}")
            raise

        # Create Pinecone index if needed
        index_name = self.pinecone_config["PINECONE_INDEX"]
        try:
            existing_indexes = [index["name"] for index in self.pinecone_instance.list_indexes()]
            if index_name not in existing_indexes:
                logger.info(f"Creating Pinecone index: {index_name}")
                self.pinecone_instance.create_index(
                    name=index_name,
                    dimension=768,  # Updated for text-embedding-004
                    metric="cosine",
                    spec=ServerlessSpec(cloud="aws", region="us-east-1")
                )
        except Exception as e:
            logger.error(f"Failed to create Pinecone index: {e}")
            raise

        # Initialize vector store
        try:
            self.vector_store = PineconeVectorStore.from_existing_index(
                index_name=index_name,
                embedding=self.embeddings,
                namespace=self.pinecone_config.get("namespace", None)
            )
        except Exception as e:
            logger.error(f"Failed to initialize PineconeVectorStore: {e}")
            raise

        # Initialize LLM
        try:
            self.llm = ChatGoogleGenerativeAI(
                model="gemini-1.5-flash",
                api_key=os.getenv("GOOGLE_API_KEY"),
                temperature=0.7
            )
            logger.info(f"Using LLM: gemini-1.5-flash")
        except Exception as e:
            logger.error(f"Failed to initialize ChatGoogleGenerativeAI: {e}")
            raise

        # Initialize memory
        self.memory = ConversationBufferMemory(
            input_key="question",
            memory_key="chat_history"
        )

        # Define prompt template
        self.qa_template = """You are Mark Musk, an AI assistant specialized in CreditChek API and SDK integration.
Provide detailed, code-focused answers for SDK questions, including examples where possible.
Use the following context to answer the question. If you don't know, say so.

Context: {context}

Chat History: {chat_history}

Question: {question}

Answer:"""
        self.qa_prompt = PromptTemplate(
            template=self.qa_template,
            input_variables=["context", "chat_history", "question"]
        )

        # Setup QA chain
        self.qa_chain = LLMChain(
            llm=self.llm,
            prompt=self.qa_prompt,
            memory=self.memory,
            output_key="answer",
            verbose=True
        )

        # Initialize document store
        self.init_document_store()

    def clear_index(self):
        """Clear all documents in the Pinecone index."""
        try:
            logger.info(f"Clearing Pinecone index {self.pinecone_config['PINECONE_INDEX']}")
            index = self.pinecone_instance.Index(self.pinecone_config["PINECONE_INDEX"])
            stats = index.describe_index_stats()
            logger.info(f"Index stats before clearing: {stats}")
            self.vector_store.delete(delete_all=True, namespace=self.pinecone_config.get("namespace", None))
            stats = index.describe_index_stats()
            logger.info(f"Index stats after clearing: {stats}")
        except Exception as e:
            logger.error(f"Failed to clear Pinecone index: {e}")
            raise

    def init_document_store(self):
        """Initialize document store with sample CreditChek docs if not already indexed."""
        try:
            logger.info("Checking Pinecone index for existing documents")
            index = self.pinecone_instance.Index(self.pinecone_config["PINECONE_INDEX"])
            stats = index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)
            logger.info(f"Index contains {total_vectors} vectors")
            
            # Only index sample docs if index is empty
            if total_vectors > 0:
                logger.info("Index is not empty, skipping sample document initialization")
                return
            
            self.sample_docs = [
                Document(page_content="To authenticate with CreditChek API...", metadata={"source": "sample_auth", "content": "..."}),
                Document(page_content="CreditChek API provides identity verification...", metadata={"source": "sample_identity", "content": "..."}),
    
                Document(
                    page_content="CreditChek webhooks enable real-time transaction updates. Register a webhook URL in the dashboard and handle POST requests with JSON payloads containing transaction_id, status, and amount.",
                    metadata={"source": "sample_webhooks", "content": "..."}
                ),
                Document(
                    page_content="RecovaPRO SDK: Initialize with `from creditchek import RecovaPRO; client = RecovaPRO(api_key='YOUR_API_KEY')`. Use client.verify_identity(first_name='John', last_name='Doe') for identity checks.",
                    metadata={"source": "sample_sdk", "content": "..."}
                ),
                Document(
                    page_content="Schedule payments via /api/v1/payments/schedule. POST with amount, recipient, and schedule_date. Requires Authorization: Bearer YOUR_API_KEY.",
                    metadata={"source": "sample_payments", "content": "..."}
                ),
            ]
            logger.info("Indexing sample documents")
            self._upload_documents_in_batches(self.sample_docs, batch_size=5)
        except Exception as e:
            logger.error(f"Failed to index sample documents: {e}")
            raise

    def _upload_documents_in_batches(self, documents: List[Document], batch_size: int = 50) -> None:
        """Upload documents to Pinecone in batches with size estimation and parallel embedding."""
        def embed_batch(batch: List[Document]) -> List[List[float]]:
            try:
                return self.embeddings.embed_documents([doc.page_content for doc in batch])
            except Exception as e:
                logger.error(f"Failed to embed batch: {e}")
                raise

        for doc in documents:
            content_size = len(doc.page_content.encode('utf-8')) / 1024
            metadata_size = sys.getsizeof(doc.metadata) / 1024
            logger.info(f"Document {doc.metadata.get('source', 'unknown')}: "
                        f"content {content_size:.2f} KB, metadata {metadata_size:.2f} KB")
            if content_size > 500:
                logger.warning(f"Skipping document {doc.metadata.get('source', 'unknown')}: "
                               f"content size {content_size:.2f} KB exceeds 500KB")
                continue
        
        # Parallel embedding with ThreadPoolExecutor
        max_workers = 4  # Adjust based on Google API rate limits and CPU cores
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        total_start_time = time.time()
        
        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            future_to_batch = {}
            for i, batch in enumerate(batches):
                batch_size_bytes = sum(len(doc.page_content.encode('utf-8')) + sys.getsizeof(doc.metadata) for doc in batch)
                embedding_size = len(batch) * 768 * 4  # Updated for text-embedding-004
                total_batch_size = batch_size_bytes + embedding_size
                logger.info(f"Submitting batch {i + 1} with {len(batch)} documents, "
                            f"estimated size: {total_batch_size / (1024 * 1024):.2f} MB "
                            f"(content+metadata: {batch_size_bytes / (1024 * 1024):.2f} MB, "
                            f"embeddings: {embedding_size / (1024 * 1024):.2f} MB)")
                future_to_batch[executor.submit(embed_batch, batch)] = (i, batch)
            
            for future in as_completed(future_to_batch):
                i, batch = future_to_batch[future]
                start_time = time.time()
                try:
                    embeddings = future.result()
                    vectors = [
                        (
                            f"doc_{i * batch_size + j}",
                            emb,
                            {**doc.metadata, "content": doc.page_content} if doc.metadata else {"content": doc.page_content}
                        )
                        for j, (emb, doc) in enumerate(zip(embeddings, batch))
                    ]
                    index = self.pinecone_instance.Index(self.pinecone_config["PINECONE_INDEX"])
                    index.upsert(vectors=vectors, namespace=self.pinecone_config.get("namespace", None))
                    logger.info(f"Batch {i + 1} uploaded in {time.time() - start_time:.2f} seconds")
                except Exception as e:
                    logger.error(f"Failed to process batch {i + 1}: {e}")
                    raise
        
        logger.info(f"Total upload time: {time.time() - total_start_time:.2f} seconds")

    def process_github_repo(self, repo_url: str, branch: str = "master"):
        """Clone GitHub repo using SSH and index Markdown files in Pinecone."""
        temp_dir = tempfile.mkdtemp()
        total_start_time = time.time()
        try:
            # Clear index before indexing new content
            self.clear_index()
            logger.info(f"Cloning repository {repo_url} branch {branch} to {temp_dir}")
            subprocess.run(
                ["git", "clone", "--branch", branch, repo_url, temp_dir],
                check=True,
                capture_output=True,
                text=True
            )
            docs = []
            max_file_size = 250 * 1024  # 250KB limit per file
            docs_dir = os.path.join(temp_dir, "docs")
            if os.path.exists(docs_dir):
                for root, _, files in os.walk(docs_dir):
                    for file in files:
                        if file.endswith((".md", ".mdx")):
                            file_path = os.path.join(root, file)
                            file_size = os.path.getsize(file_path)
                            if file_size > max_file_size:
                                logger.warning(f"Skipping {file}: size {file_size / (1024 * 1024):.2f} MB exceeds 0.25MB limit")
                                continue
                            with open(file_path, "r", encoding="utf-8") as f:
                                content = f.read()
                                docs.append(Document(
                                    page_content=content,
                                    metadata={"source": file, "repo": repo_url}
                                ))
            
            if not docs:
                logger.warning(f"No Markdown files found in {repo_url}/docs")
                return
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            split_docs = text_splitter.split_documents(docs)
            logger.info(f"Split {len(docs)} documents into {len(split_docs)} chunks")
            
            self._upload_documents_in_batches(split_docs, batch_size=50)
            logger.info(f"Indexed {len(split_docs)} document chunks from {repo_url} in {time.time() - total_start_time:.2f} seconds")
            
        except Exception as e:
            logger.error(f"Failed to process GitHub repository {repo_url}: {e}")
            raise
        finally:
            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir, ignore_errors=True)

    def process_uploaded_doc(self, uploaded_file):
        """Process uploaded PDF or TXT file and index in Pinecone."""
        try:
            if uploaded_file.type == "application/pdf":
                pdf_reader = PdfReader(uploaded_file)
                text = ""
                for page in pdf_reader.pages:
                    text += page.extract_text() or ""
            elif uploaded_file.type == "text/plain":
                text = uploaded_file.read().decode("utf-8")
            else:
                raise ValueError("Unsupported file type")
            
            text_splitter = RecursiveCharacterTextSplitter(chunk_size=1000, chunk_overlap=100)
            chunks = text_splitter.split_text(text)
            docs = [Document(page_content=chunk, metadata={"source": uploaded_file.name}) for chunk in chunks]
            
            self._upload_documents_in_batches(docs, batch_size=50)
            logger.info(f"Indexed {len(docs)} document chunks from {uploaded_file.name}")
        except Exception as e:
            logger.error(f"Failed to process uploaded file: {e}")
            raise

    def generate_response(self, query: str) -> Dict[str, Any]:
        """Generate a response to the user's query."""
        query_lower = query.lower()

        # Handle specific queries
        show_example_match = None
        for lang in self.app_config["supported_languages"]:
            if f"show {lang.lower()} example" in query_lower:
                show_example_match = lang
                break

        if show_example_match:
            return {
                "text": f"Here's a {show_example_match} example for CreditChek API:",
                "code_snippets": {
                    show_example_match: f"# Example {show_example_match} code\n..."
                }
            }

        # Perform vector search with error handling
        try:
            index = self.pinecone_instance.Index(self.pinecone_config["PINECONE_INDEX"])
            stats = index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)
            logger.info(f"Index contains {total_vectors} vectors for query: {query}")
            
            if total_vectors == 0:
                logger.warning("Index is empty, using sample documents")
                relevant_docs = self.sample_docs[:3]
            else:
                # Use raw Pinecone query to handle metadata safely
                query_embedding = self.embeddings.embed_query(query)
                query_results = index.query(
                    vector=query_embedding,
                    top_k=3,
                    include_metadata=True,
                    namespace=self.pinecone_config.get("namespace", None)
                )
                relevant_docs = []
                for match in query_results.get("matches", []):
                    metadata = match.get("metadata", {})
                    content = metadata.get("content", "No content available")
                    logger.info(f"Retrieved document ID {match.get('id')}: metadata={metadata}")
                    relevant_docs.append(Document(
                        page_content=content,
                        metadata=metadata
                    ))
        except Exception as e:
            logger.error(f"Vector search failed: {e}")
            relevant_docs = self.sample_docs[:3]

        context = "\n".join([doc.page_content for doc in relevant_docs])
        logger.info(f"Context for query '{query}': {context[:200]}...")

        # Run QA chain
        try:
            chain_response = self.qa_chain.invoke({
                "context": context,
                "question": query
            })
            response = chain_response["answer"]
        except Exception as e:
            logger.error(f"QA chain failed: {e}")
            response = "Sorry, I couldn't process your request."

        return {"text": response}