import os
import tempfile
import shutil
import time
import logging
import json
import re
from typing import List, Dict, Any, Optional
from git import Repo
from pinecone import Pinecone, ServerlessSpec
from langchain_community.document_loaders import TextLoader, PyPDFLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.documents import Document
from langchain_google_genai import GoogleGenerativeAIEmbeddings, ChatGoogleGenerativeAI
from langchain_core.prompts import PromptTemplate
from langchain.memory import ConversationBufferMemory
import httpx
from utils.config import get_app_config, get_pinecone_config
from concurrent.futures import ThreadPoolExecutor
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

logging.basicConfig(level=logging.DEBUG, format="%(asctime)s - %(name)s - %(levelname)s - %(message)s")
logger = logging.getLogger(__name__)

class BotService:
    def __init__(self, fastapi_url: str):
        self.fastapi_url = fastapi_url
        try:
            self.app_config = get_app_config()
            if not isinstance(self.app_config, dict) or "supported_languages" not in self.app_config:
                raise ValueError("get_app_config must return a dictionary with 'supported_languages'")
            # Add supported frameworks for Python
            self.app_config["supported_frameworks"] = {
                "Python": ["Flask", "FastAPI", "Django"],
                "NodeJS": ["Express"],
                "PHP Laravel": ["Laravel"],
                "GoLang": ["net/http"]
            }
        except Exception as e:
            logger.error(f"Failed to load app_config: {e}")
            raise ValueError("get_app_config must return a dictionary with 'supported_languages' and 'supported_frameworks'")
        logger.info(f"App config: {self.app_config}")
        try:
            self.pinecone_config = get_pinecone_config()
            if not isinstance(self.pinecone_config, dict):
                raise ValueError("get_pinecone_config must return a dictionary")
            if not self.pinecone_config.get("PINECONE_INDEX") or not isinstance(self.pinecone_config["PINECONE_INDEX"], str):
                logger.warning("PINECONE_INDEX is missing or invalid, defaulting to 'creditchek-docs'")
                self.pinecone_config["PINECONE_INDEX"] = "creditchek-docs"
        except Exception as e:
            logger.error(f"Failed to load pinecone_config: {e}")
            raise ValueError("get_pinecone_config must return a dictionary with a valid 'PINECONE_INDEX'")
        logger.info(f"Pinecone config: {self.pinecone_config | {'api_key': '***'}}")
        self.pinecone_instance = Pinecone(api_key=os.getenv("PINECONE_API_KEY") or self.pinecone_config.get("api_key"))
        self.index = self.create_index()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
        logger.info(f"Using embedding model: text-embedding-004 (768 dimensions)")
        self.llm = ChatGoogleGenerativeAI(
            model="gemini-2.0-flash",
            api_key=os.getenv("GOOGLE_API_KEY"),
            temperature=0.7
        )
        self.memory = ConversationBufferMemory(
            memory_key="chat_history",
            return_messages=True,
            output_key="answer"
        )
        self.agent_prompt = PromptTemplate.from_template(
            """You are Mark Musk, an AI assistant for CreditChek API and SDK integration.
            Your goal is to provide detailed, code-focused answers using tools and reasoning.
            Available tools:
            - document_retrieval: Search Pinecone for CreditChek docs (returns up to 3 documents).
            - mock_creditchek_api: Simulate CreditChek API calls (e.g., /api/v1/repayments).
            - code_generator: Generate SDK or webhook code snippets in supported languages: {supported_languages}.
            Supported frameworks for Python: {supported_frameworks_python}.

            Query: {query}
            Chat History: {chat_history}

            Steps:
            1. Analyze the query to identify required tools, steps, preferred programming language, and framework.
            2. Use tools as needed (e.g., retrieve docs, simulate API, generate code in the specified language and framework).
            3. Combine results into a detailed, code-focused answer.
            4. If information is missing, explain and suggest next steps.

            Return a valid JSON plan in the following format:
            ```json
            {
                "plan": {
                    "steps": [
                        {"tool": "document_retrieval", "params": {"query": "webhooks"}, "description": "Retrieve webhook docs"},
                        {"tool": "code_generator", "params": {"language": "Python", "framework": "Flask", "task": "webhook"}, "description": "Generate Python webhook code"}
                    ]
                }
            }
            ```
            Ensure proper JSON structure with no trailing commas or syntax errors. If you cannot generate a valid plan, return:
            ```json
            {"plan": {"steps": []}}
            ```"""
        ).partial(
            supported_languages=", ".join(self.app_config["supported_languages"]),
            supported_frameworks_python=", ".join(self.app_config["supported_frameworks"]["Python"])
        )
        self.sample_docs = [
            Document(
                page_content="To authenticate with CreditChek API, you need to obtain an API key from the dashboard. Include it in the header of all requests as 'Authorization: Bearer YOUR_API_KEY'. All API requests must be made over HTTPS to ensure security.",
                metadata={"source": "sample_auth", "content": "..."}
            ),
            Document(
                page_content="CreditChek API provides identity verification through the /api/v1/identity endpoint. This endpoint requires parameters like first_name, last_name, dob, and id_number.",
                metadata={"source": "sample_identity", "content": "..."}
            ),
            Document(
                page_content="The CreditChek API provides repayment breakdowns via GET /api/v1/repayments?customer_id={id}. Include 'Authorization: Bearer YOUR_API_KEY' header. Response includes payment_date, principal, interest, total_payment, and remaining_balance. Example: {'payments': [{'date': '2025-05-01', 'principal': 100, 'interest': 10, 'total_payment': 110, 'balance': 900}, ...]}. RecovaPRO SDK: Use client.get_repayment_schedule(customer_id).",
                metadata={"source": "sample_repayments", "content": "..."}
            ),
            Document(
                page_content="CreditChek webhooks: Register a URL in the dashboard to receive POST requests with transaction updates (transaction_id, status, amount, timestamp). Verify x-auth-signature header using LiveSecretKey. Example payload: {'transaction_id': '123', 'status': 'completed', 'amount': 100.50, 'timestamp': '2025-05-08T12:00:00Z'}. Supported frameworks for Python: Flask (lightweight, simple), FastAPI (async, high-performance), Django (robust, full-featured). Choose Flask for simplicity, FastAPI for async needs, or Django for existing Django projects.",
                metadata={"source": "sample_webhooks", "content": "..."}
            ),
            Document(
                page_content="RecovaPRO SDK: To configure, install the SDK using `pip install creditchek`. Obtain your API key from the CreditChek dashboard and store it in an environment variable (e.g., using `python-dotenv`). Initialize with `from creditchek import RecovaPRO; client = RecovaPRO(api_key=os.environ.get('CREDITCHEK_API_KEY'))`. Use methods like `client.verify_identity(first_name='John', last_name='Doe', dob='1985-01-01')` or `client.get_repayment_schedule(customer_id='12345')`. Refer to https://docs.creditchek.africa for full documentation. If the `creditchek` library is unavailable, contact CreditChek support or use raw API calls.",
                metadata={"source": "sample_sdk", "content": "..."}
            )
        ]
        self.init_document_store()

    def create_index(self):
        index_name = self.pinecone_config["PINECONE_INDEX"]
        existing_indexes = [index["name"] for index in self.pinecone_instance.list_indexes()]
        if index_name not in existing_indexes:
            logger.info(f"Creating Pinecone index: {index_name}")
            self.pinecone_instance.create_index(
                name=index_name,
                dimension=768,
                metric="cosine",
                spec=ServerlessSpec(cloud="aws", region=self.pinecone_config.get("PINECONE_ENVIRONMENT", "us-east1-aws"))
            )
        return self.pinecone_instance.Index(index_name)

    def init_document_store(self):
        logger.info("Initializing document store with sample documents")
        try:
            stats = self.index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)
            logger.info(f"Pinecone index contains {total_vectors} vectors")
            if total_vectors == 0:
                logger.info("Index is empty, uploading sample documents")
                self._upload_documents_in_batches(self.sample_docs, batch_size=5)
        except Exception as e:
            logger.error(f"Failed to initialize document store: {e}")
            raise

    def is_index_populated(self) -> bool:
        try:
            index = self.pinecone_instance.Index(self.pinecone_config["PINECONE_INDEX"])
            stats = index.describe_index_stats()
            return stats.get("total_vector_count", 0) > 0
        except Exception as e:
            logger.error(f"Failed to check index status: {e}")
            return False

    def process_github_repo(self, repo_url: str, branch: str = "master", force_reprocess: bool = False):
        if not force_reprocess and self.is_index_populated():
            logger.info(f"Index {self.pinecone_config['PINECONE_INDEX']} already populated, skipping processing")
            return
        temp_dir = tempfile.mkdtemp()
        try:
            logger.info(f"Cloning repository {repo_url} (branch: {branch}) to {temp_dir}")
            repo = Repo.clone_from(repo_url, temp_dir, branch=branch)
            docs_path = os.path.join(temp_dir, "docs")
            if not os.path.exists(docs_path):
                logger.warning(f"No 'docs' directory found in repository {repo_url}")
                return
            documents = []
            for root, _, files in os.walk(docs_path):
                for file in files:
                    if file.endswith((".md", ".mdx")):
                        file_path = os.path.join(root, file)
                        loader = TextLoader(file_path)
                        docs = loader.load()
                        documents.extend(docs)
            logger.info(f"Loaded {len(documents)} documents from repository")
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"Split {len(documents)} documents into {len(split_docs)} chunks")
            self._upload_documents_in_batches(split_docs, batch_size=50)
        except Exception as e:
            logger.error(f"Failed to process repository {repo_url}: {e}")
            raise
        finally:
            shutil.rmtree(temp_dir, ignore_errors=True)

    def _upload_documents_in_batches(self, documents: List[Document], batch_size: int):
        start_time = time.time()
        logger.info(f"Uploading {len(documents)} documents in batches of {batch_size}")
        batches = [documents[i:i + batch_size] for i in range(0, len(documents), batch_size)]
        embedding_size = len(batches) * batch_size * 768 * 4
        logger.info(f"Estimated embedding size: {embedding_size / (1024 * 1024):.2f} MB")
        with ThreadPoolExecutor(max_workers=4) as executor:
            for i, batch in enumerate(batches):
                try:
                    start_batch = time.time()
                    vectors = self._create_vectors(batch, i)
                    self.index.upsert(vectors=vectors, namespace=self.pinecone_config.get("namespace", None))
                    logger.info(f"Batch {i + 1}/{len(batches)} uploaded in {time.time() - start_batch:.2f} seconds")
                except Exception as e:
                    logger.error(f"Failed to upload batch {i + 1}: {e}")
                    raise
        logger.info(f"Indexed {len(documents)} document chunks in {time.time() - start_time:.2f} seconds")

    def _create_vectors(self, batch: List[Document], batch_index: int) -> List[Dict[str, Any]]:
        try:
            embeddings = self.embeddings.embed_documents([doc.page_content for doc in batch])
            vectors = [
                {
                    "id": f"doc_{batch_index * 1000 + i}",
                    "values": embedding,
                    "metadata": {
                        "source": doc.metadata.get("source", "unknown"),
                        "content": doc.page_content[:500]
                    }
                }
                for i, (doc, embedding) in enumerate(zip(batch, embeddings))
            ]
            return vectors
        except Exception as e:
            logger.error(f"Failed to create vectors for batch: {e}")
            raise

    def _document_retrieval_tool(self, query: str) -> List[Document]:
        try:
            index = self.pinecone_instance.Index(self.pinecone_config["PINECONE_INDEX"])
            stats = index.describe_index_stats()
            total_vectors = stats.get("total_vector_count", 0)
            if total_vectors == 0:
                logger.warning("Index is empty, using sample documents")
                return [doc for doc in self.sample_docs if "RecovaPRO SDK" in doc.page_content or query.lower() in doc.page_content.lower()][:3]
            query_embedding = self.embeddings.embed_query(query)
            query_results = index.query(
                vector=query_embedding,
                top_k=3,
                include_metadata=True,
                namespace=self.pinecone_config.get("namespace", None)
            )
            return [
                Document(
                    page_content=match.get("metadata", {}).get("content", "No content available"),
                    metadata=match.get("metadata", {})
                )
                for match in query_results.get("matches", [])
            ]
        except Exception as e:
            logger.error(f"Document retrieval failed: {e}")
            return [doc for doc in self.sample_docs if "RecovaPRO SDK" in doc.page_content or query.lower() in doc.page_content.lower()][:3]

    def _mock_creditchek_api_tool(self, endpoint: str, params: Dict[str, Any] = None) -> Dict[str, Any]:
        if endpoint == "/api/v1/repayments":
            customer_id = params.get("customer_id", "unknown")
            return {
                "payments": [
                    {
                        "date": "2025-05-01",
                        "principal": 100,
                        "interest": 10,
                        "total_payment": 110,
                        "balance": 900
                    }
                ],
                "customer_id": customer_id
            }
        elif endpoint == "/api/v1/identity":
            return {
                "status": "verified",
                "details": params
            }
        return {"error": "Endpoint not supported"}

    def _code_generator_tool(self, language: str, task: str, framework: str = None) -> str:
        language = language.lower()
        framework = framework.lower() if framework else None

        if language == "python" and "webhook" in task.lower():
            if framework == "fastapi":
                return """import os
from fastapi import FastAPI, Request, HTTPException
from fastapi.responses import JSONResponse
import hmac
import hashlib

app = FastAPI()

def verify_webhook_signature(request_body: bytes, x_auth_signature: str, live_secret_key: str) -> bool:
    signature = hmac.new(live_secret_key.encode(), request_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, x_auth_signature)

@app.post("/webhook")
async def webhook_handler(request: Request):
    request_body = await request.body()
    x_auth_signature = request.headers.get("x-auth-signature")
    live_secret_key = os.environ.get("CREDITCHEK_LIVE_SECRET_KEY")

    if not live_secret_key:
        raise HTTPException(status_code=500, detail="CREDITCHEK_LIVE_SECRET_KEY environment variable not set")

    if not verify_webhook_signature(request_body, x_auth_signature, live_secret_key):
        raise HTTPException(status_code=401, detail="Invalid signature")

    try:
        data = await request.json()
        print(f"Webhook received: {data}")
        # Process webhook data here (e.g., update database, send notifications)
        return {"status": "OK"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))
"""
            elif framework == "Django":
                return """import os
import hmac
import hashlib
import json
from django.http import JsonResponse, HttpResponse
from django.views.decorators.csrf import csrf_exempt
from django.views.decorators.http import require_POST

def verify_webhook_signature(request_body: bytes, x_auth_signature: str, live_secret_key: str) -> bool:
    signature = hmac.new(live_secret_key.encode(), request_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, x_auth_signature)

@csrf_exempt
@require_POST
def webhook_handler(request):
    request_body = request.body
    x_auth_signature = request.META.get("HTTP_X_AUTH_SIGNATURE")
    live_secret_key = os.environ.get("CREDITCHEK_LIVE_SECRET_KEY")

    if not live_secret_key:
        return JsonResponse({"error": "CREDITCHEK_LIVE_SECRET_KEY environment variable not set"}, status=500)

    if not verify_webhook_signature(request_body, x_auth_signature, live_secret_key):
        return JsonResponse({"error": "Invalid signature"}, status=401)

    try:
        data = json.loads(request_body)
        print(f"Webhook received: {data}")
        # Process webhook data here (e.g., update database, send notifications)
        return HttpResponse("OK", status=200)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)
"""
            else:  # Default to Flask
                return """from flask import Flask, request, jsonify
import hmac
import hashlib
import os

app = Flask(__name__)

def verify_webhook_signature(request_body, x_auth_signature, live_secret_key):
    signature = hmac.new(live_secret_key.encode(), request_body, hashlib.sha256).hexdigest()
    return hmac.compare_digest(signature, x_auth_signature)

@app.route('/webhook', methods=['POST'])
def webhook_handler():
    request_body = request.get_data()
    x_auth_signature = request.headers.get('x-auth-signature')
    live_secret_key = os.environ.get('CREDITCHEK_LIVE_SECRET_KEY')

    if not live_secret_key:
        return jsonify({"error": "CREDITCHEK_LIVE_SECRET_KEY environment variable not set"}), 500

    if not verify_webhook_signature(request_body, x_auth_signature, live_secret_key):
        return jsonify({"error": "Invalid signature"}), 401

    try:
        data = request.get_json()
        print(f"Webhook received: {data}")
        # Process webhook data here (e.g., update database, send notifications)
        return "OK", 200
    except Exception as e:
        return jsonify({"error": str(e)}), 500"""
        elif language == "python" and ("sdk" in task.lower() or "configure" in task.lower()):
            return """import os
from creditchek import RecovaPRO

# Initialize the RecovaPRO SDK with your API key from environment variables
client = RecovaPRO(api_key=os.environ.get('CREDITCHEK_API_KEY'))

# Example: Verify identity using the SDK
try:
    result = client.verify_identity(first_name='John', last_name='Doe', dob='1985-01-01')
    print(f"Identity Verification Result: {result}")
except Exception as e:
    print(f"Error during identity verification: {e}")

# Example: Get repayment schedule
try:
    schedule = client.get_repayment_schedule(customer_id='12345')
    print(f"Repayment Schedule: {schedule}")
except Exception as e:
    print(f"Error retrieving repayment schedule: {e}")
"""
        elif language == "nodejs" and "webhook" in task.lower():
            return """const express = require('express');
const crypto = require('crypto');
const app = express();

app.use(express.json());

function verifyWebhookSignature(requestBody, xAuthSignature, liveSecretKey) {
    const signature = crypto.createHmac('sha256', liveSecretKey)
                           .update(JSON.stringify(requestBody))
                           .digest('hex');
    return signature === xAuthSignature;
}

app.post('/webhook', (req, res) => {
    const xAuthSignature = req.headers['x-auth-signature'];
    const liveSecretKey = process.env.CREDITCHEK_LIVE_SECRET_KEY;

    if (!liveSecretKey) {
        return res.status(500).json({ error: 'CREDITCHEK_LIVE_SECRET_KEY environment variable not set' });
    }

    if (!verifyWebhookSignature(req.body, xAuthSignature, liveSecretKey)) {
        return res.status(401).json({ error: 'Invalid signature' });
    }

    console.log('Webhook received:', req.body);
    res.status(200).send('OK');
});

app.listen(3000, () => console.log('Webhook server running on port 3000'));"""
        elif language == "nodejs" and ("sdk" in task.lower() or "configure" in task.lower()):
            return """const RecovaPRO = require('recovapro');

// Initialize the RecovaPRO SDK with your API key
const client = new RecovaPRO({ apiKey: process.env.CREDITCHEK_API_KEY });

// Example: Verify identity using the SDK
client.verifyIdentity({ firstName: 'John', lastName: 'Doe', dob: '1985-01-01' })
    .then(result => console.log(result))
    .catch(err => console.error(err));

// Example: Get repayment schedule
client.getRepaymentSchedule({ customerId: '12345' })
    .then(schedule => console.log(schedule))
    .catch(err => console.error(err));
"""
        elif language == "php laravel" and "webhook" in task.lower():
            return """<?php

namespace App\Http\Controllers;

use Illuminate\Http\Request;
use Illuminate\Support\Facades\Hash;

class WebhookController extends Controller
{
    public function handleWebhook(Request $request)
    {
        $requestBody = $request->getContent();
        $xAuthSignature = $request->header('x-auth-signature');
        $liveSecretKey = env('CREDITCHEK_LIVE_SECRET_KEY');

        if (!$liveSecretKey) {
            return response()->json(['error' => 'CREDITCHEK_LIVE_SECRET_KEY environment variable not set'], 500);
        }

        $signature = hash_hmac('sha256', $requestBody, $liveSecretKey);

        if (!hash_equals($signature, xAuthSignature)) {
            return response()->json(['error' => 'Invalid signature'], 401);
        }

        \Log::info('Webhook received: ' . json_encode($request->all()));
        return response('OK', 200);
    }
}
"""
        elif language == "php laravel" and ("sdk" in task.lower() or "configure" in task.lower()):
            return """<?php

use CreditChek\RecovaPRO;

// Initialize the RecovaPRO SDK with your API key
$client = new RecovaPRO(['api_key' => env('CREDITCHEK_API_KEY')]);

// Example: Verify identity using the SDK
try {
    $result = $client->verifyIdentity([
        'first_name' => 'John',
        'last_name' => 'Doe',
        'dob' => '1985-01-01'
    ]);
    \Log::info('Verification result: ' . json_encode($result));
} catch (\Exception $e) {
    \Log::error('Error during identity verification: ' . $e->getMessage());
}

// Example: Get repayment schedule
try {
    $schedule = $client->getRepaymentSchedule(['customer_id' => '12345']);
    \Log::info('Repayment schedule: ' . json_encode($schedule));
} catch (\Exception $e) {
    \Log::error('Error retrieving repayment schedule: ' . $e->getMessage());
}
"""
        elif language == "golang" and "webhook" in task.lower():
            return """package main

import (
    "crypto/hmac"
    "crypto/sha256"
    "encoding/hex"
    "encoding/json"
    "io/ioutil"
    "log"
    "net/http"
    "os"
)

func verifyWebhookSignature(requestBody []byte, xAuthSignature, liveSecretKey string) bool {
    mac := hmac.New(sha256.New, []byte(liveSecretKey))
    mac.Write(requestBody)
    expectedSignature := hex.EncodeToString(mac.Sum(nil))
    return hmac.Equal([]byte(expectedSignature), []byte(xAuthSignature))
}

func webhookHandler(w http.ResponseWriter, r *http.Request) {
    requestBody, err := ioutil.ReadAll(r.Body)
    if err != nil {
        http.Error(w, "Error reading request body", http.StatusBadRequest)
        return
    }

    xAuthSignature := r.Header.Get("x-auth-signature")
    liveSecretKey := os.Getenv("CREDITCHEK_LIVE_SECRET_KEY")

    if liveSecretKey == "" {
        http.Error(w, `{"error": "CREDITCHEK_LIVE_SECRET_KEY environment variable not set"}`, http.StatusInternalServerError)
        return
    }

    if (!verifyWebhookSignature(requestBody, xAuthSignature, liveSecretKey)) {
        http.Error(w, `{"error": "Invalid signature"}`, http.StatusUnauthorized)
        return
    }

    var payload map[string]interface{}
    json.Unmarshal(requestBody, &payload)
    log.Printf("Webhook received: %v", payload)
    w.Write([]byte("OK"))
}

func main() {
    http.HandleFunc("/webhook", webhookHandler)
    log.Fatal(http.ListenAndServe(":8080", nil))
}
"""
        elif language == "golang" and ("sdk" in task.lower() or "configure" in task.lower()):
            return """package main

import (
    "log"
    "os"
    "github.com/creditchek/recovapro"
)

func main() {
    // Initialize the RecovaPRO SDK with your API key
    client := recovapro.NewClient(os.Getenv("CREDITCHEK_API_KEY"))

    // Example: Verify identity using the SDK
    result, err := client.VerifyIdentity(recovapro.IdentityParams{
        FirstName: "John",
        LastName:  "Doe",
        DOB:       "1985-01-01",
    })
    if err != nil {
        log.Fatalf("Verification failed: %v", err)
    }
    log.Printf("Verification result: %v", result)

    // Example: Get repayment schedule
    schedule, err := client.GetRepaymentSchedule("12345")
    if err != nil {
        log.Fatalf("Repayment schedule failed: %v", err)
    }
    log.Printf("Repayment schedule: %v", schedule)
}
"""
        return f"// {language} code for {task} (framework: {framework or 'none'})\n// Not implemented"

    def select_tools(self, query: str) -> List[Dict[str, Any]]:
        """Analyze query and suggest tools to use."""
        query_lower = query.lower()
        language, framework = self._detect_language_and_framework(query)
        tools = []

        # Keyword-based tool selection
        if "webhook" in query_lower:
            tools.append({
                "tool": "code_generator",
                "params": {"language": language, "framework": framework, "task": "webhook"},
                "description": f"Generate {language} webhook code using {framework or 'default framework'}"
            })
            tools.append({
                "tool": "document_retrieval",
                "params": {"query": "webhook"},
                "description": "Retrieve webhook documentation"
            })
        if "api" in query_lower or "endpoint" in query_lower:
            endpoint = "/api/v1/repayments" if "repayment" in query_lower else "/api/v1/identity"
            tools.append({
                "tool": "mock_creditchek_api",
                "params": {"endpoint": endpoint, "params": {"customer_id": "sample_id"}},
                "description": f"Simulate {endpoint} API call"
            })
        if "sdk" in query_lower or "verify identity" in query_lower or "configure" in query_lower:
            tools.append({
                "tool": "code_generator",
                "params": {"language": language, "task": "sdk"},
                "description": "Generate SDK code example"
            })
            tools.append({
                "tool": "document_retrieval",
                "params": {"query": "RecovaPRO SDK"},
                "description": "Retrieve SDK documentation"
            })

        # Default to document retrieval if no specific tools match
        if not tools:
            tools.append({
                "tool": "document_retrieval",
                "params": {"query": query},
                "description": "Retrieve relevant documents"
            })

        return tools

    def _detect_language_and_framework(self, query: str) -> tuple[str, Optional[str]]:
        """Detect preferred programming language and framework from query and chat history."""
        query_lower = query.lower()
        chat_history = str(self.memory.load_memory_variables({})["chat_history"]).lower()

        # Detect language
        language = "Python"  # Default
        for lang in self.app_config["supported_languages"]:
            if lang.lower() in query_lower or f"in {lang.lower()}" in query_lower:
                language = lang
                break

        # Detect framework
        framework = None
        supported_frameworks = self.app_config["supported_frameworks"].get(language, [])
        for fw in supported_frameworks:
            if fw.lower() in query_lower or f"in {fw.lower()}" in query_lower:
                framework = fw
                break
        if not framework:
            # Check chat history for framework mentions
            for fw in supported_frameworks:
                if fw.lower() in chat_history:
                    framework = fw
                    break
        if not framework and language == "Python":
            framework = "Flask"  # Default for Python
        elif not framework:
            framework = supported_frameworks[0] if supported_frameworks else None

        logger.debug(f"Detected language: {language}, framework: {framework}")
        return language, framework

    def generate_response(self, query: str) -> Dict[str, Any]:
        logger.info(f"Processing query: {query}")
        query_lower = query.lower()
        language, framework = self._detect_language_and_framework(query)
        logger.debug(f"Detected language: {language}, framework: {framework}")

        # Clear irrelevant history to prevent context mixing
        history = self.memory.load_memory_variables({})["chat_history"]
        if history and not any("webhook" in str(msg).lower() or "sdk" in str(msg).lower() for msg in history):
            logger.debug("Clearing irrelevant chat history")
            self.memory.clear()

        # Handle explicit code example requests
        if any(f"show {lang.lower()} example" in query_lower for lang in self.app_config["supported_languages"]):
            task = "webhook" if "webhook" in query_lower else "sdk"
            try:
                code = self._code_generator_tool(language, task, framework)
                response = {
                    "plan": {
                        "steps": [
                            {
                                "tool": "code_generator",
                                "params": {"language": language, "framework": framework, "task": task},
                                "description": f"Generate {language} code for {task} using {framework or 'default framework'}"
                            }
                        ]
                    },
                    "response": {
                        "text": f"Here's a {language} example for CreditChek API {task} using {framework or 'default framework'}:",
                        "code_snippets": {language: code}
                    },
                    "tool_results": [{"tool": "code_generator", "output": code}]
                }
                self.memory.save_context({"question": query}, {"answer": response["response"]["text"]})
                logger.info("Code generation response generated")
                return response
            except Exception as e:
                logger.error(f"Code generation failed for {language}/{framework}: {e}")
                return self._fallback_response(query, language, framework)

        try:
            # Generate plan: Use dynamic tool selection
            suggested_tools = self.select_tools(query)
            plan = {"steps": suggested_tools}

            # Execute plan
            results = []
            for step in plan.get("steps", []):
                tool = step.get("tool")
                params = step.get("params", {})
                logger.info(f"Executing tool: {tool} with params: {params}")
                try:
                    if tool == "document_retrieval":
                        docs = self._document_retrieval_tool(params.get("query", query))
                        results.append({"tool": "document_retrieval", "output": [doc.page_content for doc in docs]})
                    elif tool == "mock_creditchek_api":
                        api_response = self._mock_creditchek_api_tool(params.get("endpoint"), params.get("params"))
                        results.append({"tool": "mock_creditchek_api", "output": api_response})
                    elif tool == "code_generator":
                        code = self._code_generator_tool(
                            params.get("language", language),
                            params.get("task", query),
                            params.get("framework", framework)
                        )
                        results.append({"tool": "code_generator", "output": code})
                    else:
                        logger.warning(f"Unknown tool: {tool}")
                        results.append({"tool": tool, "output": f"Error: Unknown tool {tool}"})
                except Exception as e:
                    logger.error(f"Tool {tool} failed: {e}")
                    results.append({"tool": tool, "output": f"Error: {str(e)}"})

            # Skip reflection for simple queries (≤2 tools) to avoid JSON parsing issues
            if len(suggested_tools) <= 2:
                # Generate final response
                context = "\n".join([str(result["output"]) for result in results])
                final_prompt = PromptTemplate.from_template(
                    """You are Mark Musk, an AI assistant for CreditChek API integration.
                    Use the tool results to answer the query in {language} using the {framework} framework.
                    Provide code examples and suggest next steps.
                    Tool Results: {context}
                    Query: {query}
                    Chat History: {chat_history}
                    Answer in markdown format with sections:
                    - **Framework Choice**: Explain why {framework} was chosen (e.g., mentioned in query, inferred from chat history, or default).
                    - **Steps**: Step-by-step guide to address the query using {framework}.
                    - **Example**: Code example in {language} using {framework} (use ```{language}\n...\n```).
                    - **Additional Notes**: Clarifications, including alternative frameworks and how to adapt the code.
                    - **Next Steps**: Proactive suggestions for related tasks (e.g., testing API calls, setting up webhooks, processing payloads).
                    Ensure Next Steps are specific and actionable."""
                ).partial(language=language.lower(), framework=framework or "default framework")
                logger.info("Invoking LLM for final response")
                try:
                    final_response = self.llm.invoke(final_prompt.format(
                        context=context,
                        query=query,
                        chat_history=self.memory.load_memory_variables({})["chat_history"]
                    ))
                except Exception as llm_error:
                    logger.error(f"Final LLM invocation failed: {llm_error}")
                    return self._fallback_response(query, language, framework)
                response = {
                    "plan": plan,
                    "response": {
                        "text": final_response.content,
                        "code_snippets": {result["tool"]: result["output"] for result in results if result["tool"] == "code_generator"}
                    },
                    "tool_results": results
                }
                self.memory.save_context({"question": query}, {"answer": response["response"]["text"]})
                logger.info("Agentic response generated successfully")
                return response
            else:
                # For complex queries, fall back to LLM planning (without reflection)
                plan_input = {
                    "query": query,
                    "chat_history": self.memory.load_memory_variables({})["chat_history"]
                }
                logger.info("Invoking LLM for plan generation")
                try:
                    plan_response = self.llm.invoke(self.agent_prompt.format(**plan_input))
                    logger.debug(f"Raw LLM plan response: {plan_response.content!r}")
                except Exception as llm_error:
                    logger.error(f"LLM invocation failed: {llm_error}")
                    return self._fallback_response(query, language, framework)

                # Handle malformed JSON
                try:
                    content = plan_response.content.strip()
                    if content.startswith("```json") and content.endswith("```"):
                        content = content[7:-3].strip()
                    else:
                        json_match = re.search(r'\{[\s\S]*?\}', content, re.DOTALL)
                        if json_match:
                            content = json_match.group(0)
                        else:
                            raise ValueError("No JSON-like content found")
                    plan_data = json.loads(content)
                    if not isinstance(plan_data, dict) or "plan" not in plan_data:
                        raise ValueError("Invalid plan format: missing 'plan' key")
                    plan = plan_data.get("plan", {"steps": suggested_tools})
                except (json.JSONDecodeError, ValueError) as e:
                    logger.warning(f"LLM returned invalid JSON or format: {e}. Falling back to suggested tools.")
                    plan = {"steps": suggested_tools}

                # Execute plan
                results = []
                for step in plan.get("steps", []):
                    tool = step.get("tool")
                    params = step.get("params", {})
                    logger.info(f"Executing tool: {tool} with params: {params}")
                    try:
                        if tool == "document_retrieval":
                            docs = self._document_retrieval_tool(params.get("query", query))
                            results.append({"tool": "document_retrieval", "output": [doc.page_content for doc in docs]})
                        elif tool == "mock_creditchek_api":
                            api_response = self._mock_creditchek_api_tool(params.get("endpoint"), params.get("params"))
                            results.append({"tool": "mock_creditchek_api", "output": api_response})
                        elif tool == "code_generator":
                            code = self._code_generator_tool(
                                params.get("language", language),
                                params.get("task", query),
                                params.get("framework", framework)
                            )
                            results.append({"tool": "code_generator", "output": code})
                        else:
                            logger.warning(f"Unknown tool: {tool}")
                            results.append({"tool": tool, "output": f"Error: Unknown tool {tool}"})
                    except Exception as e:
                        logger.error(f"Tool {tool} failed: {e}")
                        results.append({"tool": tool, "output": f"Error: {str(e)}"})

                # Generate final response
                context = "\n".join([str(result["output"]) for result in results])
                final_prompt = PromptTemplate.from_template(
                    """You are Mark Musk, an AI assistant for CreditChek API integration.
                    Use the tool results to answer the query in {language} using the {framework} framework.
                    Provide code examples and suggest next steps.
                    Tool Results: {context}
                    Query: {query}
                    Chat History: {chat_history}
                    Answer in markdown format with sections:
                    - **Framework Choice**: Explain why {framework} was chosen (e.g., mentioned in query, inferred from chat history, or default).
                    - **Steps**: Step-by-step guide to address the query using {framework}.
                    - **Example**: Code example in {language} using {framework} (use ```{language}\n...\n```).
                    - **Additional Notes**: Clarifications, including alternative frameworks and how to adapt the code.
                    - **Next Steps**: Proactive suggestions for related tasks (e.g., testing API calls, setting up webhooks, processing payloads).
                    Ensure Next Steps are specific and actionable."""
                ).partial(language=language.lower(), framework=framework or "default framework")
                logger.info("Invoking LLM for final response")
                try:
                    final_response = self.llm.invoke(final_prompt.format(
                        context=context,
                        query=query,
                        chat_history=self.memory.load_memory_variables({})["chat_history"]
                    ))
                except Exception as llm_error:
                    logger.error(f"Final LLM invocation failed: {llm_error}")
                    return self._fallback_response(query, language, framework)
                response = {
                    "plan": plan,
                    "response": {
                        "text": final_response.content,
                        "code_snippets": {result["tool"]: result["output"] for result in results if result["tool"] == "code_generator"}
                    },
                    "tool_results": results
                }
                self.memory.save_context({"question": query}, {"answer": response["response"]["text"]})
                logger.info("Agentic response generated successfully")
                return response
        except Exception as e:
            logger.error(f"Agentic loop failed: {e}")
            return self._fallback_response(query, language, framework)

    def _fallback_response(self, query: str, language: str = "Python", framework: str = None) -> Dict[str, Any]:
        logger.info(f"Executing fallback response for query: {query} in {language}/{framework or 'default framework'}")
        try:
            docs = self._document_retrieval_tool(query)
            context = "\n".join([doc.page_content for doc in docs])
            logger.debug(f"Fallback context: {context}")
            code_snippet = {}
            code_text = f"No {language} code example available"
            task = "webhook" if "webhook" in query.lower() else "sdk"
            if "sdk" in query.lower() or "configure" in query.lower() or "webhook" in query.lower():
                code = self._code_generator_tool(language, task, framework)
                code_snippet = {language: code}
                code_text = code
            qa_prompt = PromptTemplate.from_template(
                """You are Mark Musk, an AI assistant for CreditChek API integration.
                Context: {context}
                Question: {query}
                Provide a detailed answer with steps, a {language} code example using {framework}, and next steps.
                Answer in markdown format with sections:
                - **Framework Choice**: Explain why {framework} was chosen (e.g., mentioned in query, inferred from chat history, or default).
                - **Steps**: Step-by-step guide to address the query using {framework}.
                - **Example**: Code example in {language} using {framework} (use ```{language}\n...\n```).
                - **Additional Notes**: Clarifications, including alternative frameworks and how to adapt the code.
                - **Next Steps**: Proactive suggestions for related tasks (e.g., testing API calls, setting up webhooks, processing payloads).
                Include the following code example in the Example section:
                {code_example}
                Answer:"""
            ).partial(language=language.lower(), framework=framework or "default framework")
            response = self.llm.invoke(qa_prompt.format(
                context=context,
                query=query,
                code_example=code_text
            ))
            logger.debug(f"Fallback LLM response: {response.content}")
            result = {
                "plan": {
                    "steps": [
                        {"tool": "document_retrieval", "params": {"query": query}, "description": "Retrieve documents"},
                        {"tool": "code_generator", "params": {"language": language, "framework": framework, "task": task}, "description": f"Generate {language} {task} example using {framework or 'default framework'}"} if code_snippet else {}
                    ]
                },
                "response": {
                    "text": response.content,
                    "code_snippets": code_snippet
                },
                "tool_results": [
                    {"tool": "document_retrieval", "output": [doc.page_content for doc in docs]},
                    {"tool": "code_generator", "output": code_snippet.get(language, "")} if code_snippet else {}
                ]
            }
            self.memory.save_context({"question": query}, {"answer": result["response"]["text"]})
            logger.info("Fallback response generated")
            return result
        except Exception as e:
            logger.error(f"Fallback response failed: {e}")
            return {
                "plan": {"steps": []},
                "response": {
                    "text": f"Error: Unable to generate response for '{query}'. Please check logs and ensure GOOGLE_API_KEY is valid.",
                    "code_snippets": {}
                },
                "tool_results": []
            }

    """def clear_index(self):
        try:
            index = self.pinecone_instance.Index(self.pinecone_config["PINECONE_INDEX"])
            index.delete(delete_all=True, namespace=self.pinecone_config.get("namespace", None))
            logger.info("Pinecone index cleared")
        except Exception as e:
            logger.error(f"Failed to clear index: {e}")"""

    def process_uploaded_doc(self, file_path: str):
        try:
            if file_path.endswith(".pdf"):
                loader = PyPDFLoader(file_path)
            elif file_path.endswith(".txt"):
                loader = TextLoader(file_path)
            else:
                raise ValueError("Unsupported file format")
            documents = loader.load()
            text_splitter = RecursiveCharacterTextSplitter(
                chunk_size=1000,
                chunk_overlap=200,
                length_function=len
            )
            split_docs = text_splitter.split_documents(documents)
            logger.info(f"Split uploaded document into {len(split_docs)} chunks")
            self._upload_documents_in_batches(split_docs, batch_size=50)
        except Exception as e:
            logger.error(f"Failed to process uploaded document: {e}")
            raise