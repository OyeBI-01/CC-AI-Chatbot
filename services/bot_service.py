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
        self.forced_language = None
        self.current_frameworks = []
        
        try:
            self.app_config = get_app_config()
            if not isinstance(self.app_config, dict):
                raise ValueError("get_app_config must return a dictionary")
                
            self.app_config["supported_frameworks"] = {
                "Python": ["Flask", "FastAPI", "Django"],
                "JavaScript": ["Express", "Koa", "NestJS"],
                "NodeJS": ["Express"],
                "PHP": ["Laravel", "Slim"],
                "Go": ["net/http", "Gin", "Echo"],
                "Java": ["Spring", "Jakarta EE"],
                "C#": [".NET", "ASP.NET Core"],
                "Ruby": ["Rails", "Sinatra"],
                "Rust": ["Actix", "Rocket"],
                "Swift": ["Vapor", "Perfect"],
                "Kotlin": ["Ktor", "Spring Boot"],
                "Other": ["Standard Library"]
            }
            
            if "supported_languages" not in self.app_config:
                self.app_config["supported_languages"] = list(self.app_config["supported_frameworks"].keys())
                
        except Exception as e:
            logger.error(f"Failed to load app_config: {e}")
            raise ValueError("Invalid app configuration")
            
        try:
            self.pinecone_config = get_pinecone_config()
            if not isinstance(self.pinecone_config, dict):
                raise ValueError("get_pinecone_config must return a dictionary")
            if not self.pinecone_config.get("PINECONE_INDEX"):
                self.pinecone_config["PINECONE_INDEX"] = "creditchek-docs"
        except Exception as e:
            logger.error(f"Failed to load pinecone_config: {e}")
            raise

        self.pinecone_instance = Pinecone(api_key=os.getenv("PINECONE_API_KEY") or self.pinecone_config.get("api_key"))
        self.index = self.create_index()
        self.embeddings = GoogleGenerativeAIEmbeddings(
            model="models/text-embedding-004",
            google_api_key=os.getenv("GOOGLE_API_KEY")
        )
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
        self.init_document_store()

    def set_language_preference(self, language: str):
        """Force the bot to use a specific programming language"""
        self.forced_language = language
        logger.info(f"Language preference set to: {language}")
        
        if language in self.app_config["supported_frameworks"]:
            self.current_frameworks = self.app_config["supported_frameworks"][language]
        else:
            self.current_frameworks = ["Standard Library"]
        logger.debug(f"Available frameworks: {self.current_frameworks}")

    def create_index(self):
        """Create or connect to Pinecone index"""
        index_name = self.pinecone_config["PINECONE_INDEX"]
        try:
            existing_indexes = [index["name"] for index in self.pinecone_instance.list_indexes()]
            if index_name not in existing_indexes:
                self.pinecone_instance.create_index(
                    name=index_name,
                    dimension=768,
                    metric="cosine",
                    spec=ServerlessSpec(
                        cloud="aws",
                        region=self.pinecone_config.get("PINECONE_ENVIRONMENT", "us-east1-aws")
                    )
                )
                time.sleep(1)
            return self.pinecone_instance.Index(index_name)
        except Exception as e:
            logger.error(f"Failed to create/connect to index: {e}")
            raise


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
        """Generate code snippets for any programming language with graceful fallback"""
        language = language.lower()
        framework = framework.lower() if framework else None
        
        # First check if we have a specific implementation
        specific_code = self._get_specific_implementation(language, task, framework)
        if specific_code:
            return specific_code
            
        # Generic implementation for unsupported languages
        return self._generate_generic_code(language.capitalize(), task, framework)

    def _get_specific_implementation(self, language: str, task: str, framework: str) -> Optional[str]:
        """Handle specific language implementations"""
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
        return {"status": "OK"}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))"""
            elif framework == "django":
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
    request_body = request.body()
    x_auth_signature = request.META.get("HTTP_X_AUTH_SIGNATURE")
    live_secret_key = os.environ.get("CREDITCHEK_LIVE_SECRET_KEY")

    if not live_secret_key:
        return JsonResponse({"error": "CREDITCHEK_LIVE_SECRET_KEY environment variable not set"}, status=500)

    if not verify_webhook_signature(request_body, x_auth_signature, live_secret_key):
        return JsonResponse({"error": "Invalid signature"}, status=401)

    try:
        data = json.loads(request_body)
        print(f"Webhook received: {data}")
        return HttpResponse("OK", status=200)
    except Exception as e:
        return JsonResponse({"error": str(e)}, status=500)"""
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
    print(f"Error retrieving repayment schedule: {e}")"""
                
        elif language in ["javascript", "nodejs"] and "webhook" in task.lower():
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
                
        elif language in ["javascript", "nodejs"] and ("sdk" in task.lower() or "configure" in task.lower()):
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
    .catch(err => console.error(err));"""
                
        elif language == "php" and "webhook" in task.lower():
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

        if (!hash_equals($signature, $xAuthSignature)) {
            return response()->json(['error' => 'Invalid signature'], 401);
        }

        \Log::info('Webhook received: ' . json_encode($request->all()));
        return response('OK', 200);
    }
}"""
                
        elif language == "php" and ("sdk" in task.lower() or "configure" in task.lower()):
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
}"""
                
        elif language == "go" and "webhook" in task.lower():
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

    if !verifyWebhookSignature(requestBody, xAuthSignature, liveSecretKey) {
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
}"""
                
        elif language == "go" and ("sdk" in task.lower() or "configure" in task.lower()):
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
}"""
                
        return None

    def _generate_generic_code(self, language: str, task: str, framework: str = None) -> str:
        """Generate generic code template for unsupported languages"""
        if "webhook" in task.lower():
            return f"""// {language} webhook implementation ({framework or 'standard'} framework)
// 1. Set up a web server that can handle HTTP POST requests
// 2. Create an endpoint to receive CreditChek webhook notifications
// 3. Verify the x-auth-signature header using your LiveSecretKey
//    - Compare the signature using HMAC-SHA256
// 4. Process the incoming webhook data securely
        
// Sample payload structure:
// {{
//   "transaction_id": "123",
//   "status": "completed|failed|pending",
//   "amount": 100.50,
//   "timestamp": "2025-05-08T12:00:00Z"
// }}

// Security considerations:
// - Always validate the webhook signature
// - Use HTTPS for your webhook endpoint
// - Implement proper error handling
// - Consider rate limiting to prevent abuse
// - Store the LiveSecretKey securely (environment variables/secret manager)

// Next steps:
// 1. Register your webhook URL in the CreditChek dashboard
// 2. Test with sample payloads
// 3. Implement your business logic for handling transactions"""
        
        elif "sdk" in task.lower() or "configure" in task.lower():
            return f"""// {language} SDK usage example for CreditChek API
// 1. Install the appropriate HTTP client/library for {language}
// 2. Configure authentication with your API key:
//    - Store the API key securely (environment variables/secret manager)
//    - Include in requests as: Authorization: Bearer YOUR_API_KEY

// Example API endpoints:
// - Identity verification: POST /api/v1/identity
// - Repayment schedule: GET /api/v1/repayments?customer_id=123
// - Transaction status: GET /api/v1/transactions/123

// Sample request (pseudo-code):
// headers = {{
//   "Authorization": "Bearer " + os.getenv("CREDITCHEK_API_KEY"),
//   "Content-Type": "application/json"
// }}
// response = http.post("https://api.creditchek.africa/api/v1/identity", {{
//   "first_name": "John",
//   "last_name": "Doe",
//   "dob": "1985-01-01",
//   "id_number": "ID123456"
// }}, headers)

// Documentation:
// - Full API reference: https://docs.creditchek.africa
// - SDK installation: Check if an official {language} SDK exists
// - Community libraries: Search for "CreditChek {language} SDK"

// Error handling tips:
// - Check for 401 Unauthorized (invalid API key)
// - Handle rate limits (429 Too Many Requests)
// - Implement retries for transient failures"""
        
        return f"""// {language} implementation for {task}
// Framework: {framework or 'standard library'}

// This is a generic template. For {language}-specific implementation:
// 1. Consult the {language} documentation for HTTP server/client setup
// 2. Review CreditChek API documentation at https://docs.creditchek.africa
// 3. Adapt the patterns from our other language examples

// Key requirements:
// - Secure API key management
// - Proper error handling
// - HTTPS for all communications
// - Webhook signature verification (if applicable)

// For framework-specific guidance in {language}, consider:
// {', '.join(self.app_config["supported_frameworks"].get(language, ["Standard Library"]))}"""

    def _detect_language_and_framework(self, query: str) -> tuple[str, Optional[str]]:
        """Detect language and framework with preference for forced_language"""
        query_lower = query.lower()
        
        # Use forced language if set
        if self.forced_language:
            language = self.forced_language
            logger.debug(f"Using forced language: {language}")
        else:
            language = "Python"  # Default
            for lang in self.app_config["supported_languages"]:
                if lang.lower() in query_lower or f"in {lang.lower()}" in query_lower:
                    language = lang
                    break

        # Framework detection
        framework = None
        supported_frameworks = self.app_config["supported_frameworks"].get(language, [])
        
        # First check query for framework mentions
        for fw in supported_frameworks:
            if fw.lower() in query_lower:
                framework = fw
                break
                
        # Then check current frameworks if none found
        if not framework and self.current_frameworks:
            framework = self.current_frameworks[0]

        logger.debug(f"Detected language: {language}, framework: {framework}")
        return language, framework
        

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