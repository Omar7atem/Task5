import os
from dotenv import load_dotenv
from flask import Flask, request, jsonify, render_template_string
from typing import List, Dict, Tuple
import numpy as np

from langchain_community.chat_models import ChatOpenAI
from langchain_openai import OpenAIEmbeddings
from langchain_chroma import Chroma
from langchain_core.chat_history import InMemoryChatMessageHistory
from langchain_core.prompts import ChatPromptTemplate, MessagesPlaceholder
from langchain_core.documents import Document
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_core.runnables import RunnablePassthrough
from langchain_core.output_parsers import StrOutputParser

from sentence_transformers import SentenceTransformer
from datasets import load_dataset
import faiss
import requests

load_dotenv()
api_key = os.getenv("OPENROUTER_API_KEY")
if not api_key:
    raise ValueError("OPENROUTER_API_KEY missing in .env")

os.environ["OPENAI_API_KEY"] = api_key
os.environ["OPENAI_API_BASE"] = "https://openrouter.ai/api/v1"

app = Flask(__name__)


class RAGEvaluator:
    @staticmethod
    def precision_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        if k == 0 or len(retrieved) == 0:
            return 0.0
        top_k = retrieved[:k]
        relevant_retrieved = len(set(top_k) & set(relevant))
        return relevant_retrieved / k
    
    @staticmethod
    def recall_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        if len(relevant) == 0:
            return 0.0
        top_k = retrieved[:k]
        relevant_retrieved = len(set(top_k) & set(relevant))
        return relevant_retrieved / len(relevant)
    
    @staticmethod
    def f1_at_k(retrieved: List[str], relevant: List[str], k: int) -> float:
        precision = RAGEvaluator.precision_at_k(retrieved, relevant, k)
        recall = RAGEvaluator.recall_at_k(retrieved, relevant, k)
        if precision + recall == 0:
            return 0.0
        return 2 * (precision * recall) / (precision + recall)
    
    @staticmethod
    def reciprocal_rank(retrieved: List[str], relevant: List[str]) -> float:
        for rank, doc_id in enumerate(retrieved, start=1):
            if doc_id in relevant:
                return 1.0 / rank
        return 0.0


class CompleteRAGSystem:
    def __init__(self, persist_directory: str = "./flask_rag_db"):
        self.persist_directory = persist_directory
        
        self.embeddings = OpenAIEmbeddings(
            model="text-embedding-3-small",
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        self.llm = ChatOpenAI(
            model="openai/gpt-4o-mini",
            temperature=0.7,
            openai_api_key=api_key,
            openai_api_base="https://openrouter.ai/api/v1"
        )
        
        self.vectorstore = Chroma(
            persist_directory=persist_directory,
            embedding_function=self.embeddings
        )
        
        self.memory = InMemoryChatMessageHistory()
        
        self.sentence_transformer = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
        self.faiss_index = None
        self.faiss_data = []
        self.humaneval_dataset = None
        
        system_prompt = """You are an intelligent AI assistant with the following capabilities:

Core Functions:
- Access to a knowledge base via vector search
- Remember conversation history and context
- Provide accurate, well-sourced answers
- Generate and explain code when needed

Response Style:
- Clear and concise
- Reference sources when available
- Admit when you don't know something
- Build on previous conversation context
"""
        
        self.prompt = ChatPromptTemplate.from_messages([
            ("system", system_prompt),
            MessagesPlaceholder(variable_name="chat_history"),
            ("human", "{input}")
        ])
        
        self.parser = StrOutputParser()
        
        self.chain = (
            {
                "input": RunnablePassthrough(),
                "chat_history": lambda x: self.memory.messages
            }
            | self.prompt
            | self.llm
            | self.parser
        )
        
        self.doc_ids = []
        
        print(f"✓ Complete RAG System initialized")
        print(f"✓ Vector DB: {persist_directory}")
        print(f"✓ Documents in Chroma: {self.vectorstore._collection.count()}")
    
    def add_documents(self, texts: List[str], doc_ids: List[str] = None, metadatas: List[dict] = None):
        if doc_ids is None:
            doc_ids = [f"doc_{i}" for i in range(len(texts))]
        
        documents = []
        for i, (text, doc_id) in enumerate(zip(texts, doc_ids)):
            metadata = {"doc_id": doc_id}
            if metadatas and i < len(metadatas):
                metadata.update(metadatas[i])
            doc = Document(page_content=text, metadata=metadata)
            documents.append(doc)
        
        self.vectorstore.add_documents(documents)
        self.doc_ids.extend(doc_ids)
        
        return {
            "status": "success",
            "added": len(texts),
            "total": self.vectorstore._collection.count()
        }
    
    def add_documents_from_file(self, file_path: str, chunk_size: int = 1000):
        with open(file_path, 'r', encoding='utf-8') as f:
            text = f.read()
        
        splitter = RecursiveCharacterTextSplitter(
            chunk_size=chunk_size,
            chunk_overlap=200
        )
        chunks = splitter.split_text(text)
        
        return self.add_documents(
            chunks,
            metadatas=[{"source": file_path, "chunk": i} for i in range(len(chunks))]
        )
    
    def load_humaneval_dataset(self):
        print("📥 Loading HumanEval dataset...")
        self.humaneval_dataset = load_dataset("openai/openai_humaneval", split="test")
        tasks = []
        for example in self.humaneval_dataset:
            tasks.append({
                "task_id": example["task_id"],
                "prompt": example["prompt"],
                "canonical_solution": example["canonical_solution"]
            })
        return tasks
    
    def build_faiss_index(self, texts: List[str], metadata: List[dict] = None):
        print(f"🔨 Building FAISS index with {len(texts)} documents...")
        embeddings = self.sentence_transformer.encode(texts)
        dim = embeddings.shape[1]
        self.faiss_index = faiss.IndexFlatL2(dim)
        self.faiss_index.add(np.array(embeddings).astype('float32'))
        self.faiss_data = [{"text": text, "metadata": meta} for text, meta in zip(texts, metadata or [{}]*len(texts))]
        print(f"✓ FAISS index built with {len(texts)} documents")
        return {"status": "success", "indexed": len(texts)}
    
    def search_faiss(self, query: str, top_k: int = 3):
        if self.faiss_index is None:
            return {"error": "FAISS index not built yet"}
        query_emb = self.sentence_transformer.encode([query])
        D, I = self.faiss_index.search(np.array(query_emb).astype('float32'), top_k)
        results = [self.faiss_data[i] for i in I[0] if i < len(self.faiss_data)]
        return results
    
    def generate_code(self, query: str, context: str, max_tokens: int = 256):
        url = "https://openrouter.ai/api/v1/chat/completions"
        headers = {
            "Authorization": f"Bearer {api_key}",
            "Content-Type": "application/json"
        }
        payload = {
            "model": "meta-llama/llama-3.1-8b-instruct",
            "messages": [
                {"role": "system", "content": "You are a helpful AI coding assistant."},
                {"role": "user", "content": f"### Instruction:\n{query}\n\n### Context:\n{context}\n\n### Response:\n"}
            ],
            "max_tokens": max_tokens,
            "temperature": 0.7
        }
        
        try:
            response = requests.post(url, headers=headers, json=payload, timeout=60)
            result = response.json()
            return result["choices"][0]["message"]["content"]
        except Exception as e:
            return f"Error generating code: {e}"
    
    def retrieve(self, query: str, k: int = 5) -> Tuple[List[str], List[Document]]:
        results = self.vectorstore.similarity_search(query, k=k)
        retrieved_ids = [doc.metadata.get("doc_id", "unknown") for doc in results]
        return retrieved_ids, results
    
    def chat(self, user_input: str, use_rag: bool = True) -> str:
        if use_rag and self.vectorstore._collection.count() > 0:
            _, relevant_docs = self.retrieve(user_input, k=3)
            context = "\n\n".join([doc.page_content for doc in relevant_docs])
            enhanced_input = f"Context:\n{context}\n\nQuestion: {user_input}"
        else:
            enhanced_input = user_input
        
        response = self.chain.invoke(enhanced_input)
        
        self.memory.add_user_message(user_input)
        self.memory.add_message({"role": "assistant", "content": response})
        
        return response
    
    def ask_with_sources(self, question: str) -> Dict:
        retriever = self.vectorstore.as_retriever(search_kwargs={"k": 3})
        
        def format_docs(docs):
            return "\n\n".join(doc.page_content for doc in docs)
        
        prompt = ChatPromptTemplate.from_template(
            """Use the following context to answer the question.

Context:
{context}

Question: {question}

Answer:"""
        )
        
        chain = (
            {"context": retriever | format_docs, "question": RunnablePassthrough()}
            | prompt
            | self.llm
        )
        
        response = chain.invoke(question)
        docs = retriever.invoke(question)
        
        return {
            "answer": response.content,
            "sources": [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
        }
    
    def evaluate_query(self, query: str, ground_truth: List[str], k: int = 5) -> Dict[str, float]:
        retrieved_ids, _ = self.retrieve(query, k=k)
        return {
            "precision@k": RAGEvaluator.precision_at_k(retrieved_ids, ground_truth, k),
            "recall@k": RAGEvaluator.recall_at_k(retrieved_ids, ground_truth, k),
            "f1@k": RAGEvaluator.f1_at_k(retrieved_ids, ground_truth, k),
            "mrr": RAGEvaluator.reciprocal_rank(retrieved_ids, ground_truth)
        }
    
    def clear_memory(self):
        self.memory.clear()
    
    def get_stats(self) -> Dict:
        return {
            "total_documents": self.vectorstore._collection.count(),
            "persist_directory": self.persist_directory,
            "conversation_messages": len(self.memory.messages),
            "faiss_documents": len(self.faiss_data),
            "humaneval_loaded": self.humaneval_dataset is not None
        }


rag_system = CompleteRAGSystem()


HTML_TEMPLATE = """
<!DOCTYPE html>
<html>
<head>
    <title>Complete RAG System</title>
    <style>
        body { font-family: Arial; max-width: 1200px; margin: 50px auto; padding: 20px; }
        .container { display: grid; grid-template-columns: 1fr 1fr; gap: 20px; }
        .section { border: 1px solid #ddd; padding: 20px; border-radius: 8px; }
        h1 { text-align: center; color: #333; }
        h2 { color: #555; }
        textarea { width: 100%; height: 100px; padding: 10px; }
        input[type="text"] { width: 100%; padding: 10px; margin: 5px 0; }
        button { background: #007bff; color: white; padding: 10px 20px; border: none; cursor: pointer; border-radius: 5px; }
        button:hover { background: #0056b3; }
        .response { background: #f8f9fa; padding: 15px; border-radius: 5px; margin-top: 10px; white-space: pre-wrap; }
        .stats { background: #e9ecef; padding: 10px; border-radius: 5px; }
    </style>
</head>
<body>
    <h1>🚀 Complete RAG System</h1>
    
    <div class="container">
        <div class="section">
            <h2>💬 Chat with RAG</h2>
            <textarea id="chatInput" placeholder="Ask a question..."></textarea>
            <button onclick="chat()">Send</button>
            <div id="chatResponse" class="response"></div>
        </div>
        
        <div class="section">
            <h2>📚 Ask with Sources</h2>
            <input type="text" id="sourceQuery" placeholder="Enter question...">
            <button onclick="askWithSources()">Get Answer</button>
            <div id="sourceResponse" class="response"></div>
        </div>
        
        <div class="section">
            <h2>➕ Add Documents</h2>
            <textarea id="addDocText" placeholder="Enter document text..."></textarea>
            <button onclick="addDocument()">Add to Knowledge Base</button>
            <div id="addDocResponse" class="response"></div>
        </div>
        
        <div class="section">
            <h2>📊 System Statistics</h2>
            <button onclick="getStats()">Refresh Stats</button>
            <div id="statsResponse" class="stats"></div>
        </div>
        
        <div class="section">
            <h2>🔍 Search Documents</h2>
            <input type="text" id="searchQuery" placeholder="Search query...">
            <button onclick="search()">Search</button>
            <div id="searchResponse" class="response"></div>
        </div>
        
        <div class="section">
            <h2>📈 Evaluate Retrieval</h2>
            <input type="text" id="evalQuery" placeholder="Query...">
            <input type="text" id="evalGroundTruth" placeholder="Relevant doc IDs (comma-separated)...">
            <button onclick="evaluate()">Evaluate</button>
            <div id="evalResponse" class="response"></div>
        </div>
        
        <div class="section">
            <h2>💻 Generate Code (FAISS + Llama)</h2>
            <input type="text" id="codeQuery" placeholder="Code request...">
            <button onclick="generateCode()">Generate</button>
            <div id="codeResponse" class="response"></div>
        </div>
        
        <div class="section">
            <h2>🔧 FAISS Operations</h2>
            <button onclick="loadHumanEval()">Load HumanEval Dataset</button>
            <button onclick="buildFAISS()">Build FAISS Index</button>
            <input type="text" id="faissSearch" placeholder="Search FAISS...">
            <button onclick="searchFAISS()">Search</button>
            <div id="faissResponse" class="response"></div>
        </div>
    </div>
    
    <script>
        async function chat() {
            const input = document.getElementById('chatInput').value;
            const response = await fetch('/api/chat', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: input})
            });
            const data = await response.json();
            document.getElementById('chatResponse').innerText = data.response;
        }
        
        async function askWithSources() {
            const query = document.getElementById('sourceQuery').value;
            const response = await fetch('/api/ask_sources', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            });
            const data = await response.json();
            let output = 'Answer: ' + data.answer + '\\n\\nSources:\\n';
            data.sources.forEach((s, i) => {
                output += `[${i+1}] ${s.content.substring(0, 100)}...\\n`;
            });
            document.getElementById('sourceResponse').innerText = output;
        }
        
        async function addDocument() {
            const text = document.getElementById('addDocText').value;
            const response = await fetch('/api/add_document', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({text: text})
            });
            const data = await response.json();
            document.getElementById('addDocResponse').innerText = JSON.stringify(data, null, 2);
        }
        
        async function getStats() {
            const response = await fetch('/api/stats');
            const data = await response.json();
            document.getElementById('statsResponse').innerText = JSON.stringify(data, null, 2);
        }
        
        async function search() {
            const query = document.getElementById('searchQuery').value;
            const response = await fetch('/api/search', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query, k: 3})
            });
            const data = await response.json();
            let output = 'Results:\\n';
            data.results.forEach((r, i) => {
                output += `[${i+1}] ${r.content.substring(0, 100)}...\\n`;
            });
            document.getElementById('searchResponse').innerText = output;
        }
        
        async function evaluate() {
            const query = document.getElementById('evalQuery').value;
            const groundTruth = document.getElementById('evalGroundTruth').value.split(',').map(s => s.trim());
            const response = await fetch('/api/evaluate', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query, ground_truth: groundTruth, k: 5})
            });
            const data = await response.json();
            document.getElementById('evalResponse').innerText = JSON.stringify(data, null, 2);
        }
        
        async function generateCode() {
            const query = document.getElementById('codeQuery').value;
            document.getElementById('codeResponse').innerText = 'Generating code...';
            const response = await fetch('/api/generate_code', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query})
            });
            const data = await response.json();
            document.getElementById('codeResponse').innerText = data.code || data.error;
        }
        
        async function loadHumanEval() {
            document.getElementById('faissResponse').innerText = 'Loading dataset...';
            const response = await fetch('/api/load_humaneval', {method: 'POST'});
            const data = await response.json();
            document.getElementById('faissResponse').innerText = JSON.stringify(data, null, 2);
        }
        
        async function buildFAISS() {
            document.getElementById('faissResponse').innerText = 'Building FAISS index...';
            const response = await fetch('/api/build_faiss', {method: 'POST'});
            const data = await response.json();
            document.getElementById('faissResponse').innerText = JSON.stringify(data, null, 2);
        }
        
        async function searchFAISS() {
            const query = document.getElementById('faissSearch').value;
            const response = await fetch('/api/search_faiss', {
                method: 'POST',
                headers: {'Content-Type': 'application/json'},
                body: JSON.stringify({query: query, k: 3})
            });
            const data = await response.json();
            let output = 'FAISS Results:\\n';
            if (data.results) {
                data.results.forEach((r, i) => {
                    output += `[${i+1}] ${r.text.substring(0, 100)}...\\n`;
                });
            } else {
                output = data.error || JSON.stringify(data);
            }
            document.getElementById('faissResponse').innerText = output;
        }
    </script>
</body>
</html>
"""


@app.route('/')
def home():
    return render_template_string(HTML_TEMPLATE)


@app.route('/api/chat', methods=['POST'])
def api_chat():
    data = request.json
    query = data.get('query', '')
    response = rag_system.chat(query, use_rag=True)
    return jsonify({"response": response})


@app.route('/api/ask_sources', methods=['POST'])
def api_ask_sources():
    data = request.json
    query = data.get('query', '')
    result = rag_system.ask_with_sources(query)
    return jsonify(result)


@app.route('/api/add_document', methods=['POST'])
def api_add_document():
    data = request.json
    text = data.get('text', '')
    result = rag_system.add_documents([text])
    return jsonify(result)


@app.route('/api/search', methods=['POST'])
def api_search():
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 3)
    _, docs = rag_system.retrieve(query, k=k)
    results = [{"content": doc.page_content, "metadata": doc.metadata} for doc in docs]
    return jsonify({"results": results})


@app.route('/api/evaluate', methods=['POST'])
def api_evaluate():
    data = request.json
    query = data.get('query', '')
    ground_truth = data.get('ground_truth', [])
    k = data.get('k', 5)
    metrics = rag_system.evaluate_query(query, ground_truth, k)
    return jsonify(metrics)


@app.route('/api/stats', methods=['GET'])
def api_stats():
    return jsonify(rag_system.get_stats())


@app.route('/api/clear_memory', methods=['POST'])
def api_clear_memory():
    rag_system.clear_memory()
    return jsonify({"status": "success", "message": "Memory cleared"})


@app.route('/api/generate_code', methods=['POST'])
def api_generate_code():
    data = request.json
    query = data.get('query', '')
    
    if rag_system.faiss_index is not None:
        results = rag_system.search_faiss(query, top_k=3)
        context = "\n\n".join([r.get("text", "") for r in results])
    else:
        context = "No context available. Build FAISS index first."
    
    code = rag_system.generate_code(query, context)
    return jsonify({"code": code, "context_used": len(context)})


@app.route('/api/load_humaneval', methods=['POST'])
def api_load_humaneval():
    try:
        tasks = rag_system.load_humaneval_dataset()
        return jsonify({"status": "success", "tasks_loaded": len(tasks), "sample": tasks[0] if tasks else None})
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/build_faiss', methods=['POST'])
def api_build_faiss():
    try:
        if rag_system.humaneval_dataset is None:
            tasks = rag_system.load_humaneval_dataset()
        else:
            tasks = []
            for example in rag_system.humaneval_dataset:
                tasks.append({
                    "task_id": example["task_id"],
                    "prompt": example["prompt"],
                    "canonical_solution": example["canonical_solution"]
                })
        
        prompts = [t["prompt"] for t in tasks]
        metadata = [{"task_id": t["task_id"]} for t in tasks]
        result = rag_system.build_faiss_index(prompts, metadata)
        return jsonify(result)
    except Exception as e:
        return jsonify({"status": "error", "message": str(e)})


@app.route('/api/search_faiss', methods=['POST'])
def api_search_faiss():
    data = request.json
    query = data.get('query', '')
    k = data.get('k', 3)
    results = rag_system.search_faiss(query, top_k=k)
    return jsonify({"results": results})


if __name__ == "__main__":
    print("\n" + "="*70)
    print("🚀 Complete RAG System with Flask")
    print("="*70)
    print("\n✓ System initialized successfully")
    print("✓ Using OpenRouter API for ALL operations")
    print("✓ Models:")
    print("    - Chat: openai/gpt-4o-mini")
    print("    - Embeddings: text-embedding-3-small")
    print("    - Code Gen: meta-llama/llama-3.1-8b-instruct")
    print("✓ Open your browser and go to: http://127.0.0.1:5000")
    print("\nFeatures:")
    print("  • Chat with RAG (memory + vector search)")
    print("  • Ask with source citations")
    print("  • Add documents to knowledge base")
    print("  • Search and retrieve documents (Chroma + FAISS)")
    print("  • Evaluate retrieval quality (Precision, Recall, F1, MRR)")
    print("  • Generate code with context (Llama 3.1 + FAISS)")
    print("  • Load HumanEval dataset for code generation")
    print("  • View system statistics")
    print("="*70 + "\n")
    
    app.run(debug=True, port=5000)