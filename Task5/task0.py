from __future__ import annotations

import os
import re
import requests
from typing import Literal, List, Dict, TypedDict
from langgraph.graph import StateGraph, START, END
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer, util

load_dotenv()
OPENROUTER_API_KEY = os.getenv("OPENROUTER_API_KEY")
if not OPENROUTER_API_KEY:
    raise ValueError("OPENROUTER_API_KEY missing in .env")

MODEL_COST = {
    "cheap": "openai/gpt-4o-mini",
    "medium": "google/gemini-flash-1.5",
    "premium": "anthropic/claude-3.5-sonnet"
}

def openrouter_chat(model: str, messages: List[Dict], temperature: float = 0, max_tokens: int = 500):
    url = "https://openrouter.ai/api/v1/chat/completions"
    headers = {"Authorization": f"Bearer {OPENROUTER_API_KEY}", "Content-Type": "application/json"}
    payload = {"model": model, "messages": messages, "temperature": temperature, "max_tokens": max_tokens}
    try:
        resp = requests.post(url, json=payload, headers=headers, timeout=60)
        data = resp.json()
    except Exception as e:
        return f"Request error: {e}"
    if resp.status_code != 200:
        err = data.get("error") or data.get("message") or data.get("detail") or resp.text
        return f"LLM API error {resp.status_code}: {err}"
    if isinstance(data, dict) and "choices" in data:
        try:
            choice = data["choices"][0]
            if "message" in choice and "content" in choice["message"]:
                return choice["message"]["content"]
            if "text" in choice:
                return choice["text"]
            return str(choice)
        except:
            return str(data)
    return str(data)

def choose_model(remaining_credits=None) -> str:
    if remaining_credits is None:
        return MODEL_COST["cheap"]
    try:
        rem = float(remaining_credits)
    except:
        return MODEL_COST["cheap"]
    if rem < 5:
        return MODEL_COST["cheap"]
    elif rem < 20:
        return MODEL_COST["medium"]
    else:
        return MODEL_COST["premium"]

class AssistantState(TypedDict, total=False):
    user_input: str
    intent: Literal["generate_code", "explain_code", "unknown"]
    examples: List[Dict[str, str]]
    llm_result: str

EXAMPLE_LIBRARY: Dict[str, List[Dict[str, str]]] = {
    "generate_code": [
        {"title": "Generate Fibonacci function", "description": "Simple loop-based Fibonacci implementation",
         "code": "def fibonacci(n: int) -> int:\n    a, b = 0, 1\n    for _ in range(n):\n        a, b = b, a + b\n    return a\n"},
        {"title": "Basic calculator", "description": "Evaluate arithmetic expressions from user",
         "code": "def calc():\n    expr = input('Enter expression: ')\n    result = eval(expr)\n    print('Result:', result)\n\nif __name__ == '__main__':\n    calc()\n"}
    ],
    "explain_code": [
        {"title": "List comprehension example", "description": "Compact list creation pattern",
         "code": "squares = [x * x for x in range(10)]\n"},
        {"title": "Context manager usage", "description": "Open file using with-statement",
         "code": "with open('data.txt') as f:\n    for line in f:\n        print(line.strip())\n"}
    ],
}

WORD_RE = re.compile(r"[a-zA-Z_]+")

def classify_intent(state: AssistantState) -> AssistantState:
    text = state["user_input"].lower()
    if "explain" in text or "what does" in text or "شرح" in text:
        state["intent"] = "explain_code"
    elif any(kw in text for kw in ["generate", "write", "create", "build", "code for", "function for"]):
        state["intent"] = "generate_code"
    else:
        if any(tok in text for tok in ["def ", "for ", "while ", "import ", "class "]):
            state["intent"] = "explain_code"
        else:
            state["intent"] = "generate_code"
    return state

model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def retrieve_examples(state: AssistantState) -> AssistantState:
    intent = state.get("intent", "unknown")
    user_text = state["user_input"]
    candidates = EXAMPLE_LIBRARY.get(intent, [])
    if not candidates:
        state["examples"] = []
        return state
    texts = [ex["description"] + " " + ex["code"] for ex in candidates]
    embeddings = model.encode(texts, convert_to_tensor=True)
    query_embedding = model.encode([user_text], convert_to_tensor=True)
    scores = util.cos_sim(query_embedding, embeddings)[0]
    top_k = scores.topk(k=min(2, len(candidates)))
    state["examples"] = [candidates[i] for i in top_k.indices.tolist()]
    return state

def generate_code_node(state: AssistantState) -> AssistantState:
    examples_text = "".join(f"{ex['code']}\n" for ex in state.get("examples", []))
    system_prompt = "Return Python code inside triple backticks."
    user_prompt = f"{state['user_input']}\n\nUseful examples:\n{examples_text}"
    content = openrouter_chat(model=choose_model(None),
                              messages=[{"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user_prompt}],
                              temperature=0.2)
    state["llm_result"] = content
    return state

def explain_code_node(state: AssistantState) -> AssistantState:
    examples_text = "".join(f"{ex['code']}\n" for ex in state.get("examples", []))
    system_prompt = "Explain Python code step-by-step."
    user_prompt = f"{state['user_input']}\n\nUseful examples:\n{examples_text}"
    content = openrouter_chat(model=choose_model(None),
                              messages=[{"role": "system", "content": system_prompt},
                                        {"role": "user", "content": user_prompt}],
                              temperature=0.0)
    state["llm_result"] = content
    return state

def route_after_retrieval(state: AssistantState) -> str:
    i = state.get("intent", "unknown")
    if i == "generate_code":
        return "generate_code"
    if i == "explain_code":
        return "explain_code"
    return "END"

def build_graph():
    builder = StateGraph(AssistantState)
    builder.add_node("classify_intent", classify_intent)
    builder.add_node("retrieve_examples", retrieve_examples)
    builder.add_node("generate_code", generate_code_node)
    builder.add_node("explain_code", explain_code_node)
    builder.add_edge(START, "classify_intent")
    builder.add_edge("classify_intent", "retrieve_examples")
    builder.add_conditional_edges("retrieve_examples", route_after_retrieval,
                                  {"generate_code": "generate_code",
                                   "explain_code": "explain_code",
                                   "END": END})
    builder.add_edge("generate_code", END)
    builder.add_edge("explain_code", END)
    return builder.compile()

graph = build_graph()

def run_once(user_input: str) -> str:
    try:
        result: AssistantState = graph.invoke({"user_input": user_input})
    except Exception as e:
        return f"Graph error: {e}"
    return result.get("llm_result", "")

if __name__ == "__main__":
    print("LangGraph Python Assistant (RAG + OpenRouter)")
    while True:
        text = input("\nYou: ")
        if text.strip().lower() in {"quit", "exit"}:
            break
        print("\nAssistant:\n", run_once(text))
