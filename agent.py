import os
import json
import time
import pickle
import logging
from typing import List, TypedDict, Dict, Any
from dotenv import load_dotenv
from pydantic import BaseModel, Field

# --- SILENCE AFC WARNING ---
logging.getLogger("google_genai.models").setLevel(logging.ERROR)

# 1. Load Environment Variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

# 2. Imports
from langchain_core.documents import Document
from langchain_core.messages import SystemMessage, HumanMessage, ToolMessage
from langchain_core.tools import tool
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings
from langchain_chroma import Chroma
from langchain_community.retrievers import BM25Retriever
from langchain_classic.retrievers import EnsembleRetriever, ContextualCompressionRetriever
from langchain_community.document_compressors.flashrank_rerank import FlashrankRerank
from langgraph.graph import StateGraph, END
import numexpr as ne

# Configuration 
CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "models/gemini-embedding-001"
COST_PER_1M_INPUT = 0.075  
COST_PER_1M_OUTPUT = 0.30

# --- RATE LIMIT SAFEGUARD ---
def execute_with_retry(func, *args, **kwargs):
    max_retries = 3
    for attempt in range(max_retries):
        try:
            return func(*args, **kwargs)
        except Exception as e:
            error_msg = str(e)
            if "429" in error_msg or "RESOURCE_EXHAUSTED" in error_msg or "10054" in error_msg:
                print(f"\n⚠️ Google API Limit/Network Drop Hit! Pausing for 60s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(60)
            else:
                raise e
    raise Exception("Max retries exceeded. Please wait a few minutes and try again.")

# --- 3. Setup Hybrid Retriever (OPTIMIZED WITH BM25 CACHING) ---
def setup_hybrid_retriever():
    print("Initializing Enhanced Hybrid Retriever (Multi-Page Capable)...")
    BM25_CACHE_FILE = "bm25_index.pkl"
    
    # A. Load Vector Search (Instantly from disk)
    embeddings = GoogleGenerativeAIEmbeddings(model=EMBEDDING_MODEL, google_api_key=api_key)
    chroma_db = Chroma(persist_directory=CHROMA_PERSIST_DIR, embedding_function=embeddings)
    vector_retriever = chroma_db.as_retriever(search_kwargs={"k": 10})

    # B. Load or Build Keyword Search (BM25)
    if os.path.exists(BM25_CACHE_FILE):
        print("⚡ Loading cached BM25 Index from disk (Instant)...")
        with open(BM25_CACHE_FILE, "rb") as f:
            bm25_retriever = pickle.load(f)
    else:
        print("⏳ Building BM25 Index for the first time (This will take ~30s but will be cached)...")
        with open("varsity_docs.json", "r", encoding="utf-8") as f:
            raw_docs = json.load(f)

        markdown_splitter = MarkdownHeaderTextSplitter(headers_to_split_on=[("#", "H1"), ("##", "H2"), ("###", "H3")], strip_headers=False)
        text_splitter = RecursiveCharacterTextSplitter(chunk_size=1200, chunk_overlap=150, separators=["\n\n", "\n", " ", ""])
        
        docs = []
        for item in raw_docs:
            url = item.get("url", "")
            title = item.get("title", "")
            content = item.get("content", "").strip()
            if not content: continue
                
            for cutoff in ["Key Takeaways", "Key takeaways from this chapter"]:
                if cutoff in content:
                    content = content[:content.find(cutoff) + 1500] 
                    break

            for split_doc in markdown_splitter.split_text(content):
                headers = [split_doc.metadata.get("H1"), split_doc.metadata.get("H2"), split_doc.metadata.get("H3")]
                breadcrumb = " > ".join([h for h in headers if h])
                for sub_text in text_splitter.split_text(split_doc.page_content):
                    ctx = f"[{title} | {breadcrumb}]\n" if breadcrumb else f"[{title}]\n"
                    docs.append(Document(page_content=ctx + sub_text, metadata={"source": url, "title": title}))
                    
        bm25_retriever = BM25Retriever.from_documents(docs)
        bm25_retriever.k = 20
        vector_retriever = chroma_db.as_retriever(search_kwargs={"k": 20})
        # Save to disk so we NEVER have to do this again!
        with open(BM25_CACHE_FILE, "wb") as f:
            pickle.dump(bm25_retriever, f)
        print("✅ BM25 Index Cached successfully!")
    
    # C. Combine and set up FlashRank
    ensemble = EnsembleRetriever(retrievers=[bm25_retriever, vector_retriever], weights=[0.5, 0.5])
    return ContextualCompressionRetriever(base_compressor=FlashrankRerank(top_n=7), base_retriever=ensemble)

hybrid_retriever = setup_hybrid_retriever()

# --- 4. Tool Definition ---
@tool
def calculate(expression: str) -> str:
    """Evaluates a mathematical expression and returns the exact result. Example input: '100 * (60 - 50)'"""
    try:
        # numexpr safely and instantly evaluates math strings
        result = ne.evaluate(expression).item() 
        print(f"\n   [Tool Executed] calculate('{expression}') -> {result}")
        return str(result)
    except Exception as e:
        return f"Math Error: {e}"

# --- 5. Define LangGraph State & Nodes ---
class GraphState(TypedDict):
    original_question: str
    standalone_question: str
    chat_history: List[Dict[str, str]]
    documents: List[Document]
    llm_messages: List[Any]  # Stores internal dialog between LLM and Tools
    answer: str
    sources: List[str]
    is_grounded: str 
    input_tokens: int
    output_tokens: int

def get_usage(response):
    if hasattr(response, 'usage_metadata') and response.usage_metadata:
        return response.usage_metadata.get('input_tokens', 0), response.usage_metadata.get('output_tokens', 0)
    return 0, 0

def rewrite_query_node(state: GraphState):
    in_tok, out_tok = 0, 0
    if not state.get("chat_history"):
        return {"standalone_question": state["original_question"], "input_tokens": in_tok, "output_tokens": out_tok}
        
    history_str = "\n".join([f"{msg['role'].capitalize()}: {msg['content']}" for msg in state["chat_history"]])
    prompt = f"Rewrite the latest question to be standalone based on chat history. Do not answer it.\n\nHistory:\n{history_str}\n\nQuestion: {state['original_question']}\nStandalone:"

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    response = execute_with_retry(llm.invoke, prompt)
    
    in_t, out_t = get_usage(response)
    print(f"\n[Agent: Rewrote query -> '{response.content.strip()}']")
    
    return {"standalone_question": response.content.strip(), "input_tokens": in_t, "output_tokens": out_t}

def retrieve_node(state: GraphState):
    docs = execute_with_retry(hybrid_retriever.invoke, state["standalone_question"])
    return {"documents": docs}

def generate_node(state: GraphState):
    # If this is the first time generating, build the context and system message
    if not state.get("llm_messages"):
        unique_src = []
        for d in state["documents"]:
            if d.metadata.get('source') not in unique_src: unique_src.append(d.metadata.get('source'))
                
        context = "\n\n".join([f"--- Source [{unique_src.index(d.metadata.get('source')) + 1}] ---\nContent:\n{d.page_content}" for d in state["documents"]])
        
        prompt = f"""You are an analytical educational assistant for Zerodha Varsity. 
Your goal is to answer the user's question accurately using ONLY the provided Context.
Rules:
1. If the answer is not in the Context, say exactly: "I cannot answer this based on the provided website."
2. Do not use outside knowledge. 
3. MATH & LOGIC: If a calculation is required, you MUST use the `calculate` tool to do it. Do not guess the math!
4. You MUST use inline citations referring to the Source ID (e.g., "[1]").

Context:\n{context}"""

        messages = [
            SystemMessage(content=prompt),
            HumanMessage(content=state['standalone_question'])
        ]
        sources = [f"[{i+1}] {url}" for i, url in enumerate(unique_src)]
    else:
        messages = state["llm_messages"]
        sources = state.get("sources", [])

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    llm_with_tools = llm.bind_tools([calculate])
    
    response = execute_with_retry(llm_with_tools.invoke, messages)
    messages.append(response)
    
    in_t, out_t = get_usage(response)
    
    # --- SAFE TEXT EXTRACTION FIX ---
    raw_content = response.content
    if isinstance(raw_content, list):
        # Extract text if Gemini returns a list of blocks (common with tool-enabled models)
        final_text = "".join([item.get("text", "") for item in raw_content if isinstance(item, dict)])
    else:
        final_text = str(raw_content)

    return {
        "llm_messages": messages,
        "answer": final_text, 
        "sources": sources,
        "input_tokens": state.get("input_tokens", 0) + in_t,
        "output_tokens": state.get("output_tokens", 0) + out_t
    }

def route_after_generate(state: GraphState):
    """Determines if the LLM called a tool or provided the final answer."""
    last_message = state["llm_messages"][-1]
    if last_message.tool_calls:
        return "execute_tools"
    return "grade"

def execute_tools_node(state: GraphState):
    """Executes the calculation and feeds it back to the LLM."""
    messages = state["llm_messages"]
    last_message = messages[-1]
    
    for tool_call in last_message.tool_calls:
        if tool_call["name"] == "calculate":
            result = calculate.invoke(tool_call["args"])
            messages.append(ToolMessage(content=result, tool_call_id=tool_call["id"]))
            
    return {"llm_messages": messages}

def grade_node(state: GraphState):
    if "I cannot answer this" in state["answer"]:
        return {"is_grounded": "yes"}
        
    class Grader(BaseModel):
        binary_score: str = Field(description="Score 'yes' if grounded, otherwise 'no'")

    llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", temperature=0, google_api_key=api_key)
    structured_llm = llm.with_structured_output(Grader)
    
    context = "\n\n".join([d.page_content for d in state["documents"]])
    prompt = f"Context: {context}\n\nAnswer: {state['answer']}\nDoes the answer contain outside knowledge? If mathematical/logical but derived from context, it IS grounded. Score 'no' if ungrounded outside knowledge. Score 'yes' if grounded."
    
    result = execute_with_retry(structured_llm.invoke, prompt)
    
    in_t = result.usage_metadata.get('input_tokens', 0) if hasattr(result, 'usage_metadata') and result.usage_metadata else 0
    out_t = result.usage_metadata.get('output_tokens', 0) if hasattr(result, 'usage_metadata') and result.usage_metadata else 0
    
    print(f" [Grader Node Checked Grounding: {result.binary_score.upper()}]")
    
    return {
        "is_grounded": result.binary_score.lower(),
        "input_tokens": state.get("input_tokens", 0) + in_t,
        "output_tokens": state.get("output_tokens", 0) + out_t
    }

def route_after_grade(state: GraphState):
    return END if state.get("is_grounded", "yes") == "yes" else "fallback"

def fallback_node(state: GraphState):
    return {"answer": "I cannot answer this based on the provided website. *(Intercepted by Grounding Grader)*", "sources": []}

# --- 6. Compile LangGraph ---
workflow = StateGraph(GraphState)
workflow.add_node("rewrite", rewrite_query_node)
workflow.add_node("retrieve", retrieve_node)
workflow.add_node("generate", generate_node)
workflow.add_node("execute_tools", execute_tools_node)
workflow.add_node("grade", grade_node)
workflow.add_node("fallback", fallback_node)

workflow.set_entry_point("rewrite")
workflow.add_edge("rewrite", "retrieve")
workflow.add_edge("retrieve", "generate")
workflow.add_conditional_edges("generate", route_after_generate, {
    "execute_tools": "execute_tools",
    "grade": "grade"
})
workflow.add_edge("execute_tools", "generate") # Loop back after calculating
workflow.add_conditional_edges("grade", route_after_grade, {
    "fallback": "fallback",
    END: END
})
workflow.add_edge("fallback", END)

app = workflow.compile()

# --- 7. Interactive Chat Loop ---
if __name__ == "__main__":
    print("\n✅ Agent Ready! Welcome to the Enhanced Varsity RAG Agent (with Math Tool Capabilities)")
    print("-----------------------------------------------------------------")
    
    chat_history = []
    
    while True:
        q = input("\nAsk a question: ")
        if q.lower() in ["quit", "exit"]: break
        
        initial_state = {
            "original_question": q,
            "chat_history": chat_history[-4:], 
            "llm_messages": [], 
            "input_tokens": 0,
            "output_tokens": 0
        }
        
        result = app.invoke(initial_state)
        
        print("\n--- Answer ---")
        print(result["answer"])
        if "I cannot answer this" not in result["answer"]:
            print("\n--- References ---")
            for s in result["sources"]: print(s)
            
        in_tok = result.get('input_tokens', 0)
        out_tok = result.get('output_tokens', 0)
        est_cost = ((in_tok / 1_000_000) * COST_PER_1M_INPUT) + ((out_tok / 1_000_000) * COST_PER_1M_OUTPUT)
        
        print("\n" + "-"*45)
        print(f"📊 QUERY COST RECEIPT:")
        print(f"   Input Tokens : {in_tok:,}")
        print(f"   Output Tokens: {out_tok:,}")
        print(f"   Est. Cost    : ${est_cost:.6f} USD (Free Tier: $0.00)")
        print("-"*45)
            
        chat_history.append({"role": "user", "content": q})
        chat_history.append({"role": "assistant", "content": result["answer"]})