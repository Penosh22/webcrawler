import os
import streamlit as st
import agent  # Import the base module to allow overriding its global variables
from agent import app as rag_agent, COST_PER_1M_INPUT, COST_PER_1M_OUTPUT

# --- Page Configuration ---
st.set_page_config(
    page_title="Zerodha Varsity AI Assistant",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded",
)

# --- Session State Initialization ---
if "messages" not in st.session_state:
    st.session_state.messages = []
if "total_input_tokens" not in st.session_state:
    st.session_state.total_input_tokens = 0
if "total_output_tokens" not in st.session_state:
    st.session_state.total_output_tokens = 0
if "total_cost" not in st.session_state:
    st.session_state.total_cost = 0.0
if "sample_prompt" not in st.session_state:
    st.session_state.sample_prompt = None

# --- Sidebar Metrics & Controls ---
with st.sidebar:
    st.title("🔑 Configuration")
    # API Key Input
    user_api_key = st.text_input("Gemini API Key", type="password", placeholder="Enter key to override .env")
    if user_api_key:
        # Dynamically inject the key into the running agent module
        agent.api_key = user_api_key
        os.environ["GEMINI_API_KEY"] = user_api_key

    st.markdown("---")
    st.title("📊 Session Analytics")
    st.caption("Live usage & token consumption tracker")

    # 1. Create an empty container placeholder for the metrics
    metrics_placeholder = st.empty()

    st.markdown("---")
    st.subheader("💡 Sample Questions")
    
    sample_questions = [
        "Why do companies go public?",
        "If I buy 250 shares at 120 and sell at 145, what is my profit?",
        "What happens after the IPO?",
        "What is the current stock price of Reliance?" # Tests the grounding refusal
    ]
    
    for q in sample_questions:
        if st.button(q, use_container_width=True):
            st.session_state.sample_prompt = q

# 2. Define a function to draw/redraw metrics inside that placeholder
def render_metrics():
    with metrics_placeholder.container():
        col1, col2 = st.columns(2)
        with col1:
            st.metric("Total Prompts", len([m for m in st.session_state.messages if m["role"] == "user"]))
        with col2:
            st.metric("Total Cost", f"${st.session_state.total_cost:.5f}")

        st.markdown("---")
        st.write(f"**Input Tokens:** `{st.session_state.total_input_tokens:,}`")
        st.write(f"**Output Tokens:** `{st.session_state.total_output_tokens:,}`")
        total_tokens = st.session_state.total_input_tokens + st.session_state.total_output_tokens
        st.write(f"**Cumulative Tokens:** `{total_tokens:,}`")

# Draw the initial state of the metrics
render_metrics()

with st.sidebar:
    st.markdown("---")
    st.subheader("⚙️ System Architecture")
    st.markdown(
        """
        - **Model:** Gemini 2.5 Flash
        - **Retrieval:** BM25 + Chroma (Hybrid)
        - **Reranker:** FlashRank
        - **Tooling:** `numexpr` Financial Calculator
        - **Guardrails:** Self-reflection Grounding Grader
        """
    )
    st.markdown("---")
    if st.button("🗑️ Clear Conversation", use_container_width=True):
        st.session_state.messages = []
        st.session_state.total_input_tokens = 0
        st.session_state.total_output_tokens = 0
        st.session_state.total_cost = 0.0
        st.rerun()

# --- Main Chat UI ---
st.title("📈 Zerodha Varsity Financial RAG Agent")
st.markdown("Ask anything about stock trading, derivatives, regulations, or financial math.")

# Render Conversation History
for msg in st.session_state.messages:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])
        if msg.get("sources"):
            with st.expander("📚 Citations & References", expanded=False):
                for src in msg["sources"]:
                    st.markdown(f"- {src}")
        if "meta" in msg:
            meta = msg["meta"]
            st.caption(f"⏱️ Tokens: {meta['in_tok']} in / {meta['out_tok']} out | Est. Query Cost: ${meta['cost']:.6f}")

# Handle New User Prompt
prompt = st.chat_input("E.g., If I buy 250 shares at 120 and sell at 145, what is my profit?") or st.session_state.sample_prompt
if prompt:
    st.session_state.sample_prompt = None
    
    # Gatekeeper: Check if an API key exists either from .env or UI
    if not getattr(agent, "api_key", None):
        st.error("Please provide a Gemini API Key in the sidebar configuration.")
        st.stop()

    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
        st.markdown(prompt)

    chat_history = [{"role": m["role"], "content": m["content"]} for m in st.session_state.messages[:-1]][-4:]

    initial_state = {
        "original_question": prompt,
        "chat_history": chat_history,
        "llm_messages": [],
        "input_tokens": 0,
        "output_tokens": 0,
    }

    with st.chat_message("assistant"):
        with st.spinner("Analyzing context & verifying grounding..."):
            try:
                result = rag_agent.invoke(initial_state)

                answer = result.get("answer", "No answer generated.")
                sources = result.get("sources", [])
                in_tok = result.get("input_tokens", 0)
                out_tok = result.get("output_tokens", 0)

                query_cost = ((in_tok / 1_000_000) * COST_PER_1M_INPUT) + ((out_tok / 1_000_000) * COST_PER_1M_OUTPUT)

                # Update cumulative statistics
                st.session_state.total_input_tokens += in_tok
                st.session_state.total_output_tokens += out_tok
                st.session_state.total_cost += query_cost

                # Render response
                st.markdown(answer)

                valid_sources = []
                if "I cannot answer this" not in answer and sources:
                    valid_sources = sources
                    with st.expander("📚 Citations & References", expanded=False):
                        for s in valid_sources:
                            st.markdown(f"- {s}")

                st.caption(f"⏱️ Tokens: {in_tok} in / {out_tok} out | Est. Query Cost: ${query_cost:.6f}")

                st.session_state.messages.append({
                    "role": "assistant",
                    "content": answer,
                    "sources": valid_sources,
                    "meta": {"in_tok": in_tok, "out_tok": out_tok, "cost": query_cost},
                })
                
                # 3. Instantly redraw the sidebar metrics with the new totals
                render_metrics()

            except Exception as e:
                error_msg = f"⚠️ An error occurred while processing: {str(e)}"
                st.error(error_msg)
                st.session_state.messages.append({"role": "assistant", "content": error_msg})