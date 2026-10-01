# 📈 Zerodha Varsity AI Assistant

An agentic, multi-turn Retrieval-Augmented Generation (RAG) assistant built to provide grounded, citation-backed answers to stock trading, investing, derivatives, and financial regulatory questions using educational content from [Zerodha Varsity](https://zerodha.com/varsity/).

Powered by **Crawl4AI**,**LangGraph**, **Google Gemini 2.5 Flash**, **ChromaDB**, **BM25**, and **FlashRank**, the assistant combines hybrid retrieval, deterministic math evaluation, self-reflective grounding validation, and real-time cost telemetry.

---

## 🏛️️ System Architecture

The project is split into two phases: an **offline data ingestion pipeline** and a **live agentic graph** orchestrated with LangGraph.

### Architecture Diagram
```text
[OFFLINE PIPELINE]
 ┌────────────┐     ┌────────────┐     ┌────────────────────────────────────┐
 │ crawler.py │ ──► │ ingest.py  │ ──► │  chroma_db/ (Dense Vector Store)   │
 └────────────┘     └────────────┘     │  bm25_index.pkl (Keyword Search)   │
                                       └────────────────────────────────────┘
                                                         │
[LIVE AGENT PIPELINE]                                    │
 [User Input + Chat History]                             │
             │                                           ▼
             ▼                                   ┌───────────────┐
     ┌───────────────┐                           │   retrieve    │ 
     │ rewrite_query │ ──► Reformulates ─────────┤ (Hybrid Search│
     └───────────────┘     standalone prompt     │ + FlashRank)  │
             │                                   └───────────────┘
             ▼                                           │
     ┌───────────────┐                                   │
 ┌──►│   generate    │ ◄── LLM generates answer w/ citations [1]
 │   └───────────────┘
 │           │
 │     [Tool Calls?]
 │      ├─ Yes ──► ┌───────────────┐
 │      │          │ execute_tools │ ──► Evaluates expressions safely via `numexpr`
 └─── Loop Back ───└───────────────┘
        │
        └─ No
             │
             ▼
     ┌───────────────┐
     │  grade_node   │ ──► Evaluates hallucination / factual grounding
     └───────────────┘
             │
      [Is Grounded?]
        ├─ Yes ──► (End / Render Answer to UI)
        └─ No  ──► ┌───────────────┐
                   │   fallback    │ ──► "I cannot answer this based on the provided website."
                   └───────────────┘

```
<img width="1536" height="1024" alt="image" src="https://github.com/user-attachments/assets/9ed4edc6-b037-4f86-83b5-c98c23be31da" />

---

## ⚙️ Key Technical Decisions

1. **Hybrid Retrieval + Cross-Encoder Reranking:**
* **Dense Vectors (ChromaDB):** Captures high-level semantic intent using `gemini-embedding-001`.
* **Keyword Search (BM25):** Financial terminology is full of specific acronyms (e.g., *STT*, *SEBI*, *ATM/OTM*). BM25 guarantees exact keyword hits where dense embeddings fail.
* **FlashRank Reranking:** Rather than flooding context with raw top-$k$ results, retrieved documents pass through an ultra-lightweight, local cross-encoder, compressing candidates down to the top 7–12 most relevant chunks.


2. **Deterministic Financial Math (`numexpr` Tool):** Large Language Models frequently make arithmetic errors on multi-step financial calculations. When arithmetic is detected, the agent defers to a sandboxed `calculate` tool to compute deterministic numerical answers.
3. **Self-Reflective Grounding Grader:** Every response passes through an automated structured output check. If outside knowledge or ungrounded claims are detected, the response is intercepted and replaced with a strict fallback refusal.
4. **Zero-Leak "Bring Your Own Key" (BYOK):** To avoid committing secrets or mandating `.env` files on shared deployments (like Streamlit Cloud), the pipeline uses lazy initialization. Vector stores and LLM clients are built on-demand using the live API key supplied directly through the Streamlit sidebar.

---

## 💰 Cost Analysis

The system uses **Google Gemini 2.5 Flash**, which offers exceptionally low API pricing:

* **Input Tokens:** ~$0.30 per 1 Million tokens
* **Output Tokens:** ~$2.50 per 1 Million tokens [1]

### Example Query Breakdown

**Query:** *"If I buy 250 shares at 120 and sell at 145, what is my profit? Also explain what an iron condor is."*

* **Estimated Input Tokens:** ~1,800 tokens *(Includes chat history + system prompt + 7 to 12 retrieved document chunks)*
* **Estimated Output Tokens:** ~250 tokens *(The generated explanation and math calculation)*

**Cost per average query:**

* Input Cost: `1,800 * ($0.30 / 1,000,000) = $0.00054`
* Output Cost: `250 * ($2.50 / 1,000,000) = $0.000625`
* **Total Estimated Cost: ~$0.0011 USD per query**

### Scaling Projections

| Traffic Volume | Estimated Cost (USD) | Notes |
| --- | --- | --- |
| **100 Queries** | **~$0.11** | Negligible cost for personal daily use or testing. |
| **1,000 Queries** | **~$1.10** | Great for small classroom environments or portfolio showcases. |
| **10,000 Queries** | **~$11.00** | Production-scale testing; highly cost-effective compared to GPT-4o. |

*(Note: The Google AI Studio Free Tier provides up to 15 Requests Per Minute at $0.00 cost, making personal usage completely free).*

---

## 📂 Project Structure

```text
├── crawler.py             # Scrapes raw content/URLs from the Zerodha Varsity website
├── ingest.py              # Chunks data and builds the local Chroma & BM25 indexes
├── agent.py               # Compiled LangGraph pipeline & node implementations
├── streamlit_app.py       # Streamlit web application & session telemetry
├── evaluate.py            # RAGAS evaluation benchmark suite
├── varsity_docs.json      # Structured documents output by crawler.py
├── requirements.txt       # Production dependencies
├── .gitignore             # Excludes binary stores, envs, and cache artifacts
├── chroma_db/             # Local vector database directory (generated by ingest.py)
└── bm25_index.pkl         # Pickled BM25 index cache (generated by ingest.py)

```

---

## 🚀 Setup & Installation

### Prerequisites

* Python 3.10 or 3.11
* A Google Gemini API Key ([Get one at Google AI Studio](https://aistudio.google.com/))

### 1. Clone the Repository

```bash
git clone [https://github.com/Penosh22/webcrawler.git](https://github.com/Penosh22/webcrawler.git)
cd webcrawler

```

### 2. Create and Activate a Virtual Environment

```bash
# macOS/Linux
python3 -m venv venv
source venv/bin/activate

# Windows (Command Prompt)
python -m venv venv
venv\Scripts\activate

```

### 3. Install Dependencies

```bash
pip install --upgrade pip
pip install -r requirements.txt

```

### 4. (Optional) Configure Environment Variables

If running locally via the CLI, you can create a `.env` file in the root directory:

```env
GEMINI_API_KEY=your_gemini_api_key_here

```

*(Note: If running the Streamlit UI, you can skip this step entirely and paste your key directly into the app sidebar).*

---

## 🏗️ Data Ingestion Pipeline (Run Once)

Before chatting with the agent, you must build the vector database and search indexes.

1. **(Optional) Crawl the Website:** If `varsity_docs.json` is not present, run the crawler to fetch fresh documentation.
```bash
python crawler.py

```


2. **Build the Search Indexes:** Run the ingestion script. This will chunk the JSON data, generate dense embeddings, and save `chroma_db` and `bm25_index.pkl` to your local folder (takes ~30-60 seconds depending on API limits).
```bash
python ingest.py

```



---

## 🖥️ How to Run the Live Agent

### Option A: Interactive Web UI (Streamlit)

The recommended way to interact with the assistant. Features live cost tracking and citation expanders.

```bash
streamlit run streamlit_app.py

```

1. Open your browser at `http://localhost:8501`.
2. Enter your Gemini API key in the sidebar under **Configuration**.
3. Ask questions (e.g., *"What is an iron condor?"* followed by *"When should I enter one?"*).

### Option B: Terminal Command-Line Interface (CLI)

Starts an interactive shell loop directly in your console. (Requires `.env` file setup).

```bash
python agent.py

```

Type `exit` or `quit` to terminate the session.

---

## 📊 Evaluation (RAGAS)

Evaluation metrics are computed using the **RAGAS** (Retrieval Augmented Generation Assessment) framework against curated ground-truth QA pairs. The script tests for:

* **Faithfulness:** Factual consistency of the answer against retrieved context.
* **Answer Relevance:** How directly the answer addresses the prompt.
* **Context Precision:** Whether the most relevant chunks are ranked at the top.
* **Context Recall:** Verifies all necessary information was successfully retrieved.

To run the evaluation suite:

```bash
python evaluate.py

```

Results will print to stdout and export a timestamped benchmark summary to CSV.

---

## ⚠ Known Limitations

1. **Static Knowledge Base:** The vector store and BM25 index reflect static snapshots of Zerodha Varsity content. Real-time market changes, live stock quotes, or newly revised tax slabs are not reflected unless you re-run `crawler.py` and `ingest.py`.
2. **Free-Tier Rate Limits:** The Google AI Studio free tier enforces a 15 RPM (Requests Per Minute) cap. The codebase handles `429 RESOURCE_EXHAUSTED` errors with exponential backoff, but concurrent high-throughput requests or large ingestion workloads may experience delays.
3. **Pickle Index Portability:** `bm25_index.pkl` is serialized using Python's standard `pickle`. If switching major Python versions (e.g., building the cache on Python 3.12 and loading it on Python 3.10), simply delete `bm25_index.pkl` and re-run `python ingest.py`.

```

```
