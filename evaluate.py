import sys
import types
import os
import json
import time
import argparse
import warnings
from datasets import Dataset

# =====================================================================
# 1. UPSTREAM RAGAS PATCHES (MUST RUN BEFORE IMPORTS)
# =====================================================================
# Silence the messy deprecation warnings caused by the RAGAS 0.4.x transition bug
warnings.filterwarnings("ignore", category=DeprecationWarning)

try:
    import langchain_community.chat_models
    vertexai_dummy = types.ModuleType("langchain_community.chat_models.vertexai")
    vertexai_dummy.ChatVertexAI = type("ChatVertexAI", (object,), {})
    sys.modules["langchain_community.chat_models.vertexai"] = vertexai_dummy
    setattr(langchain_community.chat_models, "vertexai", vertexai_dummy)
except Exception:
    pass
# =====================================================================

from ragas import evaluate
# FIX: Use the legacy pre-instantiated metrics that evaluate() still requires
from ragas.metrics import faithfulness, answer_relevancy, context_precision, context_recall
from langchain_google_genai import ChatGoogleGenerativeAI, GoogleGenerativeAIEmbeddings

# Import Agent & Credentials
from agent import app, api_key


def invoke_agent_with_retry(payload, max_retries=3):
    """Protects against Google Free Tier rate limits during sequential calls."""
    for attempt in range(max_retries):
        try:
            return app.invoke(payload)
        except Exception as e:
            err = str(e)
            if "429" in err or "RESOURCE_EXHAUSTED" in err:
                print(f"\n⚠️ Rate limit hit! Sleeping for 60s (Attempt {attempt+1}/{max_retries})...")
                time.sleep(60)
            else:
                raise e
    raise Exception("Evaluation max retries exceeded.")


def load_dataset(filepath: str):
    """Loads and validates evaluation dataset from JSON."""
    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Dataset file not found at: {filepath}")
    
    with open(filepath, "r", encoding="utf-8") as f:
        data = json.load(f)
        
    print(f"✅ Loaded {len(data)} test items from '{filepath}'")
    return data


def run_evaluation(dataset_path: str):
    eval_items = load_dataset(dataset_path)

    base_name = os.path.splitext(os.path.basename(dataset_path))[0]
    output_csv = f"ragas_report_{base_name}.csv"

    questions = []
    answers = []
    contexts = []
    ground_truths = []
    categories = []

    print(f"\n🚀 Running Evaluation on '{base_name}'...\n")

    for idx, item in enumerate(eval_items):
        q = item["question"]
        cat = item.get("category", "General")
        history = item.get("chat_history", [])

        print(f"[{idx+1}/{len(eval_items)}] Testing [{cat}]: '{q}'")

        payload = {
            "original_question": q,
            "chat_history": history,
            "llm_messages": [],
            "input_tokens": 0,
            "output_tokens": 0
        }

        result = invoke_agent_with_retry(payload)

        questions.append(q)
        answers.append(result["answer"])
        ground_truths.append(item.get("ground_truth", ""))
        categories.append(cat)

        doc_texts = [doc.page_content for doc in result.get("documents", [])]
        contexts.append(doc_texts if doc_texts else ["No context retrieved."])

    # Convert to HuggingFace Dataset
    dataset = Dataset.from_dict({
        "question": questions,
        "answer": answers,
        "contexts": contexts,
        "ground_truth": ground_truths
    })

    print("\n📊 Computing RAGAS Metrics (Retrieval & Generation)...")
    
    # 1. Initialize standard LangChain models (No wrappers needed!)
    ragas_llm = ChatGoogleGenerativeAI(model="gemini-2.5-flash", google_api_key=api_key)
    ragas_embeddings = GoogleGenerativeAIEmbeddings(model="models/gemini-embedding-001", google_api_key=api_key)
    
    # 2. Pass them directly to evaluate() with the legacy metrics
    eval_result = evaluate(
        dataset=dataset,
        metrics=[
            faithfulness,
            answer_relevancy,
            context_precision,
            context_recall
        ],
        llm=ragas_llm,
        embeddings=ragas_embeddings
    )

    print("\n" + "="*50)
    print(f"  🏆 RAGAS REPORT: {base_name.upper()}")
    print("="*50)
    print(eval_result)
    print("="*50)

    # Save breakdown to CSV
    df = eval_result.to_pandas()
    df["category"] = categories
    df.to_csv(output_csv, index=False)
    print(f"\n📁 Detailed scorecard saved to '{output_csv}'")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate Varsity RAG Agent using JSON datasets.")
    parser.add_argument(
        "--dataset",
        type=str,
        default="eval_data/comprehensive_eval.json",
        help="Path to the JSON evaluation dataset file (default: eval_data/real_eval.json)"
    )
    args = parser.parse_args()
    run_evaluation(dataset_path=args.dataset)