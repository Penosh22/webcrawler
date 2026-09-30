import json
import os
import tiktoken
import time
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_google_genai import GoogleGenerativeAIEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()
api_key = os.getenv("GEMINI_API_KEY")
if not api_key:
    raise ValueError("GEMINI_API_KEY is missing from .env")

CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "models/gemini-embedding-001" 
EMBEDDING_COST_PER_MILLION_PAID_REF = 0.15

def clean_and_load_docs(json_path="varsity_docs.json"):
    with open(json_path, "r", encoding="utf-8") as f:
        raw_docs = json.load(f)

    headers_to_split_on = [
        ("#", "Header 1"),
        ("##", "Header 2"),
        ("###", "Header 3"),
    ]
    markdown_splitter = MarkdownHeaderTextSplitter(
        headers_to_split_on=headers_to_split_on,
        strip_headers=False
    )

    text_splitter = RecursiveCharacterTextSplitter(
        chunk_size=1200,
        chunk_overlap=150,
        separators=["\n\n", "\n", " ", ""]
    )

    final_chunks = []

    for item in raw_docs:
        url = item.get("url", "")
        title = item.get("title", "")
        content = item.get("content", "").strip()

        if not content:
            continue

        for cutoff_phrase in ["Key Takeaways", "Key takeaways from this chapter"]:
            if cutoff_phrase in content:
                split_idx = content.find(cutoff_phrase)
                content = content[:split_idx + 1500] 
                break

        header_splits = markdown_splitter.split_text(content)

        for split_doc in header_splits:
            sub_splits = text_splitter.split_text(split_doc.page_content)
            
            headers = [
                split_doc.metadata.get("Header 1"),
                split_doc.metadata.get("Header 2"),
                split_doc.metadata.get("Header 3")
            ]
            breadcrumb = " > ".join([h for h in headers if h])

            for sub_text in sub_splits:
                context_prefix = f"[{title} | {breadcrumb}]\n" if breadcrumb else f"[{title}]\n"
                enriched_content = context_prefix + sub_text

                final_chunks.append(
                    Document(
                        page_content=enriched_content,
                        metadata={
                            "source": url,
                            "title": title,
                            "section": breadcrumb or title,
                            **split_doc.metadata
                        }
                    )
                )

    return final_chunks

def track_ingestion_cost(documents):
    encoding = tiktoken.get_encoding("cl100k_base")
    total_tokens = sum(len(encoding.encode(doc.page_content)) for doc in documents)
    cost_paid_ref = (total_tokens / 1_000_000) * EMBEDDING_COST_PER_MILLION_PAID_REF
    return total_tokens, 0.0, cost_paid_ref


def main():
    print("Cleaning noise and structurally chunking documents...")
    docs = clean_and_load_docs()

    # --- 1. TOKEN REPORT ---
    total_tokens, free_cost, paid_ref_cost = track_ingestion_cost(docs)
    print("\n" + "="*45)
    print("       INGESTION TOKEN & COST REPORT")
    print("="*45)
    print(f"Total Chunks Created : {len(docs)}")
    print(f"Total Tokens Indexed : {total_tokens:,}")
    print(f"Actual Cost (Free)   : ${free_cost:.4f} USD")
    print("="*45 + "\n")

    embeddings = GoogleGenerativeAIEmbeddings(
        model=EMBEDDING_MODEL,
        google_api_key=api_key
    )

    print(f"Connecting to Vector Store at '{CHROMA_PERSIST_DIR}'...")
    vectorstore = Chroma(
        embedding_function=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

    # --- 2. RESUME LOGIC (Using text content instead of IDs) ---
    existing_data = vectorstore.get(include=["documents"])
    existing_texts = set(existing_data["documents"])
    
    remaining_docs = [doc for doc in docs if doc.page_content not in existing_texts]

    print(f"\nTotal chunks in dataset: {len(docs)}")
    print(f"Chunks already embedded: {len(existing_texts)}")
    print(f"Remaining chunks to embed: {len(remaining_docs)}")

    if len(remaining_docs) == 0:
        print("✅ Everything is already embedded!")
        return

    # --- 3. RATE LIMIT SAFEGUARD: BATCH INGESTION ---
    batch_size = 10 
    total_batches = (len(remaining_docs) // batch_size) + 1
    
    print(f"Resuming rate-limited ingestion ({total_batches} batches left)...")
    
    for i in range(0, len(remaining_docs), batch_size):
        batch = remaining_docs[i:i + batch_size]
        current_batch = (i // batch_size) + 1
        print(f"Embedding batch {current_batch} of {total_batches}...")
        
        try:
            vectorstore.add_documents(batch)
        except Exception as e:
            print(f"⚠️ Rate limit hit! Pausing for 60 seconds before retrying...")
            time.sleep(60)
            vectorstore.add_documents(batch) 
            
        time.sleep(6) 

    print("\n✅ Ingestion complete! Structurally chunked vector store saved.")

if __name__ == "__main__":
    main()