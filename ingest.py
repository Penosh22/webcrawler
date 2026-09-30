import json
import os
import tiktoken
from dotenv import load_dotenv
from langchain_core.documents import Document
from langchain_text_splitters import MarkdownHeaderTextSplitter, RecursiveCharacterTextSplitter
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

# Load environment variables
load_dotenv()

CHROMA_PERSIST_DIR = "./chroma_db"
EMBEDDING_MODEL = "all-MiniLM-L6-v2" # Extremely fast, lightweight local model

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

        # --- COMMENT FILTERING ---
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
    return total_tokens, 0.0, 0.0 # Truly $0.00 since it runs locally!

def main():
    print("Cleaning noise and structurally chunking documents...")
    docs = clean_and_load_docs()

    total_tokens, free_cost, paid_ref_cost = track_ingestion_cost(docs)
    print("\n" + "="*45)
    print("       INGESTION TOKEN & COST REPORT")
    print("="*45)
    print(f"Embedding Model      : {EMBEDDING_MODEL} (Local)")
    print(f"Total Chunks Created : {len(docs)}")
    print(f"Total Tokens Indexed : {total_tokens:,}")
    print(f"Actual Cost          : $0.00 USD (Unlimited)")
    print("="*45 + "\n")

    print(f"Downloading/Loading Local Model '{EMBEDDING_MODEL}'...")
    embeddings = HuggingFaceEmbeddings(model_name=EMBEDDING_MODEL)

    print(f"Initializing Vector Store at '{CHROMA_PERSIST_DIR}' and embedding all chunks...")
    # Because it's local, we can throw all documents at it at once!
    vectorstore = Chroma.from_documents(
        documents=docs,
        embedding=embeddings,
        persist_directory=CHROMA_PERSIST_DIR
    )

    print("\n✅ Ingestion complete! Structurally chunked vector store saved.")

if __name__ == "__main__":
    main()