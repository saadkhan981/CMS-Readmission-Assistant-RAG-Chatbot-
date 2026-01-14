

import os
from pathlib import Path

from langchain_community.document_loaders import PyPDFLoader, DirectoryLoader
from langchain_text_splitters import RecursiveCharacterTextSplitter
from langchain_chroma import Chroma
from langchain_openai import OpenAIEmbeddings
from dotenv import load_dotenv

# Load environment variables
load_dotenv(override=True)

# 🔑 IMPORTANT PATH FIX
# Moves from /implementation → /week5
BASE_DIR = Path(__file__).resolve().parent.parent

DB_NAME = str(BASE_DIR / "vector_db_readmissions")
KNOWLEDGE_BASE = BASE_DIR / "knowledge-base" / "contracts"

# Embeddings
embeddings = OpenAIEmbeddings(model="text-embedding-3-large")


def fetch_documents():
    """
    Load CMS Readmission PDF document from knowledge-base/contracts
    """
    # Option 1: Load single specific PDF
    pdf_path = KNOWLEDGE_BASE / "Version1.0_Hospital-Wide_Readmission_Measure_Methodology_Report_7.25.12.pdf"
    
    if not pdf_path.exists():
        raise FileNotFoundError(f"PDF not found at: {pdf_path}")
    
    loader = PyPDFLoader(str(pdf_path))
    docs = loader.load()
    
    # Add metadata to all pages
    for doc in docs:
        doc.metadata["doc_type"] = "cms_readmission"
        doc.metadata["source"] = "CMS Hospital-Wide Readmission Measure"
        doc.metadata["file_name"] = "Version1.0_Hospital-Wide_Readmission_Measure_Methodology_Report_7.25.12.pdf"
    
    return docs


def create_chunks(documents):
    """
    Chunk CMS methodology text while preserving section-level context.
    
    For PDFs, we want slightly larger chunks since PDF parsing can be messy.
    """
    splitter = RecursiveCharacterTextSplitter(
        chunk_size=1000,  # Increased for PDF content
        chunk_overlap=200,  # More overlap to preserve context
        separators=["\n\n", "\n", ". ", " ", ""],  # PDF-friendly separators
        length_function=len,
    )
    
    return splitter.split_documents(documents)


def create_embeddings(chunks):
    """
    Create and persist vector embeddings.
    """
    if os.path.exists(DB_NAME):
        print("🗑️  Deleting existing vector database...")
        Chroma(
            persist_directory=DB_NAME,
            embedding_function=embeddings,
        ).delete_collection()
    
    print("📦 Creating embeddings...")
    vectorstore = Chroma.from_documents(
        documents=chunks,
        embedding=embeddings,
        persist_directory=DB_NAME,
    )
    
    collection = vectorstore._collection
    count = collection.count()
    sample_embedding = collection.get(limit=1, include=["embeddings"])["embeddings"][0]
    
    print(f"✅ Created {count} chunks with {len(sample_embedding)}-dimension embeddings")
    
    return vectorstore


if __name__ == "__main__":
    print("📄 Loading CMS Readmission PDF...")
    docs = fetch_documents()
    print(f"✅ Loaded {len(docs)} pages from PDF")
    
    print("\n✂️  Creating chunks...")
    chunks = create_chunks(docs)
    print(f"✅ Created {len(chunks)} chunks")
    
    print("\n🔮 Creating embeddings...")
    create_embeddings(chunks)
    
    print("\n✅ CMS Readmission PDF ingestion complete!")
    print(f"📊 Vector database saved to: {DB_NAME}")