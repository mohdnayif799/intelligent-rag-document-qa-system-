import os
import glob
import shutil
import time
import gc
import streamlit as st
from langchain_huggingface import HuggingFaceEmbeddings
from langchain_community.vectorstores import Chroma

CHROMA_BASE_DIR = os.path.join(
    os.path.dirname(os.path.abspath(__file__)), "..", "chroma_db"
)
EMBEDDING_MODEL = "all-MiniLM-L6-v2"


@st.cache_resource
def get_embeddings() -> HuggingFaceEmbeddings:
    """Load the embedding model once and cache it for the session."""
    return HuggingFaceEmbeddings(
        model_name=EMBEDDING_MODEL,
        model_kwargs={"device": "cpu"},
        encode_kwargs={"normalize_embeddings": True, "batch_size": 64},
    )


# ── Used by evaluate.py ───────────────────────────────────────────────────────

def build_vector_store(chunks: list) -> tuple:
    """Create a new store + clean up old ones. Used only by evaluate.py."""
    chroma_dir = f"{CHROMA_BASE_DIR}_{int(time.time())}"
    embeddings = get_embeddings()
    store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=chroma_dir,
    )
    print(f"[INFO] Vector store built at '{chroma_dir}'.")
    _cleanup_old_chroma_dirs(keep=chroma_dir)
    return store, chroma_dir


def _cleanup_old_chroma_dirs(keep: str):
    pattern = os.path.join(os.path.dirname(CHROMA_BASE_DIR), "chroma_db*")
    for dir_path in glob.glob(pattern):
        if dir_path != keep and os.path.isdir(dir_path):
            try:
                shutil.rmtree(dir_path)
                print(f"[INFO] Deleted old store: '{dir_path}'.")
            except Exception:
                pass


# ── Used by app.py (per-chat, no cross-chat cleanup) ─────────────────────────

def create_chat_vector_store(chunks: list) -> tuple:
    """
    Create a new ChromaDB store for a chat's first document upload.
    Does NOT delete other chats' stores.
    """
    chroma_dir = f"{CHROMA_BASE_DIR}_{int(time.time())}"
    embeddings = get_embeddings()
    store = Chroma.from_documents(
        documents=chunks, embedding=embeddings, persist_directory=chroma_dir,
    )
    count = store._collection.count()
    print(f"[INFO] Chat store created at '{chroma_dir}' with {count} chunks.")
    return store, chroma_dir


def append_to_vector_store(chunks: list, chroma_dir: str):
    """
    Append new document chunks to an existing ChromaDB store.

    Root cause of multi-doc bug: having TWO Chroma client objects pointing
    at the same SQLite file simultaneously causes locking and data loss on
    Windows. Fix: clear the Streamlit-cached connection BEFORE opening the
    append connection, collect garbage so the old handle is released, then
    add documents, then delete the temporary connection.

    The next call to load_vector_store() creates a fresh client that reads
    all data — both the original and the newly appended chunks.
    """
    # Step 1: Release existing cached connection to avoid SQLite lock conflict
    load_vector_store.clear()
    gc.collect()
    time.sleep(0.4)   # give Windows time to release file handles

    # Step 2: Open a fresh connection and append
    embeddings = get_embeddings()
    store = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)

    before = store._collection.count()
    store.add_documents(chunks)
    after = store._collection.count()

    print(f"[INFO] Appended {len(chunks)} chunks. "
          f"Store: {before} → {after} total chunks at '{chroma_dir}'.")

    # Step 3: Explicitly release the append connection
    del store
    gc.collect()
    # Cache is already cleared from Step 1; next load_vector_store call
    # will open a new connection that sees all {after} chunks.


def get_store_stats(chroma_dir: str) -> dict:
    """
    Return diagnostic info about a vector store.
    Used by the status indicator to show how many chunks are indexed.
    """
    try:
        embeddings = get_embeddings()
        store = Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
        total = store._collection.count()
        # Sample up to 500 chunks to count per-document breakdown
        sample = store._collection.get(limit=500, include=["metadatas"])
        from collections import Counter
        counts = Counter(
            os.path.basename(m.get("source", "unknown"))
            for m in (sample["metadatas"] or [])
        )
        del store
        return {"total_chunks": total, "per_doc": dict(counts)}
    except Exception as e:
        return {"total_chunks": 0, "per_doc": {}, "error": str(e)}


@st.cache_resource
def load_vector_store(chroma_dir: str) -> Chroma:
    """Load a ChromaDB store from disk and cache the connection per directory."""
    embeddings = get_embeddings()
    return Chroma(persist_directory=chroma_dir, embedding_function=embeddings)
