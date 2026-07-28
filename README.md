<div align="center">

# Intelligent Document QA System

**A production-grade Retrieval-Augmented Generation (RAG) application for conversational document question answering**

[!\[Python](https://img.shields.io/badge/Python-3.10.11-3776AB?style=flat-square&logo=python&logoColor=white)](https://python.org)
[!\[Streamlit](https://img.shields.io/badge/Streamlit-1.40+-FF4B4B?style=flat-square&logo=streamlit&logoColor=white)](https://streamlit.io)
[!\[LangChain](https://img.shields.io/badge/LangChain-1.3.9-1C3C3C?style=flat-square)](https://langchain.com)
[!\[ChromaDB](https://img.shields.io/badge/ChromaDB-0.5+-F97316?style=flat-square)](https://trychroma.com)
[!\[Gemini](https://img.shields.io/badge/Gemini-2.5_Flash-4285F4?style=flat-square&logo=google&logoColor=white)](https://ai.google.dev)
[!\[License](https://img.shields.io/badge/License-MIT-22C55E?style=flat-square)](LICENSE)

</div>

\---

## Overview

Most RAG implementations treat document Q\&A as a single-session, single-document problem. This project takes a different approach — it is built around the engineering challenges that arise when users work with **multiple documents across multiple conversations simultaneously**.

The system allows users to upload PDFs, Word documents, PowerPoint presentations, Markdown files, and plain text. It builds a semantic embedding index, retrieves contextually relevant chunks using MMR retrieval, rewrites conversational follow-up questions into standalone queries before retrieval, and returns grounded answers with exact source citations showing page numbers and slide numbers.

Each conversation is an isolated workspace. Switching between chats restores the previous conversation, its uploaded documents, and its entire retrieval context — without reprocessing anything.

\---

## Screenshots

### Conversational Q\&A Across Multiple Documents

"C:\\Users\\Muhammed Nayifuddin\\rag\_qa\_system\\Images\\Screenshot 2026-07-27 223927.png"

*Querying across three simultaneously indexed documents — a DOCX question bank, a PPTX presentation, and a plain text file — in a single chat session.*

\---

### Conversational Context Resolution

"C:\\Users\\Muhammed Nayifuddin\\rag\_qa\_system\\Images\\Screenshot 2026-07-27 224001.png"

*"Tell me about those planets" correctly resolves the pronoun reference from the previous exchange without any explicit document mention. The query rewriter expands "those" to "the eight planets in our solar system" before hitting the retriever.*

\---

### Multi-Chat Isolation

"C:\\Users\\Muhammed Nayifuddin\\rag\_qa\_system\\Images\\Screenshot 2026-07-27 224229.png"

*Two independent chats with separate document contexts. The active chat is highlighted with an indigo accent. Switching chats restores the full conversation and its associated vector store without re-uploading or reprocessing.*

\---

### Source Citations

"C:\\Users\\Muhammed Nayifuddin\\rag\_qa\_system\\Images\\Screenshot 2026-07-27 224019.png"

*Every answer includes inline source citations with exact locations — page numbers for PDFs, slide numbers for PPTX, character offsets for plain text. An expandable panel shows the retrieved chunks verbatim.*

\---

## Features

|Feature|Description|
|-|-|
|**Multi-chat workspace**|Independent conversations each with their own documents, history, and vector store|
|**Multi-document retrieval**|Upload multiple files per chat; all remain simultaneously searchable|
|**Incremental indexing**|Uploading a second document appends to the existing store — no rebuild, no data loss|
|**Conversational memory**|Previous exchanges are passed to the rewriter and the answer generator|
|**MMR retrieval**|Max Marginal Relevance ensures diverse chunks across documents rather than clustering on one source|
|**Query rewriting**|Ambiguous follow-ups ("next?", "tell me about them") are expanded into standalone queries before retrieval|
|**Exact citations**|Page numbers for PDF, slide numbers for PPTX, character offsets for TXT|
|**Session response cache**|Repeated questions are answered instantly from in-memory cache without an LLM call|
|**LLM-generated titles**|First question in each chat generates a concise 3–5 word title automatically|
|**OCR fallback**|Scanned PDFs are processed via EasyOCR on a per-page basis; typed pages use fast text extraction|
|**Per-chat doc isolation**|File uploader resets when switching chats; documents never leak between sessions|
|**Status indicator**|Sidebar displays readiness, chunk count, document count, and active model|

\---

## Architecture

"C:\\Users\\Muhammed Nayifuddin\\rag\_qa\_system\\Images\\Architecture.png"
flowchart TD
    User(\\\\\\\["👤 User"])

    subgraph UI\\\\\\\["Streamlit Interface"]
        direction TB
        Sidebar\\\\\\\["Sidebar\\\\\\\\nFile Upload · Model Select · Chat List"]
        ChatArea\\\\\\\["Chat Area\\\\\\\\nMessage History · Input · Citations"]
    end

    subgraph ChatMgr\\\\\\\["Multi-Chat Manager (st.session\\\\\\\_state)"]
        direction LR
        C1\\\\\\\["Chat 1\\\\\\\\nDocs · History · Cache · VectorStore"]
        C2\\\\\\\["Chat 2\\\\\\\\nDocs · History · Cache · VectorStore"]
        Cn\\\\\\\["Chat N\\\\\\\\n..."]
    end

    subgraph Ingestion\\\\\\\["Document Ingestion Pipeline"]
        direction TB
        Loader\\\\\\\["Format Router\\\\\\\\nPDF · DOCX · PPTX · TXT · MD"]
        OCR\\\\\\\["OCR Fallback\\\\\\\\nEasyOCR + PyMuPDF\\\\\\\\n(scanned pages only)"]
        Splitter\\\\\\\["RecursiveCharacterTextSplitter\\\\\\\\nchunk\\\\\\\_size=1000, overlap=200"]
    end

    subgraph VectorLayer\\\\\\\["Vector Layer (per chat)"]
        Embedder\\\\\\\["HuggingFace Embeddings\\\\\\\\nall-MiniLM-L6-v2 · CPU"]
        Chroma\\\\\\\["ChromaDB\\\\\\\\nPersisted per chat"]
        MMR\\\\\\\["MMR Retriever\\\\\\\\nfetch\\\\\\\_k=20 · k=5 · λ=0.6"]
    end

    subgraph LLMLayer\\\\\\\["LLM Layer (Google Gemini 2.5 Flash)"]
        Rewriter\\\\\\\["Query Rewriter\\\\\\\\nResolves pronouns \\\\\\\& references"]
        Generator\\\\\\\["Answer Generator\\\\\\\\nContext + History → Grounded Answer"]
    end

    Citations\\\\\\\["📄 Inline Source Citations\\\\\\\\nPage · Slide · Chunk"]

    User --> UI
    Sidebar -->|"Upload"| Ingestion
    Loader --> OCR --> Splitter
    Splitter --> Embedder --> Chroma
    Chroma --> MMR
    UI --> ChatMgr
    ChatMgr --> MMR
    ChatMgr -->|"lc\\\\\\\_history"| Rewriter
    MMR --> Rewriter
    Rewriter -->|"standalone query"| MMR
    MMR -->|"top-5 chunks"| Generator
    ChatMgr -->|"conversation history"| Generator
    Generator --> ChatArea
    Generator --> Citations
```

\---

## Workflow

```## 🔄 Workflow



The following sequence illustrates how a user query travels through the Retrieval-Augmented Generation (RAG) pipeline, from document upload to grounded answer generation.

User

&#x20;↓

Upload Documents

&#x20;↓

Chunking

&#x20;↓

Embeddings

&#x20;↓

ChromaDB

&#x20;↓

Question

&#x20;↓

Query Rewriter

&#x20;↓

MMR Retrieval

&#x20;↓

Gemini

&#x20;↓

Answer + Citations






    actor User
    participant UI as Streamlit UI
    participant Cache as Session Cache
    participant Rewriter as Query Rewriter<br/>(Gemini)
    participant Retriever as MMR Retriever<br/>(ChromaDB)
    participant LLM as Answer Generator<br/>(Gemini)

    User->>UI: Upload document(s)
    UI->>UI: Detect format, extract text
    UI->>UI: Chunk with RecursiveCharacterTextSplitter
    UI->>Retriever: Embed \\\\\\\& index chunks (append or create)
    UI-->>User: ✅ N chunks indexed

    User->>UI: Ask question
    UI->>Cache: Check response cache
    alt Cache hit
        Cache-->>UI: Cached answer (instant)
    else Cache miss
        UI->>Rewriter: Question + conversation history
        Rewriter-->>UI: Standalone, reference-resolved query
        UI->>Retriever: MMR search (fetch\\\\\\\_k=20, k=5)
        Retriever-->>UI: Top-5 diverse chunks + metadata
        UI->>LLM: Chunks + history + original question
        LLM-->>UI: Grounded answer
        UI->>Cache: Store answer
    end
    UI-->>User: Answer + inline citations
```

\---

## Engineering Challenges

|Challenge|Root Cause|Solution|
|-|-|-|
|**Documents disappearing after second upload**|Two concurrent SQLite clients on the same ChromaDB file caused write conflicts on Windows|Clear `@st.cache\\\\\\\_resource` connection and run `gc.collect()` before opening append client; sleep 400ms for handle release|
|**All retrieved chunks from one document**|Standard cosine similarity returns top-k nearest neighbors, which cluster on one source|MMR retrieval: fetch 20 candidates, select 5 maximally relevant AND diverse|
|**"tell me about them" returns no results**|Retriever received the raw follow-up without knowing what "them" referred to|Two-pass pipeline: query rewriter resolves references first, then retriever searches the expanded query|
|**LLM says "I don't have information" despite correct retrieval**|QA prompt received original ambiguous question with no conversation context|Conversation history passed to both the rewriter and the answer generator|
|**All TXT and DOCX pages show as page 0**|TextLoader and Docx2txtLoader have no page structure; placeholder `page=0` was set|Display logic branches on file extension: PDFs show `Page N`, PPTX shows `Slide N`, others show `Chunk N` or character offset|
|**Documents from Chat A visible in Chat B uploader**|Streamlit file uploader widget maintains state across rerenders|Per-chat uploader key: `key=f"uploader\\\\\\\_{chat\\\\\\\_id}"` creates isolated widget instances|
|**Conversational titles like "New Chat"**|Auto-title was a simple string truncation of the first question|Single Gemini call on first message: "Generate a 3–5 word title for this question"|
|**Scanned PDFs return empty chunks**|PyPDFLoader cannot extract text from image-only pages|Page-level OCR fallback: pages below 20-character threshold are rendered via PyMuPDF and processed by EasyOCR|
|**WinError 32 on Windows during reprocessing**|Streamlit-cached Chroma connection holds SQLite file handle; deletion fails|Build each upload into a new uniquely-timestamped directory; never delete an open connection|

\---

## Project Structure

```
intelligent-rag-document-qa-system/
│
├── app.py                      # Streamlit application — UI, chat management, state
├── evaluate.py                 # LLM-as-judge evaluation script (Faithfulness + Relevancy)
├── requirements.txt            # All dependencies
├── .env.example                # Environment variable template
├── README.md
│
├── src/
│   ├── \\\\\\\_\\\\\\\_init\\\\\\\_\\\\\\\_.py
│   ├── data\\\\\\\_ingestion.py       # Format router, OCR fallback, chunking
│   ├── vector\\\\\\\_store.py         # ChromaDB create/append/load, MMR setup
│   └── rag\\\\\\\_chain.py            # Query rewriter + answer generator (google.genai)
│
├── chroma\\\\\\\_db\\\\\\\_\\\\\\\*/                # Per-chat vector stores (auto-created, gitignored)
│
└── assets/
    └── screenshots/
        ├── multi\\\\\\\_doc\\\\\\\_qa.png
        ├── conversational\\\\\\\_context.png
        ├── multi\\\\\\\_chat.png
        └── citations.png
```

\---

## Installation

### Prerequisites

* Python 3.10 or higher
* A free [Google AI Studio](https://aistudio.google.com) account for the Gemini API key

### Setup

```bash
# 1. Clone the repository
git clone https://github.com/muhammed-nayifuddin/intelligent-rag-document-qa-system.git
cd intelligent-rag-document-qa-system

# 2. Create and activate a virtual environment
python -m venv venv

# Windows
venv\\\\\\\\Scripts\\\\\\\\activate

# macOS / Linux
source venv/bin/activate

# 3. Install all dependencies
pip install -r requirements.txt
```

> \\\\\\\*\\\\\\\*Note:\\\\\\\*\\\\\\\* The first run downloads the `all-MiniLM-L6-v2` embedding model (\\\\\\\~90 MB) and EasyOCR weights. Subsequent runs use the local cache and start significantly faster.

### Environment Variables

Copy the example file and add your API key:

```bash
cp .env.example .env
```

`.env.example`:

```env
GEMINI\\\\\\\_API\\\\\\\_KEY=your\\\\\\\_gemini\\\\\\\_api\\\\\\\_key\\\\\\\_here
```

Get your free key at [aistudio.google.com](https://aistudio.google.com) → **Get API Key** → **Create API key in new project**.

### Run

```bash
streamlit run app.py
```

The application opens at `http://localhost:8501`.

\---

## Usage

1. **Enter your Gemini API key** in the sidebar Configuration panel.
2. **Upload one or more documents** using the file uploader (PDF, DOCX, PPTX, TXT, MD supported).
3. **Click Process Documents**. The progress bar shows reading, chunking, and embedding stages.
4. **Ask questions** in the chat input. Follow-up questions referencing previous answers work naturally.
5. **Switch between chats** using the sidebar list. Each chat restores its own documents and conversation history instantly.
6. **Inspect sources** using the Sources section below each answer or the expandable Source Chunks panel.

\---

## Evaluation

The repository includes `evaluate.py`, a custom LLM-as-judge evaluation script that measures:

* **Faithfulness** — whether every claim in the answer is supported by the retrieved context
* **Answer Relevancy** — whether the answer addresses the question asked

```bash
# From the project root with venv active
python evaluate.py
```

Results are saved to `ragas\\\\\\\_baseline.csv`. Run before and after adding the reranker to measure the improvement delta.

\---

## Why This Project Stands Out

Most RAG tutorials and demo projects make the same simplifying assumptions: one document, one conversation, one retrieval pass, no follow-up questions. This project was built by systematically rejecting those assumptions.

**Multi-document retrieval** required solving a genuine ChromaDB concurrency problem on Windows where two Chroma clients open to the same SQLite file caused silent write failures. The fix required clearing the Streamlit cache connection before opening an append client, running garbage collection, and sleeping 400ms for Windows to release file handles.

**Conversational references** required a two-stage architecture: a dedicated query rewriting call that receives the full conversation history and produces a reference-resolved standalone query, followed by retrieval and a separate generation call that also receives history so the LLM can resolve any remaining ambiguity. A single-stage approach does not solve this correctly.

**Retrieval diversity** required switching from standard cosine similarity to MMR. With three documents indexed, cosine similarity consistently returned all five results from whichever document scored highest — typically the PPT because slide text is denser and more explanatory than question-bank entries. MMR fetches 20 candidates and selects 5 that are jointly relevant and maximally diverse, ensuring multi-document queries return chunks from across the knowledge base.

**Per-chat isolation** required Streamlit-specific engineering: each chat's file uploader has a unique widget key tied to the chat ID, which causes Streamlit to instantiate a fresh, empty uploader when switching chats. Without this, the file list from one chat persisted visually into another.

The architecture is modular by design. `data\\\\\\\_ingestion.py`, `vector\\\\\\\_store.py`, and `rag\\\\\\\_chain.py` are independent of each other and of the UI. Swapping the LLM from Gemini to another provider, or replacing ChromaDB with a different vector store, requires changing one file.

\---

## Technology Stack

|Layer|Technology|
|-|-|
|Interface|Streamlit|
|LLM|Google Gemini 2.5 Flash (via `google.genai` SDK)|
|Embeddings|`sentence-transformers/all-MiniLM-L6-v2` (CPU, local)|
|Vector database|ChromaDB|
|Orchestration|LangChain, LangChain Community|
|PDF processing|PyPDFLoader, PyMuPDF|
|OCR|EasyOCR|
|DOCX processing|Docx2txt|
|PPTX processing|python-pptx|
|Environment|python-dotenv|
|Retrieval strategy|MMR (Max Marginal Relevance)|

\---

## Future Improvements

* **Streaming responses** — Stream Gemini output token-by-token for lower perceived latency
* **Hybrid search** — Combine semantic similarity with BM25 keyword search for better recall on technical documents
* **Cross-document reasoning** — Explicitly retrieve and synthesize across multiple source documents in a single answer
* **Reranker** — Add a CrossEncoder reranker as a second retrieval stage to improve precision
* **Docker support** — Containerize for consistent deployment across environments
* **Streamlit Cloud deployment** — One-click live demo URL
* **PostgreSQL metadata store** — Persist chat history and document metadata across sessions
* **Redis caching** — Distributed response cache for multi-user deployments
* **User authentication** — Per-user document isolation in a shared deployment
* **Additional formats** — Excel, CSV, HTML, and audio transcript support
* **RAGAS evaluation** — Integrate the RAGAS framework once upstream compatibility with modern LangChain is restored

\---

## Author

**Muhammed Nayifuddin**
B.E. Computer Science \& Engineering (AI \& ML)
Neil Gogte Institute of Technology, Hyderabad

\---

## License

This project is licensed under the MIT License. See [LICENSE](LICENSE) for details.

\---

<div align="center">

*Built with LangChain · ChromaDB · Google Gemini · Streamlit*

</div>

