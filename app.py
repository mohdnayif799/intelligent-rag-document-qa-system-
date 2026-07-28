import gc
import os
import tempfile
import time
import uuid

import streamlit as st
from dotenv import load_dotenv
from google import genai as google_genai
from langchain_core.messages import HumanMessage, AIMessage

from src.data_ingestion import load_documents, chunk_documents, SUPPORTED_EXTENSIONS
from src.vector_store import (
    create_chat_vector_store,
    append_to_vector_store,
    load_vector_store,
    get_store_stats,
)
from src.rag_chain import build_rag_chain

load_dotenv()

# ── Page config ───────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Document QA System",
    page_icon="\U0001f4c4",
    layout="wide",
)

# ── Global CSS ────────────────────────────────────────────────────────────────
st.markdown("""
<style>
/* Active chat item */
.active-chat-item {
    background: linear-gradient(90deg, rgba(79,70,229,0.13) 0%, rgba(79,70,229,0.04) 100%);
    border-left: 3px solid #4F46E5;
    border-radius: 0 8px 8px 0;
    padding: 9px 14px;
    font-weight: 600;
    color: #4F46E5;
    margin: 2px 0 2px 0;
    font-size: 14px;
    line-height: 1.4;
    cursor: default;
    white-space: nowrap;
    overflow: hidden;
    text-overflow: ellipsis;
}
/* Inactive chat buttons */
div[data-testid="stSidebar"] div.stButton button {
    border-radius: 8px;
    font-size: 14px;
    text-align: left;
}
/* Status box */
.status-box {
    background: rgba(16, 185, 129, 0.08);
    border: 1px solid rgba(16, 185, 129, 0.3);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    line-height: 1.7;
}
.status-box-warn {
    background: rgba(245, 158, 11, 0.08);
    border: 1px solid rgba(245, 158, 11, 0.3);
    border-radius: 8px;
    padding: 10px 14px;
    font-size: 13px;
    line-height: 1.7;
}
</style>
""", unsafe_allow_html=True)


# ── Helpers: citation display ─────────────────────────────────────────────────

def _format_location(doc, chunk_index: int) -> str:
    source = doc.metadata.get("source", "")
    ext    = os.path.splitext(source)[1].lower()
    page   = doc.metadata.get("page", None)
    if ext == ".pdf" and page is not None:
        return f"Page {int(page) + 1}"
    if ext == ".pptx" and page is not None:
        return f"Slide {int(page) + 1}"
    if ext in {".txt", ".md"}:
        start = doc.metadata.get("start_index", None)
        return f"~char {int(start):,}" if start is not None else f"Chunk {chunk_index}"
    return f"Chunk {chunk_index}"


def _doc_type_label(source: str) -> str:
    ext = os.path.splitext(source)[1].lower()
    return {
        ".pdf":  "\U0001f4d5 PDF",
        ".pptx": "\U0001f4ca Presentation",
        ".docx": "\U0001f4c4 Document",
        ".doc":  "\U0001f4c4 Document",
        ".txt":  "\U0001f4dd Text file",
        ".md":   "\U0001f4dd Markdown",
    }.get(ext, "\U0001f4c4 File")


def _build_inline_citations(source_docs: list) -> str:
    grouped: dict = {}
    for i, doc in enumerate(source_docs, 1):
        sf  = doc.metadata.get("source", "unknown")
        key = os.path.basename(sf)
        loc = _format_location(doc, i)
        grouped.setdefault(key, []).append(loc)
    lines = []
    for fname, locs in grouped.items():
        ext   = os.path.splitext(fname)[1].lower()
        emoji = {".pdf": "\U0001f4d5", ".pptx": "\U0001f4ca",
                 ".docx": "\U0001f4c4", ".doc": "\U0001f4c4",
                 ".txt": "\U0001f4dd", ".md": "\U0001f4dd"}.get(ext, "\U0001f4c4")
        lines.append(f"{emoji} **{fname}** \u2014 {', '.join(locs)}")
    return "\n\n".join(lines)


# ── Helper: LLM-generated chat title ─────────────────────────────────────────

def _generate_title(question: str, api_key: str, model: str) -> str:
    """
    Use Gemini to generate a concise 3-5 word title from the first question.
    Falls back to truncated question if the API call fails.
    """
    try:
        client = google_genai.Client(api_key=api_key)
        prompt = (
            "Generate a concise chat title (3 to 5 words maximum) that captures "
            "the topic of this question. Return ONLY the title — no punctuation, "
            "no quotes, no explanation.\n\n"
            f"Question: {question}"
        )
        resp  = client.models.generate_content(model=model, contents=prompt)
        title = resp.text.strip().strip('"\'').strip()
        words = title.split()
        if len(words) > 6:
            title = " ".join(words[:5]) + "..."
        return title if title else _auto_title(question)
    except Exception:
        return _auto_title(question)


# ── State helpers ─────────────────────────────────────────────────────────────

def _blank_chat() -> dict:
    return {
        "title":          "New Chat",
        "messages":       [],
        "lc_history":     [],
        "chroma_dir":     None,
        "uploaded_files": [],
        "response_cache": {},
    }


def _init():
    if "chats" not in st.session_state:
        first_id = str(uuid.uuid4())
        st.session_state.chats      = {first_id: _blank_chat()}
        st.session_state.chat_order = [first_id]
        st.session_state.active_id  = first_id


def _active() -> dict:
    return st.session_state.chats[st.session_state.active_id]


def _auto_title(text: str) -> str:
    text = text.strip()
    return (text[:30] + "...") if len(text) > 30 else text


def _new_chat():
    nid = str(uuid.uuid4())
    st.session_state.chats[nid] = _blank_chat()
    st.session_state.chat_order.insert(0, nid)
    st.session_state.active_id = nid


# ── Bootstrap ─────────────────────────────────────────────────────────────────
_init()

# ── Sidebar ───────────────────────────────────────────────────────────────────
with st.sidebar:

    # Configuration
    st.markdown("### \u2699\ufe0f Configuration")
    gemini_api_key = st.text_input(
        "Gemini API Key",
        type="password",
        value=os.getenv("GEMINI_API_KEY", ""),
        help="Get your free key at https://aistudio.google.com",
    )
    model_name = st.selectbox(
        "Model",
        options=["gemini-2.5-flash", "gemini-2.0-flash", "gemini-2.0-flash-lite"],
        index=0,
    )

    st.divider()

    # New Chat
    if st.button("\u2795  New Chat", use_container_width=True):
        _new_chat()
        st.rerun()

    # Chat list with active highlighting
    if st.session_state.chat_order:
        st.markdown("**Chats**")
        for cid in st.session_state.chat_order:
            c      = st.session_state.chats[cid]
            active = cid == st.session_state.active_id
            title  = c["title"]
            if active:
                # Styled div — no click needed, already active
                st.markdown(
                    f'<div class="active-chat-item">\u25b6&nbsp;&nbsp;{title}</div>',
                    unsafe_allow_html=True,
                )
            else:
                if st.button(
                    f"\u00a0\u00a0\u00a0{title}",
                    key=f"sw_{cid}",
                    use_container_width=True,
                ):
                    st.session_state.active_id = cid
                    st.rerun()

    st.divider()

    # Upload Documents
    # ── Improvement 4 fix: unique key per chat ─────────────────────────────
    # When the active chat changes, Streamlit creates a new uploader widget
    # (blank) for the new chat's ID. This prevents documents from one chat
    # visually carrying over to another chat's uploader.
    aid_for_key = st.session_state.active_id

    st.markdown("**\U0001f4c1 Upload Documents**")
    st.caption(f"Supported: {', '.join(sorted(SUPPORTED_EXTENSIONS))}")

    uploaded_files = st.file_uploader(
        "Drop files here or click Browse",
        type=[ext.lstrip(".") for ext in SUPPORTED_EXTENSIONS],
        accept_multiple_files=True,
        label_visibility="collapsed",
        key=f"uploader_{aid_for_key}",    # unique per chat — resets on chat switch
    )

    process_btn = st.button(
        "\u26a1 Process Documents",
        disabled=not uploaded_files,
        use_container_width=True,
        type="primary",
    )

    if process_btn:
        if not gemini_api_key:
            st.error("Enter your Gemini API key above.")
        else:
            aid         = st.session_state.active_id
            known_files = st.session_state.chats[aid]["uploaded_files"]

            new_uploads  = [f for f in uploaded_files if f.name not in known_files]
            already_have = [f.name for f in uploaded_files if f.name in known_files]

            if already_have:
                st.info(f"Already indexed, skipping: {', '.join(already_have)}")

            if not new_uploads:
                st.warning("No new documents to process.")
            else:
                with st.spinner(f"Processing {len(new_uploads)} file(s)\u2026"):
                    try:
                        temp_paths, original_names = [], []
                        for f in new_uploads:
                            suffix = os.path.splitext(f.name)[1]
                            with tempfile.NamedTemporaryFile(
                                delete=False, suffix=suffix
                            ) as tmp:
                                tmp.write(f.read())
                                temp_paths.append(tmp.name)
                                original_names.append(f.name)

                        progress_bar = st.progress(0, text="Reading pages\u2026")

                        def update_progress(current, total):
                            pct = int((current / total) * 80)
                            progress_bar.progress(
                                pct, text=f"Reading page {current}/{total}\u2026"
                            )

                        docs = load_documents(temp_paths, progress_callback=update_progress)

                        for doc in docs:
                            for tp, on in zip(temp_paths, original_names):
                                if doc.metadata.get("source") == tp:
                                    doc.metadata["source"] = on
                                    break

                        progress_bar.progress(85, text="Chunking text\u2026")
                        chunks = chunk_documents(docs)
                        progress_bar.progress(90, text="Embedding chunks\u2026")

                        current_chroma = st.session_state.chats[aid]["chroma_dir"]

                        if current_chroma is None:
                            _, chroma_dir = create_chat_vector_store(chunks)
                            st.session_state.chats[aid]["chroma_dir"] = chroma_dir
                        else:
                            # Safely append — cache cleared before open to avoid SQLite lock
                            append_to_vector_store(chunks, current_chroma)

                        st.session_state.chats[aid]["uploaded_files"].extend(original_names)

                        for p in temp_paths:
                            os.unlink(p)

                        progress_bar.progress(100, text="Done!")
                        n_docs = len(st.session_state.chats[aid]["uploaded_files"])
                        st.success(
                            f"\u2705 Indexed **{len(chunks)} new chunks**.\n"
                            f"This chat now has **{n_docs} document(s)** searchable."
                        )

                    except Exception as e:
                        st.error(f"Failed to process documents: {e}")

    st.divider()

    # Active Documents list
    active_chat = _active()
    st.markdown("**\U0001f4cb Active Documents**")
    if active_chat["uploaded_files"]:
        for fname in active_chat["uploaded_files"]:
            st.caption(f"\u2022 {fname}  ·  {_doc_type_label(fname)}")
    else:
        st.caption("No documents loaded for this chat.")

    st.divider()

    # ── Improvement 3: Status Indicator ───────────────────────────────────────
    st.markdown("**\U0001f4e1 System Status**")
    if not gemini_api_key:
        st.markdown(
            '<div class="status-box-warn">\U0001f7e1 <b>Waiting</b><br>'            'Enter Gemini API key to begin.</div>',
            unsafe_allow_html=True,
        )
    elif not active_chat["chroma_dir"]:
        st.markdown(
            '<div class="status-box-warn">\U0001f7e1 <b>No Documents</b><br>'            f'Model: {model_name}<br>'            'Upload and process a document to start.</div>',
            unsafe_allow_html=True,
        )
    else:
        stats      = get_store_stats(active_chat["chroma_dir"])
        total      = stats.get("total_chunks", 0)
        n_docs     = len(active_chat["uploaded_files"])
        st.markdown(
            '<div class="status-box">'            '\U0001f7e2 <b>Ready</b><br>'            f'Documents: {n_docs} indexed<br>'            f'Chunks in store: {total}<br>'            f'Model: {model_name}</div>',
            unsafe_allow_html=True,
        )


# ── Main area ─────────────────────────────────────────────────────────────────
st.title("\U0001f4c4 Intelligent Document QA System")
st.caption("Upload documents in the sidebar, then ask questions below.")

active_chat = _active()

for msg in active_chat["messages"]:
    with st.chat_message(msg["role"]):
        st.markdown(msg["content"])

# ── Chat input ────────────────────────────────────────────────────────────────
if user_question := st.chat_input("Ask a question about your documents\u2026"):
    if not gemini_api_key:
        st.warning("\u26a0\ufe0f Enter your Gemini API key in the sidebar.")
    elif not active_chat["chroma_dir"]:
        st.warning("\u26a0\ufe0f Upload and process at least one document first.")
    else:
        aid = st.session_state.active_id

        # ── Improvement 2: LLM-generated title on first message ───────────────
        if not st.session_state.chats[aid]["messages"]:
            st.session_state.chats[aid]["title"] = _generate_title(
                user_question, gemini_api_key, model_name
            )

        st.session_state.chats[aid]["messages"].append(
            {"role": "user", "content": user_question}
        )
        with st.chat_message("user"):
            st.markdown(user_question)

        with st.chat_message("assistant"):
            try:
                cache_key = user_question.strip().lower()
                cache     = st.session_state.chats[aid]["response_cache"]

                if cache_key in cache:
                    answer      = cache[cache_key]["answer"]
                    source_docs = cache[cache_key]["source_docs"]
                    st.caption("\U0001f5c4\ufe0f *(retrieved from session cache)*")
                else:
                    with st.spinner("Searching documents and generating answer\u2026"):
                        chain = build_rag_chain(
                            gemini_api_key,
                            model_name,
                            st.session_state.chats[aid]["chroma_dir"],
                        )
                        result = chain.invoke({
                            "input":        user_question,
                            "chat_history": st.session_state.chats[aid]["lc_history"],
                        })
                        answer      = result["answer"]
                        source_docs = result["context"]
                        cache[cache_key] = {"answer": answer, "source_docs": source_docs}

                st.markdown(answer)

                if source_docs:
                    citations = _build_inline_citations(source_docs)
                    st.markdown("---\n**Sources:**\n\n" + citations)

                if source_docs:
                    with st.expander("\U0001f4da Source Chunks Used"):
                        for i, doc in enumerate(source_docs, 1):
                            sf  = doc.metadata.get("source", "unknown")
                            loc = _format_location(doc, i)
                            st.markdown(
                                f"**Chunk {i}** \u2014 `{os.path.basename(sf)}` "
                                f"· {_doc_type_label(sf)} · **{loc}**"
                            )
                            st.caption(doc.page_content[:500] + "\u2026")
                            st.divider()

                st.session_state.chats[aid]["messages"].append(
                    {"role": "assistant", "content": answer}
                )
                st.session_state.chats[aid]["lc_history"].extend([
                    HumanMessage(content=user_question),
                    AIMessage(content=answer),
                ])

            except Exception as e:
                error_msg = f"Error generating answer: {e}"
                st.error(error_msg)
                st.session_state.chats[aid]["messages"].append(
                    {"role": "assistant", "content": error_msg}
                )
