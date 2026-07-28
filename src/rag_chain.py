from google import genai as google_genai
from langchain_core.runnables import RunnableLambda
from langchain_core.messages import HumanMessage
from src.vector_store import load_vector_store

# ── Prompt 1: Rewrite ambiguous queries for retrieval ────────────────────────
# Critical for "next?", "tell me about them", "its advantages", etc.
# Extra instructions handle very short/vague queries specifically.

_CONTEXTUALIZE_TEMPLATE = """You are a search query rewriter for a document retrieval system.

Given the conversation history and the user's latest message, rewrite the message
into a fully explicit, self-contained search query.

Rules:
1. Resolve ALL pronouns: it, its, them, they, this, that, those, these, their, he, she, his, her.
2. Resolve positional references: next, previous, next one, next chapter, next unit,
   the first one, the second one, former, latter.
3. If the message is very short (like "next?", "and?", "more?", "them?"), use the
   conversation history to determine what the user is asking about and write a
   complete, specific search query.
4. Do NOT answer the question. Return ONLY the rewritten query.
5. If the message is already fully self-contained, return it exactly as written.

Conversation history:
{history}

User's message: {question}

Rewritten search query (be specific — expand abbreviations and resolve all references):"""


# ── Prompt 2: Generate grounded answer with conversation context ──────────────
_QA_TEMPLATE = """You are a helpful assistant that answers questions strictly from the provided documents.

Use ONLY the context below to answer the question.
If the context does not contain the answer, say exactly:
"I don't have enough information in the provided documents to answer this."
Do not fabricate any information.
{history_block}
Context from retrieved documents:
{context}

Question: {question}

Answer:"""


def _history_to_text(chat_history: list) -> str:
    lines = []
    for msg in chat_history:
        role = "User" if isinstance(msg, HumanMessage) else "Assistant"
        # Truncate very long assistant messages to keep the rewrite prompt focused
        content = msg.content
        if role == "Assistant" and len(content) > 800:
            content = content[:800] + "... [truncated]"
        lines.append(f"{role}: {content}")
    return "\n".join(lines)


def build_rag_chain(gemini_api_key: str, model_name: str, chroma_dir: str):
    """
    Build a conversational RAG chain using google.genai SDK directly.
    Compatible with AQ... and AIza... API key formats.

    Pipeline per question:
      1. Rewrite the question using chat history to resolve all references.
         This fixes "next?", "tell me about them", "its advantages", etc.
      2. Retrieve top-5 chunks using MMR (Maximal Marginal Relevance).
         MMR ensures diversity across documents — prevents all 5 chunks
         coming from the same document when multiple docs are indexed.
         fetch_k=20 candidates are retrieved, then 5 diverse ones selected.
      3. Generate a grounded answer passing BOTH the retrieved context
         AND the conversation history so the LLM can resolve any remaining
         references in the original question.
    """
    client = google_genai.Client(api_key=gemini_api_key)

    vector_store = load_vector_store(chroma_dir)

    # MMR retrieval: ensures we get diverse chunks across all uploaded documents.
    # Without MMR, similarity search returns the top-k nearest neighbors which
    # can all come from a single document that happens to score well.
    retriever = vector_store.as_retriever(
        search_type="mmr",
        search_kwargs={
            "k":           5,    # return 5 chunks
            "fetch_k":     20,   # fetch 20 candidates first
            "lambda_mult": 0.6,  # 0 = max diversity, 1 = max relevance
        },
    )

    def run_pipeline(input_dict: dict) -> dict:
        question     = input_dict["input"]
        chat_history = input_dict.get("chat_history", [])

        # ── Step 1: rewrite question for retrieval ────────────────────────────
        if chat_history:
            history_text = _history_to_text(chat_history)
            rewrite_prompt = _CONTEXTUALIZE_TEMPLATE.format(
                history=history_text,
                question=question,
            )
            rw = client.models.generate_content(
                model=model_name, contents=rewrite_prompt
            )
            standalone = rw.text.strip()
            # Safety: if the model returns something too long or weird, fall back
            if len(standalone) > 500 or "\n" in standalone[:50]:
                standalone = question
        else:
            standalone = question

        # ── Step 2: MMR retrieval ─────────────────────────────────────────────
        source_docs = retriever.invoke(standalone)
        context     = "\n\n".join(doc.page_content for doc in source_docs)

        # ── Step 3: grounded answer with history for reference resolution ─────
        if chat_history:
            history_text = _history_to_text(chat_history)
            history_block = (
                "\nConversation history (use this to resolve references like "
                "\"them\", \"it\", \"next\", etc. — do NOT use it as a source of facts):\n"
                + history_text + "\n"
            )
        else:
            history_block = ""

        qa_prompt = _QA_TEMPLATE.format(
            context=context,
            question=question,
            history_block=history_block,
        )
        qa = client.models.generate_content(model=model_name, contents=qa_prompt)
        return {"answer": qa.text.strip(), "context": source_docs}

    return RunnableLambda(run_pipeline)
