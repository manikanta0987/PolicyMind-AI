"""PolicyMind AI — run from the project folder with: streamlit run app.py.

Uses the existing src/ modules and Ollama setup. No new dependencies required.
"""

from html import escape

import streamlit as st

from src.document_processor import extract_pdf_documents, split_documents
from src.embeddings import get_embeddings
from src.vector_store import create_vector_store, search_policies
from src.rag_pipeline import ask_policy_question


st.set_page_config(
    page_title="PolicyMind AI | Policy assistant",
    page_icon="🌿",
    layout="wide",
    initial_sidebar_state="expanded",
)

for key, value in {
    "vector_store": None,
    "documents": [],
    "chunks": [],
    "chat_history": [],
    "upload_version": 0,
}.items():
    if key not in st.session_state:
        st.session_state[key] = value


# All styling lives here so the Python workflow stays easy to follow.
st.markdown(
    """
<style>
:root { --ink: #20382f; --muted: #69776f; --green: #287052; }
.stApp { background: #f7f8f4; color: var(--ink); }
[data-testid="stHeader"] { background: #f7f8f4; }
.block-container { max-width: 1160px; padding-top: 3rem; padding-bottom: 2rem; }
h1, h2, h3, p { color: inherit; }
[data-testid="stCaptionContainer"] { color: var(--muted); }
.eyebrow { font-size: 11px; letter-spacing: 2px; font-weight: 700; color: #6c7e72; }
.topbar { display: flex; justify-content: space-between; align-items: center;
    gap: 12px; border-bottom: 1px solid #dfe5dc; padding-bottom: 20px; margin-bottom: 34px; }
.breadcrumb { font-size: 13px; color: #748077; }
.breadcrumb strong { color: #283f34; font-weight: 600; }
.status { font-size: 12px; color: #42654e; border: 1px solid #d4e1d4;
    background: #edf4e9; border-radius: 24px; padding: 7px 12px; white-space: nowrap; }
.hero { padding: 0 0 27px; }
.hero h1 { font-size: clamp(32px, 4vw, 49px); font-weight: 600;
    letter-spacing: -1.8px; line-height: 1.12; margin: 12px 0 14px; padding: 0; }
.hero h1 span { color: #39805b; }
.hero p { color: #69776f; font-size: 15px; line-height: 1.7; max-width: 550px; margin: 0; }
.guide { background: #eaf0e4; border: 1px solid #dbe5d4; border-radius: 16px;
    padding: 19px 22px; margin: 0 0 24px; display: flex; gap: 16px; align-items: center; }
.step { flex-shrink: 0; height: 34px; width: 34px; border-radius: 10px;
    display: grid; place-items: center; background: #d6e4ca; color: #315f40; font-weight: 700; }
.guide strong { font-size: 14px; font-weight: 650; }
.guide p { font-size: 13px; color: #62725e; line-height: 1.5; margin: 3px 0 0; }
.section-heading { font-size: 11px; font-weight: 700; letter-spacing: 1.7px;
    color: #748076; margin-bottom: 12px; }
.topic { margin: 9px 0 6px; font-size: 16px; font-weight: 650; letter-spacing: -.3px; }
.topic-description { color: #6a786e; font-size: 13px; line-height: 1.6; min-height: 43px; }
.topic-icon { background: #eef3e8; color: #46754e; width: 34px; height: 34px;
    display: grid; place-items: center; border-radius: 10px; font-size: 17px; }
[data-testid="stVerticalBlockBorderWrapper"] > div {
    border-color: #e0e6dc !important; border-radius: 16px !important; background: #fff; }
[data-testid="stButton"] button, [data-testid="stDownloadButton"] button {
    border: 1px solid #d7e1d3; border-radius: 10px; background: #fff;
    color: #315741; font-weight: 500; min-height: 42px; }
[data-testid="stButton"] button:hover { border-color: #588364; background: #edf4e9; color: #244c35; }
[data-testid="stButton"] button:focus-visible { outline: 3px solid #7aa889; outline-offset: 2px; }
[data-testid="stButton"] button:disabled { opacity: .5; }
[data-testid="stButton"] button[kind="primary"] { background: #326f50; border-color: #326f50; color: #fff; }
[data-testid="stButton"] button[kind="primary"]:hover { background: #24573c; color: #fff; }
.conversation-heading { font-size: 18px; font-weight: 600; margin-top: 21px; }
.conversation-caption { color: #718075; font-size: 13px; margin-top: 4px; margin-bottom: 16px; }
.empty-chat { padding: 26px 20px; text-align: center; background: #fff;
    border: 1px dashed #d9e2d3; border-radius: 16px; margin-bottom: 12px; }
.empty-chat strong { font-size: 14px; color: #536c58; }
.empty-chat p { color: #7b877c; font-size: 13px; margin: 5px 0 0; }
[data-testid="stChatMessage"] { background: #fff; border: 1px solid #e0e7dc; border-radius: 16px; padding: 20px; }
[data-testid="stChatMessage"] p { line-height: 1.75; }
[data-testid="stExpander"] { background: #fff; color: #365641; border-radius: 10px; }
[data-testid="stBottom"], [data-testid="stBottom"] > div { background: #f7f8f4; }
[data-testid="stChatInput"] { border-radius: 16px; border: 1px solid #ccdac5;
    background: #fff; box-shadow: 0 4px 18px #253c2a08; }
[data-testid="stChatInput"] textarea { color: #263e2e; }
.source-row { padding: 10px 0; border-bottom: 1px solid #edf0e8; font-size: 13px; overflow-wrap: anywhere; }
.source-row span { display: block; font-size: 12px; color: #71806e; margin-top: 4px; }
.footnote { text-align: center; color: #7b887b; font-size: 11px; margin: 17px 0 0; }

/* Document controls live in the sidebar; chat owns the main canvas. */
[data-testid="stSidebar"] { background: #192e27; border-right: none; }
[data-testid="stSidebar"] [data-testid="stSidebarContent"] { background: #192e27; }
[data-testid="stSidebar"] [data-testid="stSidebarUserContent"] { padding: 24px 22px; }
[data-testid="stSidebar"] p, [data-testid="stSidebar"] label,
[data-testid="stSidebar"] [data-testid="stCaptionContainer"] { color: #afc1b4; }
[data-testid="stSidebar"] hr { border-color: #344a3e; margin: 23px 0; }
.brand { display: flex; align-items: center; gap: 10px; margin-bottom: 6px; }
.brand-mark { width: 35px; height: 35px; border-radius: 10px; display: grid; place-items: center;
    background: #c9e5b3; color: #234b31; font-size: 23px; font-weight: 700; }
.brand-name { color: #f3f7ed; font-size: 22px; font-weight: 650; letter-spacing: -.8px; }
.brand-name span { font-size: 10px; padding: 3px 5px; margin-left: 3px; border: 1px solid #526b57;
    border-radius: 5px; color: #c0d6b9; vertical-align: middle; letter-spacing: .5px; }
.brand-caption { color: #a3b7a8; font-size: 12px; margin: 9px 0 26px; }
.side-label { color: #a4b8a8; font-size: 10px; font-weight: 650; letter-spacing: 1.6px; margin-bottom: 12px; }
.side-stats { display: flex; gap: 10px; margin-bottom: 24px; }
.side-stat { flex: 1; background: #223b30; border: 1px solid #365040; padding: 12px 14px; border-radius: 12px; }
.side-stat strong { color: #ecf4e4; font-size: 23px; font-weight: 550; }
.side-stat span { display: block; color: #a6bcab; font-size: 11px; margin-top: 3px; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] {
    background: #233b31; border: 1px dashed #59735f; border-radius: 12px; padding: 12px; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] * { color: #c7d5c6; }
[data-testid="stSidebar"] [data-testid="stFileUploaderDropzone"] button { background: #355342; color: #f0f5e8; border-color: #55705a; }
[data-testid="stSidebar"] [data-testid="stButton"] button { background: #243e31; border-color: #46614b; color: #dbe8d1; }
[data-testid="stSidebar"] [data-testid="stButton"] button p { color: inherit; }
[data-testid="stSidebar"] [data-testid="stButton"] button[kind="primary"] { background: #c8e3ad; color: #253f2c; border-color: #c8e3ad; }
[data-testid="stSidebar"] [data-testid="stButton"] button:hover { border-color: #a8c696; }
.doc-row { color: #d9e7d1; font-size: 12px; border-bottom: 1px solid #344c3c;
    padding: 10px 0; overflow-wrap: anywhere; line-height: 1.6; }
.doc-row small { color: #9ab391; display: block; font-size: 10px; }
.engine { background: #213a2d; border-radius: 12px; padding: 14px; margin-bottom: 13px; }
.engine strong { color: #d4e2c8; font-size: 12px; font-weight: 550; }
.engine p { font-size: 11px; line-height: 1.7; margin: 5px 0 0; color: #a7bba3; }
@media (max-width: 700px) {
    .block-container { padding: 2.7rem 1rem 1.5rem; }
    .hero h1 { font-size: 34px; letter-spacing: -1px; }
    .topbar { margin-bottom: 25px; }
    .status { font-size: 10px; padding: 6px 9px; }
    .guide { padding: 15px; }
}
</style>
""",
    unsafe_allow_html=True,
)


def clear_workspace():
    """Reset documents, conversation, and the uploader together."""
    st.session_state.vector_store = None
    st.session_state.documents = []
    st.session_state.chunks = []
    st.session_state.chat_history = []
    st.session_state.upload_version += 1


def render_message(message):
    with st.chat_message(message["role"]):
        st.markdown(message["content"])
        if message.get("sources"):
            with st.expander(f"View sources · {len(message['sources'])}"):
                for source in message["sources"]:
                    # Escape filenames and metadata before putting them in HTML.
                    name = escape(str(source["source"]))
                    page = escape(str(source["page"]))
                    st.markdown(
                        f'<div class="source-row">{name}<span>Page {page}</span></div>',
                        unsafe_allow_html=True,
                    )


with st.sidebar:
    st.markdown(
        '<div class="brand"><div class="brand-mark">P</div>'
        '<div class="brand-name">PolicyMind <span>AI</span></div></div>'
        '<div class="brand-caption">Your company knowledge, made clear.</div>',
        unsafe_allow_html=True,
    )
    st.button("＋  New conversation", use_container_width=True,
              disabled=not st.session_state.chat_history,
              on_click=lambda: st.session_state.update(chat_history=[]))
    st.divider()
    st.markdown('<div class="side-label">POLICY WORKSPACE</div>', unsafe_allow_html=True)
    st.markdown(
        f'<div class="side-stats"><div class="side-stat"><strong>{len(st.session_state.documents)}</strong>'
        f'<span>Documents</span></div><div class="side-stat"><strong>{len(st.session_state.chunks)}</strong>'
        '<span>Searchable sections</span></div></div>', unsafe_allow_html=True,
    )
    st.markdown('<div class="side-label">ADD DOCUMENTS</div>', unsafe_allow_html=True)
    uploaded_files = st.file_uploader(
        "Upload company policy PDFs", type=["pdf"], accept_multiple_files=True,
        key=f"policy_upload_{st.session_state.upload_version}", label_visibility="collapsed",
    )
    st.caption("Employee handbooks, leave policies, travel guidelines and more.")
    if st.button("Process policies  →", type="primary", use_container_width=True,
                 disabled=not uploaded_files):
        try:
            with st.spinner("Preparing your policy workspace…"):
                all_documents = []
                for uploaded_file in uploaded_files:
                    uploaded_file.seek(0)
                    all_documents.extend(extract_pdf_documents(uploaded_file))
                if not all_documents:
                    st.error("No readable text found. Upload a text-based PDF; scanned pages may need OCR.")
                else:
                    chunks = split_documents(all_documents)
                    if not chunks:
                        st.error("No searchable sections found. Try another PDF.")
                    else:
                        embeddings = get_embeddings()
                        vector_store = create_vector_store(chunks, embeddings)
                        # Commit only after indexing succeeds. Old answers belong to the old PDFs.
                        st.session_state.documents = list(uploaded_files)
                        st.session_state.chunks = chunks
                        st.session_state.vector_store = vector_store
                        st.session_state.chat_history = []
                        st.rerun()
        except Exception as error:
            st.error("Could not process the PDFs. Check that Ollama is running and your models are installed.")
            with st.expander("Error details"):
                st.code(str(error))
    if st.session_state.documents:
        st.caption("Processing a new selection replaces the current documents and conversation.")
        st.markdown('<div class="side-label">YOUR DOCUMENTS</div>', unsafe_allow_html=True)
        for document in st.session_state.documents:
            st.markdown(
                f'<div class="doc-row">{escape(document.name)}<small>Indexed for search</small></div>',
                unsafe_allow_html=True,
            )
    st.divider()
    st.markdown(
        '<div class="engine"><strong>Local AI · Ollama</strong>'
        '<p>Uses your project’s Ollama configuration.<br>Answers reference your uploaded policies.</p></div>',
        unsafe_allow_html=True,
    )
    st.button("Clear workspace", use_container_width=True, on_click=clear_workspace,
              disabled=not (st.session_state.documents or uploaded_files or st.session_state.chat_history))


ready = st.session_state.vector_store is not None
status = "●  Policies ready" if ready else "○  Awaiting documents"
st.markdown(
    '<div class="topbar"><div class="breadcrumb">Workspace &nbsp;/&nbsp; <strong>Policy assistant</strong></div>'
    f'<div class="status">{status}</div></div>', unsafe_allow_html=True,
)
st.markdown(
    '<div class="hero"><div class="eyebrow">LESS SEARCHING. MORE CLARITY.</div>'
    '<h1>Your policies.<br><span>Plain and simple.</span></h1>'
    '<p>From time off to working from home, find answers in your company documents, with sources you can check.</p></div>',
    unsafe_allow_html=True,
)

if not ready:
    st.markdown(
        '<div class="guide"><div class="step">1</div><div><strong>Start with your company documents</strong>'
        '<p>Upload PDFs in the sidebar, then select Process policies to start asking questions.</p></div></div>',
        unsafe_allow_html=True,
    )

question = None
if not st.session_state.chat_history:
    st.markdown('<div class="section-heading">A FEW PLACES TO START</div>', unsafe_allow_html=True)
    topics = [
        ("◷", "Time off & leave", "Understand leave allowances and how to apply.", "What is the annual leave policy?"),
        ("⌂", "Work from anywhere", "Check remote work rules and eligibility.", "What are the work-from-home rules?"),
        ("↗", "Expenses & travel", "Find reimbursement rules and claim steps.", "How do I claim travel expenses?"),
    ]
    for column, (icon, title, description, prompt) in zip(st.columns(3), topics):
        with column:
            with st.container(border=True):
                st.markdown(
                    f'<div class="topic-icon">{icon}</div><div class="topic">{title}</div>'
                    f'<div class="topic-description">{description}</div>', unsafe_allow_html=True,
                )
                if st.button("Ask about this  ↗", key=title, use_container_width=True, disabled=not ready):
                    question = prompt

st.markdown('<div class="conversation-heading">Your conversation</div>', unsafe_allow_html=True)
st.markdown(
    '<div class="conversation-caption">Ask a question. Get an answer grounded in your documents.</div>',
    unsafe_allow_html=True,
)
if not st.session_state.chat_history and not question:
    message = "What would you like to know?" if ready else "Your policy answers start here"
    detail = "Choose a topic above or write your own question below." if ready else "Once your documents are ready, ask your first question below."
    st.markdown(f'<div class="empty-chat"><strong>{message}</strong><p>{detail}</p></div>', unsafe_allow_html=True)

for message in st.session_state.chat_history:
    render_message(message)

typed_question = st.chat_input(
    "Ask about leave, benefits, expenses…" if ready else "Upload and process your PDFs to start…",
    disabled=not ready,
)
question = typed_question or question
if question and ready:
    user_message = {"role": "user", "content": question}
    st.session_state.chat_history.append(user_message)
    render_message(user_message)
    try:
        with st.spinner("Finding answers in your policies…"):
            retrieved_documents = search_policies(st.session_state.vector_store, question, k=4)
            sources = []
            if retrieved_documents:
                answer = ask_policy_question(question, retrieved_documents)
                seen_sources = set()
                for document in retrieved_documents:
                    source = document.metadata.get("source", "Unknown document")
                    page = document.metadata.get("page", "Unknown")
                    source_key = (source, page)
                    if source_key not in seen_sources:
                        sources.append({"source": source, "page": page})
                        seen_sources.add(source_key)
            else:
                answer = "I couldn't find relevant policy sections. Try rephrasing your question or uploading the policy that covers this topic."
        st.session_state.chat_history.append({"role": "assistant", "content": answer, "sources": sources})
        st.rerun()
    except Exception as error:
        # Keep a visible, truthful outcome in history if the local model fails.
        failure = {"role": "assistant", "content": "I couldn't complete that request. Check that Ollama is running, then send your question again."}
        st.session_state.chat_history.append(failure)
        render_message(failure)
        with st.expander("Error details"):
            st.code(str(error))

st.markdown(
    '<div class="footnote">AI answers can be incomplete. Check the cited policy or confirm with HR.</div>',
    unsafe_allow_html=True,
)
