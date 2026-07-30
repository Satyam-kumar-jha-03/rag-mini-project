import streamlit as st
from openai import OpenAI, RateLimitError, APIStatusError
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import re

GROQ_API_KEY = ""

st.set_page_config(
    page_title="RAG Mini Groq",
    page_icon="📄",
    layout="centered",
    initial_sidebar_state="expanded"
)
st.markdown("""
<style>
    /* Hide Streamlit header, footer, and menu */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Adjust top padding */
    .block-container {
        padding-top: 2rem;
        padding-bottom: 2rem;
    }
    /* Custom Title */
    .main-title {
        font-size: 2.8rem;
        font-weight: 800;
        background: linear-gradient(90deg, #F85A3E, #FF7700);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 0px;
    }
    
    .sub-title {
        text-align: center;
        font-size: 1.1rem;
        color: #888;
        margin-top: 5px;
        margin-bottom: 30px;
    }

    /* Metric cards styling */
    div[data-testid="metric-container"] {
        background-color: #f4f6f9;
        border-radius: 12px;
        padding: 15px;
        border: 1px solid #e0e4e8;
        box-shadow: 0 2px 4px rgba(0,0,0,0.03);
    }
    
    /* Dark mode metric cards */
    @media (prefers-color-scheme: dark) {
        div[data-testid="metric-container"] {
            background-color: #1e1e24;
            border: 1px solid #2b2b36;
        }
    }
    
    /* Chat bubbles styling */
    .stChatMessage {
        border-radius: 15px;
        padding: 10px 15px;
    }
</style>
""", unsafe_allow_html=True)

st.markdown('<p class="main-title">RAG Mini</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Instantly Ask Questions About Your Document (Powered by Groq)</p>', unsafe_allow_html=True)

defaults = {
    "messages": [],
    "document_text": "",
    "document_chunks": [],
    "processed_file_name": None,
    "last_request_time": 0.0,
    "retry_after": 0.0,
    "request_count": 0,
    "model_name": "llama-3.3-70b-versatile",
    "questions_asked": 0,
}
for k, v in defaults.items():
    if k not in st.session_state:
        st.session_state[k] = v

def get_client() -> OpenAI | None:
    """Return an OpenAI-compatible client pointing to Groq."""
    key = GROQ_API_KEY.strip()
    if not key or key == "gsk_your_api_key_here":
        return None
    return OpenAI(
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
    )

def wait_for_rate_limit():
    """Enforce minimum gap between requests to stay within free-tier limits."""
    current_time = time.time()
    if st.session_state.retry_after > current_time:
        time.sleep(st.session_state.retry_after - current_time + 1)
    time_since_last = current_time - st.session_state.last_request_time
    if time_since_last < 2:
        time.sleep(2 - time_since_last)
    st.session_state.last_request_time = time.time()
    st.session_state.request_count += 1


def call_with_retry(client: OpenAI, model: str, system: str, user: str,
                    max_retries: int = 3) -> str:
    """Call Groq chat completion and auto-retry on 429s with UI countdown."""
    for attempt in range(max_retries):
        try:
            wait_for_rate_limit()
            resp = client.chat.completions.create(
                model=model,
                messages=[
                    {"role": "system", "content": system},
                    {"role": "user",   "content": user},
                ],
                max_tokens=1024,
            )
            return resp.choices[0].message.content or ""
        except RateLimitError as e:
            if attempt >= max_retries - 1:
                raise
            error_str = str(e)
            retry_match = re.search(r'retry[_ ](?:after|in)[: ]*(\d+\.?\d*)', error_str, re.IGNORECASE)
            wait_sec = int(float(retry_match.group(1))) + 2 if retry_match else 5
            st.session_state.retry_after = time.time() + wait_sec
            slot = st.empty()
            for remaining in range(wait_sec, 0, -1):
                slot.warning(f"Rate limit hit — auto-retrying in **{remaining}s** (attempt {attempt + 1}/{max_retries})…")
                time.sleep(1)
            slot.empty()
        except APIStatusError as e:
            raise


def find_relevant_chunks(query: str, chunks: list[str], top_k: int = 4) -> list[str]:
    """Keyword-based retrieval."""
    query_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())
    scored = sorted(
        chunks,
        key=lambda c: sum(1 for w in query_words if w in c.lower()),
        reverse=True,
    )
    top = [c for c in scored[:top_k] if any(w in c.lower() for w in query_words)]
    return top if top else chunks[:top_k]

with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3135/3135673.png", width=60)
    st.subheader("Settings & Status")
    
    st.markdown("### Model Selection")
    model_choice = st.selectbox(
        "Choose AI Brain",
        options=[
            "llama-3.3-70b-versatile",
            "llama-3.1-8b-instant",
            "mixtral-8x7b-32768",
            "gemma2-9b-it",
        ],
        index=0,
        help="llama-3.3-70b-versatile is recommended for best reasoning.",
    )
    st.session_state.model_name = model_choice

    st.divider()
    st.markdown("### Usage Stats")
    c1, c2 = st.columns(2)
    c1.metric("Total Requests", st.session_state.request_count)
    c2.metric("Last gap (s)", round(time.time() - st.session_state.last_request_time, 1))

    st.divider()
    st.markdown("### File Limit Active")
    st.info("You can ask up to **3 questions** per document to conserve API limits. Upload a new file to reset the counter.")

    st.divider()
    st.caption("Built with Streamlit + Groq")

st.markdown("### 1 Upload your file")
uploaded_file = st.file_uploader("Upload a PDF or Text file to start chatting", type=["pdf", "txt"], label_visibility="collapsed")

if uploaded_file:
    # Process only when a new file is uploaded
    if uploaded_file.name != st.session_state.processed_file_name:
        text = ""
        with st.spinner("Reading document…"):
            if uploaded_file.type == "application/pdf":
                reader = PyPDF2.PdfReader(uploaded_file)
                for page in reader.pages:
                    text += page.extract_text() or ""
            else:
                text = uploaded_file.read().decode("utf-8")

        splitter = RecursiveCharacterTextSplitter(
            chunk_size=600,
            chunk_overlap=60,
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        chunks = splitter.split_text(text)

        st.session_state.document_text = text
        st.session_state.document_chunks = chunks
        st.session_state.processed_file_name = uploaded_file.name
        st.session_state.messages = []
        st.session_state.questions_asked = 0

        st.success(f"Document loaded! ({len(text):,} characters, {len(chunks)} chunks)")
    else:
        st.info(f"document **{uploaded_file.name}** is loaded and ready.")

    # Stats & preview
    with st.expander("Document Stats & Preview"):
        chunks = st.session_state.document_chunks
        c1, c2, c3 = st.columns(3)
        c1.metric("Characters", f"{len(st.session_state.document_text):,}")
        c2.metric("Chunks", len(chunks))
        avg = sum(len(c) for c in chunks) // max(len(chunks), 1)
        c3.metric("Avg Chunk Size", f"{avg} chars")
        
        st.divider()
        st.markdown("**Preview (first 500 characters):**")
        st.caption(st.session_state.document_text[:500] + "…")

    st.divider()
    st.markdown("### 2️Chat with your document")
    progress_val = st.session_state.questions_asked / 3.0
    st.progress(progress_val, text=f"Questions asked: {st.session_state.questions_asked} / 3")

    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
            
    limit_reached = st.session_state.questions_asked >= 3

    if not limit_reached:
        with st.expander("Not sure what to ask? Click a suggestion"):
            suggestions = [
                "What is my CGPA?",
                "What projects have I done?",
                "What technical skills do I have?",
                "Where did I study?",
                "What certifications do I have?",
            ]
            for i, q in enumerate(suggestions):
                if st.button(q, key=f"sq_{i}", use_container_width=True):
                    st.session_state.messages.append({"role": "user", "content": q})
                    st.session_state.questions_asked += 1
                    st.rerun()
    
    if limit_reached:
        st.error("**Limit Reached:** You have asked 3 questions about this document. Please upload a new file to reset the limit.")

    prompt = st.chat_input("Ask anything about your document…", disabled=limit_reached)

    if prompt:
        st.session_state.questions_asked += 1
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            client = get_client()
            if not client:
                st.error("**API Key Missing!** You need to edit `app.py` and put your Groq API Key in the `GROQ_API_KEY` variable at the top.")
            else:
                with st.spinner("the model is Thinking…"):
                    relevant = find_relevant_chunks(prompt, st.session_state.document_chunks, top_k=4)
                    context = "\n\n---\n\n".join(relevant)

                    system_msg = (
                        "You are a precise document analyst. "
                        "Answer questions based ONLY on the context provided. "
                        "If the answer is not in the context, say: "
                        "'I cannot find this information in the document.' "
                        "Be concise and quote from the context when relevant."
                    )
                    user_msg = f"CONTEXT:\n{context}\n\nQUESTION: {prompt}"

                    try:
                        answer = call_with_retry(
                            client,
                            st.session_state.model_name,
                            system_msg,
                            user_msg,
                        )
                        if "cannot find" in answer.lower():
                            st.warning("Cannot find the requested information in the document.")
                        else:
                            st.success("successfully found the information: " + answer)
                        st.session_state.messages.append({"role": "assistant", "content": answer})
                        with st.expander("Source context used"):
                            st.write(context)
                            
                        st.rerun()

                    except RateLimitError:
                        st.error("Rate limit exhausted after all retries. Please wait a minute.")
                    except Exception as e:
                        error_str = str(e)
                        if "invalid-argument" in error_str or "401" in error_str:
                            st.error("**Invalid API Key!** The key hardcoded in `app.py` is incorrect.")
                        else:
                            st.error(f"Error: {e}")
                            st.info("Check your API key or try a different model.")

    st.divider()
    col1, col2 = st.columns(2)
    with col1:
        if st.button("Clear Chat History", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.session_state.messages:
            chat_text = "\n\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in st.session_state.messages
            )
            st.download_button(
                label="Download Chat Log",
                data=chat_text,
                file_name="chat_history.txt",
                mime="text/plain",
                use_container_width=True
            )

else:
    st.info("👆 Please upload a document to get started!")
    
    st.markdown("""
    <div style="padding: 20px; border-radius: 10px; background-color: rgba(255, 119, 0, 0.1); border: 1px solid rgba(255, 119, 0, 0.2); margin-top: 20px;">
        <h4 style="margin-top: 0;">How it works</h4>
        <ol style="margin-bottom: 0;">
            <li><b>Upload</b> a PDF or Text file</li>
            <li><b>Ask</b> any question based on the document contents</li>
            <li>Get <b>instant answers</b> powered by Groq's lightning fast AI</li>
        </ol>
    </div>
    """, unsafe_allow_html=True)