import streamlit as st
from openai import OpenAI, RateLimitError, APIStatusError
import PyPDF2
from langchain_text_splitters import RecursiveCharacterTextSplitter
import time
import re
from datetime import datetime

GROQ_API_KEY = st.secrets["GROQ_API_KEY"]

st.set_page_config(
    page_title="RAG Mini Groq",
    page_icon="",
    layout="wide",
    initial_sidebar_state="expanded"
)

# Enhanced CSS with modern design
st.markdown("""
<style>
    /* Hide Streamlit defaults */
    #MainMenu {visibility: hidden;}
    footer {visibility: hidden;}
    header {visibility: hidden;}
    
    /* Global styles */
    .stApp {
        background: linear-gradient(135deg, #0f0c29, #302b63, #24243e);
    }
    
    .main-title {
        font-size: 3.5rem;
        font-weight: 800;
        background: linear-gradient(135deg, #FF6B6B, #FF8E53, #FFC947);
        -webkit-background-clip: text;
        -webkit-text-fill-color: transparent;
        text-align: center;
        margin-bottom: 0px;
        padding-bottom: 0px;
        animation: fadeInDown 0.8s ease-out;
        letter-spacing: -1px;
    }
    
    .sub-title {
        text-align: center;
        font-size: 1.2rem;
        color: #a0a0c0;
        margin-top: 5px;
        margin-bottom: 30px;
        font-weight: 300;
        animation: fadeInUp 0.8s ease-out;
    }
    
    @keyframes fadeInDown {
        from {
            opacity: 0;
            transform: translateY(-30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    @keyframes fadeInUp {
        from {
            opacity: 0;
            transform: translateY(30px);
        }
        to {
            opacity: 1;
            transform: translateY(0);
        }
    }
    
    /* Card styling */
    .custom-card {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 16px;
        padding: 24px;
        border: 1px solid rgba(255, 255, 255, 0.1);
        box-shadow: 0 8px 32px rgba(0, 0, 0, 0.3);
        margin-bottom: 20px;
        transition: all 0.3s ease;
    }
    
    .custom-card:hover {
        transform: translateY(-2px);
        box-shadow: 0 12px 40px rgba(0, 0, 0, 0.4);
        border-color: rgba(255, 107, 107, 0.3);
    }
    
    /* Metric cards */
    div[data-testid="metric-container"] {
        background: rgba(255, 255, 255, 0.05);
        backdrop-filter: blur(10px);
        border-radius: 12px;
        padding: 20px;
        border: 1px solid rgba(255, 255, 255, 0.08);
        transition: all 0.3s ease;
    }
    
    div[data-testid="metric-container"]:hover {
        background: rgba(255, 255, 255, 0.08);
        border-color: rgba(255, 107, 107, 0.2);
        transform: scale(1.02);
    }
    
    /* Chat message styling */
    .stChatMessage {
        border-radius: 16px !important;
        padding: 16px 20px !important;
        margin-bottom: 12px !important;
        animation: slideIn 0.3s ease-out;
        backdrop-filter: blur(10px);
    }
    
    @keyframes slideIn {
        from {
            opacity: 0;
            transform: translateX(-20px);
        }
        to {
            opacity: 1;
            transform: translateX(0);
        }
    }
    
    .stChatMessage[data-testid="chat-message-user"] {
        background: linear-gradient(135deg, rgba(255, 107, 107, 0.2), rgba(255, 142, 83, 0.2)) !important;
        border: 1px solid rgba(255, 107, 107, 0.2);
        margin-left: 20%;
    }
    
    .stChatMessage[data-testid="chat-message-assistant"] {
        background: linear-gradient(135deg, rgba(99, 102, 241, 0.15), rgba(129, 140, 248, 0.15)) !important;
        border: 1px solid rgba(99, 102, 241, 0.15);
        margin-right: 20%;
    }
    
    /* Custom button styling */
    .stButton > button {
        background: linear-gradient(135deg, #FF6B6B, #FF8E53);
        color: white;
        border: none;
        border-radius: 12px;
        padding: 10px 24px;
        font-weight: 600;
        transition: all 0.3s ease;
        text-transform: none;
        letter-spacing: 0.5px;
    }
    
    .stButton > button:hover {
        transform: translateY(-2px);
        box-shadow: 0 8px 25px rgba(255, 107, 107, 0.4);
        background: linear-gradient(135deg, #FF5A5A, #FF7A3A);
    }
    
    /* File uploader styling */
    .stFileUploader > div {
        border: 2px dashed rgba(255, 107, 107, 0.3);
        border-radius: 16px;
        padding: 30px;
        background: rgba(255, 255, 255, 0.03);
        transition: all 0.3s ease;
    }
    
    .stFileUploader > div:hover {
        border-color: #FF6B6B;
        background: rgba(255, 107, 107, 0.05);
    }
    
    /* Progress bar */
    .stProgress > div > div {
        background: linear-gradient(135deg, #FF6B6B, #FFC947);
        border-radius: 20px;
    }
    
    /* Info boxes */
    .stAlert {
        border-radius: 12px !important;
        border-left: 4px solid #FF6B6B !important;
        background: rgba(255, 107, 107, 0.1) !important;
        backdrop-filter: blur(10px);
    }
    
    /* Sidebar styling */
    .css-1d391kg {
        background: rgba(15, 12, 41, 0.8);
        backdrop-filter: blur(20px);
        border-right: 1px solid rgba(255, 255, 255, 0.05);
    }
    
    /* Scrollbar styling */
    ::-webkit-scrollbar {
        width: 8px;
        height: 8px;
    }
    
    ::-webkit-scrollbar-track {
        background: rgba(255, 255, 255, 0.05);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb {
        background: linear-gradient(180deg, #FF6B6B, #FFC947);
        border-radius: 10px;
    }
    
    ::-webkit-scrollbar-thumb:hover {
        background: linear-gradient(180deg, #FF5A5A, #FFB830);
    }
    
    /* Typing indicator */
    .typing-indicator {
        display: inline-block;
        padding: 10px 20px;
        background: rgba(255, 255, 255, 0.05);
        border-radius: 20px;
        color: #a0a0c0;
    }
    
    .typing-dot {
        display: inline-block;
        width: 8px;
        height: 8px;
        margin: 0 2px;
        background: #FF6B6B;
        border-radius: 50%;
        animation: bounce 1.4s infinite both;
    }
    
    .typing-dot:nth-child(1) { animation-delay: -0.32s; }
    .typing-dot:nth-child(2) { animation-delay: -0.16s; }
    
    @keyframes bounce {
        0%, 80%, 100% { transform: scale(0); }
        40% { transform: scale(1); }
    }
    
    /* Status badges */
    .status-badge {
        display: inline-block;
        padding: 4px 12px;
        border-radius: 20px;
        font-size: 0.8rem;
        font-weight: 600;
        background: rgba(255, 107, 107, 0.2);
        color: #FF6B6B;
        border: 1px solid rgba(255, 107, 107, 0.2);
    }
    
    .status-badge.success {
        background: rgba(51, 217, 178, 0.2);
        color: #33d9b2;
        border-color: rgba(51, 217, 178, 0.2);
    }
</style>
""", unsafe_allow_html=True)

# Title section with animation
st.markdown('<p class="main-title"> RAG Mini</p>', unsafe_allow_html=True)
st.markdown('<p class="sub-title">Instant Document Q&A Powered by Groq</p>', unsafe_allow_html=True)

# Session state initialization
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
    key = GROQ_API_KEY.strip()
    if not key or key == "gsk_your_api_key_here":
        return None
    return OpenAI(
        api_key=key,
        base_url="https://api.groq.com/openai/v1",
    )

def wait_for_rate_limit():
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
                slot.warning(f"⏳ Rate limit hit — retrying in **{remaining}s** (attempt {attempt + 1}/{max_retries})…")
                time.sleep(1)
            slot.empty()
        except APIStatusError as e:
            raise

def find_relevant_chunks(query: str, chunks: list[str], top_k: int = 4) -> list[str]:
    query_words = set(re.sub(r'[^\w\s]', '', query.lower()).split())
    scored = sorted(
        chunks,
        key=lambda c: sum(1 for w in query_words if w in c.lower()),
        reverse=True,
    )
    top = [c for c in scored[:top_k] if any(w in c.lower() for w in query_words)]
    return top if top else chunks[:top_k]

# Sidebar with enhanced design
with st.sidebar:
    st.markdown("### Control Panel")
    
    with st.container():
        st.markdown("#### Model Selection")
        model_choice = st.selectbox(
            "Choose AI Brain",
            options=[
                "llama-3.3-70b-versatile",
                "llama-3.1-8b-instant",
                "mixtral-8x7b-32768",
                "gemma2-9b-it",
            ],
            index=0,
            help="llama-3.3-70b-versatile recommended for best reasoning",
        )
        st.session_state.model_name = model_choice
    
    st.divider()
    
    st.markdown("#### Usage Statistics")
    col1, col2 = st.columns(2)
    with col1:
        st.metric("Total Requests", st.session_state.request_count)
    with col2:
        time_since = round(time.time() - st.session_state.last_request_time, 1)
        st.metric("⏱️ Last Request", f"{time_since}s ago" if time_since < 60 else f"{int(time_since/60)}m ago")
    
    st.divider()
    
    st.markdown("#### Document Status")
    if st.session_state.processed_file_name:
        status_color = "success" if st.session_state.questions_asked < 3 else ""
        st.markdown(f'<span class="status-badge {status_color}">📄 {st.session_state.processed_file_name}</span>', unsafe_allow_html=True)
        st.progress(st.session_state.questions_asked / 3.0, text=f"Questions: {st.session_state.questions_asked}/3")
    else:
        st.markdown('<span class="status-badge"> No document loaded</span>', unsafe_allow_html=True)
    
    st.divider()
    
    st.markdown("####Quick Tips")
    with st.expander("How to use"):
        st.markdown("""
        1. **Upload** a PDF or text file
        2. **Ask** questions about the content
        3. Get **instant AI-powered answers**
        4. **3 questions** per document (API limit)
        """)
    
    st.divider()
    st.caption("Built with Streamlit + Groq • v2.0")

# Main content area
col1, col2, col3 = st.columns([1, 2, 1])
with col2:
    uploaded_file = st.file_uploader(
        "Upload your document",
        type=["pdf", "txt"],
        label_visibility="collapsed",
        help="Supported formats: PDF, TXT"
    )

if uploaded_file:
    # Process file
    if uploaded_file.name != st.session_state.processed_file_name:
        with st.spinner("Reading and processing document..."):
            text = ""
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

        st.success(f"Document loaded successfully! ({len(text):,} chars, {len(chunks)} chunks)")
        st.balloons()
    
    # Document stats
    with st.expander("Document Statistics", expanded=False):
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Characters", f"{len(st.session_state.document_text):,}")
        with col2:
            st.metric("Chunks", len(st.session_state.document_chunks))
        with col3:
            avg = sum(len(c) for c in st.session_state.document_chunks) // max(len(st.session_state.document_chunks), 1)
            st.metric("Avg Chunk", f"{avg} chars")
        with col4:
            st.metric("Questions", f"{st.session_state.questions_asked}/3")
        
        if st.session_state.document_text:
            with st.expander("Preview"):
                st.text(st.session_state.document_text[:500] + "…")

    # Chat interface
    st.markdown("### Chat with your document")
    
    # Chat history
    for msg in st.session_state.messages:
        with st.chat_message(msg["role"]):
            st.write(msg["content"])
    
    # Quick suggestion buttons
    limit_reached = st.session_state.questions_asked >= 3
    
    if not limit_reached and st.session_state.document_chunks:
        with st.expander("Quick Questions", expanded=False):
            suggestions = [
                "What is the main topic?",
                "What are the key points?",
                "Summarize this document",
                "What questions does this answer?",
            ]
            cols = st.columns(2)
            for i, q in enumerate(suggestions):
                with cols[i % 2]:
                    if st.button(q, key=f"sq_{i}", use_container_width=True):
                        st.session_state.messages.append({"role": "user", "content": q})
                        st.session_state.questions_asked += 1
                        st.rerun()
    
    if limit_reached:
        st.warning("**Question limit reached!** Upload a new document to continue chatting.")
    
    # Chat input
    prompt = st.chat_input("Ask anything about your document...", disabled=limit_reached)

    if prompt:
        st.session_state.questions_asked += 1
        st.session_state.messages.append({"role": "user", "content": prompt})
        with st.chat_message("user"):
            st.write(prompt)

        with st.chat_message("assistant"):
            client = get_client()
            if not client:
                st.error("**API Key Missing!** Please add your Groq API key to `secrets.toml`")
            else:
                # Typing indicator
                typing_placeholder = st.empty()
                typing_placeholder.markdown("""
                    <div class="typing-indicator">
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        <span class="typing-dot"></span>
                        &nbsp; Thinking...
                    </div>
                """, unsafe_allow_html=True)
                
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
                    
                    typing_placeholder.empty()
                    
                    if "cannot find" in answer.lower():
                        st.warning("Cannot find the requested information in the document.")
                    else:
                        st.success("Sucessfully Found answer!")
                    
                    st.write(answer)
                    st.session_state.messages.append({"role": "assistant", "content": answer})
                    
                    with st.expander("📖 Source Context"):
                        st.text(context)
                    
                    st.rerun()

                except RateLimitError:
                    typing_placeholder.empty()
                    st.error("⏳ Rate limit exhausted. Please wait a moment.")
                except Exception as e:
                    typing_placeholder.empty()
                    error_str = str(e)
                    if "invalid-argument" in error_str or "401" in error_str:
                        st.error("❌ **Invalid API Key!** Please check your Groq API key.")
                    else:
                        st.error(f"❌ Error: {e}")
                        st.info("💡 Try switching the model or check your API key.")

    # Action buttons
    col1, col2, col3 = st.columns([1, 1, 2])
    with col1:
        if st.button("🗑️ Clear Chat", use_container_width=True):
            st.session_state.messages = []
            st.rerun()
    with col2:
        if st.session_state.messages:
            chat_text = "\n\n".join(
                f"{m['role'].upper()}: {m['content']}"
                for m in st.session_state.messages
            )
            st.download_button(
                label="💾 Download Chat",
                data=chat_text,
                file_name=f"chat_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                mime="text/plain",
                use_container_width=True
            )

else:
    # Welcome screen
    st.markdown("""
    <div style="text-align: center; padding: 40px 20px;">
        <div style="font-size: 4rem; margin-bottom: 20px;">📄</div>
        <h2 style="color: #FF6B6B; margin-bottom: 10px;">Ready to Chat with Your Documents?</h2>
        <p style="color: #a0a0c0; font-size: 1.1rem;">Upload a PDF or text file to start asking questions</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Feature cards
    col1, col2, col3 = st.columns(3)
    with col1:
        st.markdown("""
        <div class="custom-card" style="text-align: center;">
            <div style="font-size: 2.5rem;">📤</div>
            <h4>Upload</h4>
            <p style="color: #a0a0c0; font-size: 0.9rem;">PDF or TXT files supported</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col2:
        st.markdown("""
        <div class="custom-card" style="text-align: center;">
            <div style="font-size: 2.5rem;">thinking</div>
            <h4>Ask</h4>
            <p style="color: #a0a0c0; font-size: 0.9rem;">Natural language questions</p>
        </div>
        """, unsafe_allow_html=True)
    
    with col3:
        st.markdown("""
        <div class="custom-card" style="text-align: center;">
            <div style="font-size: 2.5rem;">⚡</div>
            <h4>Get Answers</h4>
            <p style="color: #a0a0c0; font-size: 0.9rem;">Powered by Groq's lightning speed</p>
        </div>
        """, unsafe_allow_html=True)