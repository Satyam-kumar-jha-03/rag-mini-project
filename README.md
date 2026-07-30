
                   RAG Mini - Document Q&A System

Features
- Upload PDF/TXT files
- Ask questions about your documents
- Context-aware answers from Groq's Llama 3.3 70B
- Live usage tracking
- Download chat history

Tech Stack
- Python, Streamlit, Groq API, LangChain, PyPDF2

Live Demo
[https://rag-mini-project-im7sg6m283xtyup5w2hwxf.streamlit.app/]

How It Works
1. Document is split into 600-character chunks
2. Your question retrieves top 4 relevant chunks
3. Groq AI generates answer based ONLY on those chunks
