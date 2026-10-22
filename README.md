# 🎓 StudyMate — AI-Powered Study Assistant & Agent

**StudyMate** is an intelligent, multi-modal AI study companion and agentic orchestration system designed to help students master course materials faster. Upload lecture slides, textbooks, or research papers in PDF format and seamlessly chat, summarize, generate quizzes, build structured study plans, and trigger multi-step autonomous study agent workflows.

---

## ✨ Features

- 📑 **Smart Document Ingestion (RAG)**:
  - Upload PDF textbooks, notes, and research papers.
  - Chunked and embedded using **FastEmbed** or **Google Generative AI Embeddings** with a high-performance local **FAISS** vector store.
- 🤖 **Autonomous Multi-Step Study Agent**:
  - Structured agent orchestration with explicit planning, step-by-step execution, and user feedback loops.
  - Generates comprehensive notes, key takeaways, and flashcards grounded in your uploaded documents.
- 💬 **Interactive Document Chat**:
  - Context-grounded conversational Q&A with direct citation and retrieval.
- 📝 **Auto-Generated Summaries & Notes**:
  - High-yield bullet summaries, executive takeaways, and concept breakdowns.
- 🎯 **Interactive Quizzes & Practice Tests**:
  - Automatically generate multiple-choice quizzes to test your knowledge retention with instant grading and explanations.
- ⚡ **Dual LLM Provider Support**:
  - Ultra-fast responses powered by **Groq** (`ChatGroq`).
  - Native integration with **Google Gemini** (`ChatGoogleGenerativeAI`).

---

## 🛠️ Tech Stack

- **Frontend & App Framework**: [Streamlit](https://streamlit.io/)
- **LLM Orchestration**: [LangChain](https://www.langchain.com/) (`langchain-core`, `langchain-community`, `langchain-groq`, `langchain-google-genai`)
- **Vector Search**: [FAISS](https://github.com/facebookresearch/faiss) (`faiss-cpu`)
- **Embeddings**: [FastEmbed](https://github.com/qdrant/fastembed) & Google Generative AI Embeddings
- **Document Parsing**: [pypdf](https://github.com/py-pdf/pypdf)
- **Environment**: `python-dotenv`

---

## 🚀 Quick Start

### 1. Clone the Repository
```bash
git clone https://github.com/latish-charpe/StudyMate.git
cd StudyMate
```

### 2. Set Up a Virtual Environment
```bash
# Windows (PowerShell)
python -m venv .venv
.\.venv\Scripts\Activate.ps1

# macOS / Linux
python3 -m venv .venv
source .venv/bin/activate
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Configure Environment Variables
Copy `.env.example` to `.env`:
```bash
cp .env.example .env
```

Open `.env` and configure your API key(s):
```env
# Groq API Configuration (https://console.groq.com/keys)
GROQ_API_KEY=your_groq_api_key_here
GROQ_MODEL=openai/gpt-oss-20b

# (Optional) Google Gemini Configuration
# GOOGLE_API_KEY=your_gemini_api_key_here
# GEMINI_CHAT_MODEL=gemini-3.8-flash
# GEMINI_EMBEDDING_MODEL=gemini-embedding-001
```

### 5. Run the Application
```bash
streamlit run app.py
```

The app will launch in your default web browser at `http://localhost:8501`.

---

## 📂 Project Structure

```text
StudyMate/
├── app.py                 # Streamlit web application & user interface
├── study_agent.py         # Autonomous multi-step study agent controller & tools
├── requirements.txt       # Project dependencies
├── .env.example           # Example environment variables template
├── .gitignore             # Git ignore patterns (protects secrets & caches)
├── .streamlit/
│   └── config.toml        # Streamlit theme & server configuration
├── assets/                # UI graphics and card images
└── vectorstore/           # Local FAISS vector index (generated on document upload)
```

---

## 🛡️ License

This project is licensed under the [MIT License](LICENSE) (or specify your preferred license).
