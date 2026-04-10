# 🤖 RAG Q&A Chatbot

Welcome to the **RAG Q&A Chatbot**, a comprehensive, end-to-end Retrieval-Augmented Generation (RAG) project that allows you to have intelligent, context-aware conversations with your own PDF documents. 

This educational and practical project leverages the power of **Python**, **LangChain**, and **Streamlit** to create a pipeline that ingests raw documents and turns them into a queryable knowledge base. Unlike basic chatbots, this system significantly reduces hallucination by rigorously rooting its answers in the provided context and providing transparent source citations for every response.

### How it works deep dive:
1. **Intelligent Ingestion:** It uses LangChain's `PyPDFLoader` to parse all PDF documents placed in the local `pdfs/` directory, extracting both text and metadata.
2. **Semantic Chunking:** Instead of blindly cutting text by character counts (which often breaks sentences or thoughts in half), the project uses `SemanticChunker` from `langchain_experimental`. This groups text by similar meaning using embeddings, ensuring that chunks are contextually cohesive.
3. **Local Vector Storage:** These cohesive chunks are converted into mathematical vectors via `OpenAIEmbeddings` and stored in **ChromaDB**, an open-source vector database. Crucially, this database is persisted locally in the `chroma_db/` folder, meaning you only pay OpenAI to embed your documents once, saving time and API credits on subsequent runs.
4. **Retrieval & Synthesis:** When you ask a question, the system converts your query into an embedding and performs a similarity search in ChromaDB. The top most relevant document chunks are retrieved and injected as context into a prompt template. Finally, `gpt-3.5-turbo` synthesizes a concise, accurate answer based *only* on that context.
5. **Dual Interfaces:** You can interact with the bot through a beautifully crafted **Streamlit web interface** or a fast, text-based **terminal interface**. Both interfaces provide exact citations, detailing the source file, page number, chunk index, and word count used to formulate the answer.

## ✨ Features

- **Dual Interfaces:** Chat via a beautiful **Streamlit Web UI** or directly in your terminal.
- **Smart Semantic Chunking:** Uses LangChain's AI-driven `SemanticChunker` to break down documents intelligently by topic and meaning, rather than rigid, unreadable character counts.
- **Local Vector Database:** Powered by **ChromaDB**, ensuring fast similarity searches. Embeddings are persisted locally in `chroma_db/` so you don't waste API credits reloading them on every run.
- **Precise Source Attribution:** Whenever the bot answers a question, it explicitly cites the exact PDF `file_name`, `page number`, `chunk_index`, and `word count` it used as context.
- **Minimal Configuration:** Uses simple `.env` setups and standard Python dependencies.

## 🛠️ Technology Stack

- **[LangChain](https://python.langchain.com/)**: Core framework handling data loading, chunking, and RAG retrieval chains.
- **[Streamlit](https://streamlit.io/)**: For the highly-interactive frontend application.
- **[Chroma DB](https://www.trychroma.com/)**: The open-source vector database used for embedding storage.
- **[OpenAI](https://openai.com/)**: 
  - *Embeddings:* Default `OpenAIEmbeddings`
  - *LLM:* `gpt-3.5-turbo` handling final generation.

## 🚀 Quick Start Guide

### 1. Clone the Repository
```bash
git clone https://github.com/yourusername/your-repo-name.git
cd your-repo-name
```

### 2. Create and Activate a Virtual Environment
It is highly recommended to use a virtual environment to manage dependencies:
```bash
python3 -m venv .venv
source .venv/bin/activate  # On macOS/Linux
# .venv\Scripts\activate   # On Windows
```

### 3. Install Dependencies
```bash
pip install -r requirements.txt
```

### 4. Setup Environment Variables
Create a newly renamed `.env` file in the root directory (you can use `.env.example` as a template) and add your OpenAI API Key:
```env
OPENAI_API_KEY=sk-your-openai-api-key-here
```

### 5. Add Your Documents
Create a folder named `pdfs/` in the root directory if it doesn't already exist, and drop any PDF files you want to analyze inside of it.

### 6. Run the Application
You can interact with your documents in two amazing ways:

**Option A: Streamlit Web UI (Recommended)**
```bash
streamlit run app.py
```
*Access the interface locally at `http://localhost:8501`*

**Option B: Terminal Interface**
```bash
python rag_chatbot.py
```

## 📂 Project Structure

```text
.
├── app.py                  # Streamlit Web Application interface
├── rag_chatbot.py          # Core RAG logic Engine & interactive Terminal UI
├── requirements.txt        # Core Python dependencies
├── .env                    # Secret API Key config (Not checked into git)
├── .env.example            # Boilerplate config file for Git
├── chroma_db/              # (Auto-generated) Local persistent vector store
└── pdfs/                   # (User-created) Place your target knowledge PDFs here
```

## 🧠 How It Works Under The Hood
1. **Document Loading**: Unpacks local files using `PyPDFLoader` and strips out initial metadata.
2. **Text Splitting**: Uses `SemanticChunker` to map transitions in meaning and safely cut text sections without ruining context. 
3. **Embedding**: Transforms textual chunks to numbers via `OpenAIEmbeddings`.
4. **Storage**: Saves embeddings securely inside a local `ChromaDB` directory for caching.
5. **Retrieval & QA**: When a user prompts the system, ChromaDB runs a vector similarity search for the top most relevant documents, and passes them to an LLM chain to synthesize an accurate, concise answer without hallucinating.
