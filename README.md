# LLM-Based Knowledge Assistant (RAG System)

An interactive, open-source **Retrieval-Augmented Generation (RAG)** based knowledge assistant that allows users to upload PDF documents and ask natural-language questions grounded strictly in the uploaded content.

The system is built entirely using **open-source models** and does **not rely on OpenAI or proprietary APIs**.

---

## 🚀 Features

- Upload multiple PDF documents through the UI
- Semantic search using vector embeddings (FAISS)
- Context-aware answer generation using an open-source LLM
- Automatically generated **suggested questions** based on uploaded documents
- Reload questions option for diverse exploration
- Session-level query history
- Transparent display of retrieved context chunks
- Optimized performance using model and index caching
- Simple, interactive Streamlit-based UI

---

## 🧠 System Overview

The system follows a **Retrieval-Augmented Generation (RAG)** pipeline:

1. User uploads PDF documents
2. Documents are split into overlapping text chunks
3. Each chunk is converted into a vector embedding
4. Embeddings are stored in a FAISS vector index
5. User asks a natural-language question
6. Relevant chunks are retrieved using semantic similarity
7. A language model generates an answer using the retrieved context
8. The UI displays:
   - Answer
   - Retrieved context
   - Source documents
   - Query history

---


## 📦 Tech Stack

- **Frontend / UI**: Streamlit
- **LLM**: google/flan-t5-base
- **Embeddings**: sentence-transformers/all-MiniLM-L6-v2
- **Vector Store**: FAISS
- **Framework**: LangChain (community modules)
- **Document Parsing**: PyPDF
- **Language**: Python 3.10+


