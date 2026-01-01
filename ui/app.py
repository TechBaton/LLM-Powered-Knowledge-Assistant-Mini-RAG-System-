import streamlit as st
import os
import shutil
from langchain_community.document_loaders import PyPDFLoader
from langchain.text_splitter import RecursiveCharacterTextSplitter
from langchain_community.embeddings import HuggingFaceEmbeddings
from langchain_community.vectorstores import FAISS
from langchain_community.llms import HuggingFacePipeline
from langchain.chains import RetrievalQA
from transformers import pipeline

from random import sample

def generate_suggested_questions(chunks, llm, max_questions=6):
    """
    Generate suggested user questions based on sampled document chunks.
    """
    prompt_template = (
        "You are given a document excerpt.\n"
        "Generate ONE clear and relevant question that a user might ask "
        "based strictly on this content.\n\n"
        "Content:\n{content}\n\nQuestion:"
    )

    questions = []

    sampled_chunks = sample(chunks, min(3, len(chunks)))

    for doc in sampled_chunks:
        response = llm(
            prompt_template.format(
                content=doc.page_content[:500]
            )
        )
        questions.append(response.strip())

        if len(questions) >= max_questions:
            break

    return questions

if "suggested_questions" not in st.session_state:
    st.session_state.suggested_questions = []

# -------- Session State --------
if "history" not in st.session_state:
    st.session_state.history = []

# ---------------- CONFIG ----------------
DATA_DIR = "data/documents"
INDEX_DIR = "faiss_index"

st.set_page_config(page_title="Knowledge Assistant", layout="wide")
st.title("LLM-Powered Knowledge Assistant")

# ---------------- CACHING ----------------

@st.cache_resource
def load_llm():
    return HuggingFacePipeline(
        pipeline=pipeline(
            "text2text-generation",
            model="google/flan-t5-base",
            max_length=512,
            device=-1
        )
    )

@st.cache_resource
def load_embeddings():
    return HuggingFaceEmbeddings(
        model_name="sentence-transformers/all-MiniLM-L6-v2"
    )

@st.cache_resource
def load_db():
    if not os.path.exists(INDEX_DIR):
        return None
    return FAISS.load_local(
        INDEX_DIR,
        load_embeddings(),
        allow_dangerous_deserialization=True
    )

llm = load_llm()
db = load_db()

# ---------------- FILE UPLOAD ----------------

st.sidebar.header("Document Upload")

uploaded_files = st.sidebar.file_uploader(
    "Upload PDF documents",
    type=["pdf"],
    accept_multiple_files=True
)

if st.sidebar.button("Build / Rebuild Index"):
    if not uploaded_files:
        st.sidebar.warning("Please upload at least one PDF.")
    else:
        with st.spinner("Building vector index..."):
            # Reset folders
            shutil.rmtree(DATA_DIR, ignore_errors=True)
            shutil.rmtree(INDEX_DIR, ignore_errors=True)
            os.makedirs(DATA_DIR, exist_ok=True)

            # Save PDFs
            for file in uploaded_files:
                with open(os.path.join(DATA_DIR, file.name), "wb") as f:
                    f.write(file.read())

            # Ingest documents
            documents = []
            for file in os.listdir(DATA_DIR):
                loader = PyPDFLoader(os.path.join(DATA_DIR, file))
                documents.extend(loader.load())

            splitter = RecursiveCharacterTextSplitter(
                chunk_size=500,
                chunk_overlap=50
            )
            chunks = splitter.split_documents(documents)
            # Store chunks for later question regeneration
            st.session_state.chunks = chunks


            db = FAISS.from_documents(chunks, load_embeddings())
            db.save_local(INDEX_DIR)

            # Generate suggested questions once after indexing
            st.session_state.suggested_questions = generate_suggested_questions(
            chunks, llm
            )   

        st.sidebar.success("Index built successfully")

if st.sidebar.button("Clear Query History"):
    st.session_state.history = []
    st.sidebar.success("Query history cleared.")
    st.rerun()


# ---------------- QUERY UI ----------------

st.header("Ask a Question")

# -------- Suggested Questions --------
if st.session_state.suggested_questions:
    st.subheader("Suggested Questions")

    for q in st.session_state.suggested_questions:
        if st.button(q):
            st.session_state.prefill_query = q
            st.rerun()

    # Reload questions button
    if st.button("Reload questions"):
        with st.spinner("Generating new suggested questions..."):
            st.session_state.suggested_questions = generate_suggested_questions(
                st.session_state.chunks,
                llm
            )
        st.rerun()
else:
    st.info("Suggested questions will appear after documents are indexed.")

# -------- Query Input --------
query = st.text_input(
    "Enter your question",
    value=st.session_state.get("prefill_query", "")
)

top_k = st.slider("Top-K Retrieved Chunks", 1, 5, 3)

if query:
    if db is None:
        st.warning("Please upload documents and build the index first.")
    else:
        with st.spinner("Retrieving documents and generating answer..."):
            retriever = db.as_retriever(search_kwargs={"k": top_k})

            qa = RetrievalQA.from_chain_type(
                llm=llm,
                retriever=retriever,
                return_source_documents=True
            )

            result = qa(query)
            
            # Save to session history
            st.session_state.history.append({
            "query": query,
            "answer": result["result"],
            "sources": [doc.metadata.get("source", "Unknown") 
                for doc in result["source_documents"]]
            })

            # Clear prefilled query after use
            if "prefill_query" in st.session_state:
                del st.session_state.prefill_query



        # ----------- OUTPUT -----------

        st.subheader("Answer")
        st.write(result["result"])

        st.subheader("Retrieved Context")
        for i, doc in enumerate(result["source_documents"], 1):
            with st.expander(f"Chunk {i}"):
                st.write(doc.page_content)

        st.subheader("Sources")
        for doc in result["source_documents"]:
            st.write(doc.metadata.get("source", "Unknown"))

        st.subheader("Query History")

        if st.session_state.history:
            for i, item in enumerate(reversed(st.session_state.history), 1):
                with st.expander(f"Query {len(st.session_state.history) - i + 1}"):
                    st.markdown("**Question:**")
                    st.write(item["query"])

                    st.markdown("**Answer:**")
                    st.write(item["answer"])

                    st.markdown("**Sources:**")
                    for src in item["sources"]:
                        st.write("-", src)
        else:
            st.write("No queries yet in this session.")
