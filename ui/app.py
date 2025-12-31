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

            db = FAISS.from_documents(chunks, load_embeddings())
            db.save_local(INDEX_DIR)

        st.sidebar.success("Index built successfully")

# ---------------- QUERY UI ----------------

st.header("Ask a Question")

query = st.text_input("Enter your question")
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
