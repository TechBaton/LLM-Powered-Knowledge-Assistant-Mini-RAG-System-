import sys
import os

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

import streamlit as st
from core.rag import load_rag
from langchain.chains import RetrievalQA

st.set_page_config(page_title="Knowledge Assistant", layout="wide")
st.title("LLM-Powered Knowledge Assistant")

query = st.text_input("Ask a question")
top_k = st.slider("Top-K Retrieved Chunks", 1, 5, 3)

db, llm = load_rag()

if query:
    retriever = db.as_retriever(search_kwargs={"k": top_k})

    qa = RetrievalQA.from_chain_type(
        llm=llm,
        retriever=retriever,
        return_source_documents=True
    )

    result = qa(query)

    st.subheader("Answer")
    st.write(result["result"])

    st.subheader("Retrieved Context")
    for i, doc in enumerate(result["source_documents"], 1):
        with st.expander(f"Chunk {i}"):
            st.write(doc.page_content)

    st.subheader("Sources")
    for doc in result["source_documents"]:
        st.write(doc.metadata.get("source", "Unknown"))
