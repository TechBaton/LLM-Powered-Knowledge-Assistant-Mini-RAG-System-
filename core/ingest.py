import os #used to iterate files over a directory
from langchain_community.document_loaders import PyPDFLoader #to extract text from pdf files
from langchain.text_splitter import RecursiveCharacterTextSplitter #split large documents into small chunks
from langchain_community.embeddings import HuggingFaceEmbeddings #text to dense vector embeddings
from langchain_community.vectorstores import FAISS #stores and searches for vectors

DATA_PATH = "data/documents"

documents = []
for file in os.listdir(DATA_PATH):
    if file.endswith(".pdf"):
        loader = PyPDFLoader(os.path.join(DATA_PATH, file))
        documents.extend(loader.load()) #extracted text is stored as "documents" with metadata

splitter = RecursiveCharacterTextSplitter(chunk_size=500, chunk_overlap=50) #500 character chunks, 50 character-overlap to ensure semantic continuity
chunks = splitter.split_documents(documents)

embeddings = HuggingFaceEmbeddings(
    model_name="sentence-transformers/all-MiniLM-L6-v2"
) #converts each chunk into a high-dimensional vector

db = FAISS.from_documents(chunks, embeddings) #FAISS indexes the embeddings for fast similarity search (FAISS acts as retrieval backend)
db.save_local("faiss_index") #indexes are saved locally to avoid reuse during querying

print(f"Loaded {len(documents)} pages")
print(f"Created {len(chunks)} chunks")
