import chromadb
from sentence_transformers import SentenceTransformer

# Persistent vector DB (local)
client = chromadb.PersistentClient(path="../data/chroma")
collection = client.get_or_create_collection("docs")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def chunk_text(text, size=500, overlap=50):
    chunks = []
    start = 0
    while start < len(text):
        end = start + size
        chunks.append(text[start:end])
        start = end - overlap
    return chunks

def ingest_doc(doc_id: str, text: str):
    chunks = chunk_text(text)
    embeddings = embedder.encode(chunks).tolist()
    collection.add(
        ids=[f"{doc_id}_{i}" for i in range(len(chunks))],
        embeddings=embeddings,
        documents=chunks
    )
    return len(chunks)
