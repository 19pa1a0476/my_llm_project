import chromadb
from sentence_transformers import SentenceTransformer
from .model_loader import model, tokenizer
import torch

client = chromadb.PersistentClient(path="data/chroma")
collection = client.get_or_create_collection("docs")

embedder = SentenceTransformer("all-MiniLM-L6-v2")

def get_answer(query, k=3):
    qvec = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[qvec], n_results=k)
    docs = results['documents'][0] if results['documents'] else []
    context = "\n".join(docs)
    prompt = f"Context:\n{context}\n\nQuestion: {query}\nAnswer:"

    inputs = tokenizer(prompt, return_tensors="pt").to(model.device)
    output = model.generate(**inputs, max_new_tokens=256)
    return tokenizer.decode(output[0], skip_special_tokens=True)
