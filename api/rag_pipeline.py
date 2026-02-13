import chromadb
from sentence_transformers import SentenceTransformer

from api.app.cache import get_cached_answer, make_cache_key, set_cached_answer
from api.model_loader import model, tokenizer

client = chromadb.PersistentClient(path='data/chroma')
collection = client.get_or_create_collection('docs')

embedder = SentenceTransformer('all-MiniLM-L6-v2')


def get_answer(query: str, k: int = 3) -> tuple[str, bool]:
    cache_key = make_cache_key({'query': query, 'top_k': k})
    cached = get_cached_answer(cache_key)
    if cached:
        return cached, True

    qvec = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[qvec], n_results=k)
    docs = results['documents'][0] if results['documents'] else []
    context = '\n'.join(docs)
    prompt = f'Context:\n{context}\n\nQuestion: {query}\nAnswer:'

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=256)
    answer = tokenizer.decode(output[0], skip_special_tokens=True)

    set_cached_answer(cache_key, answer, ttl_seconds=600)
    return answer, False
