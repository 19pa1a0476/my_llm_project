import chromadb
from sentence_transformers import SentenceTransformer

from api.app.cache import get_cached_answer, make_cache_key, set_cached_answer
from api.model_loader import model, tokenizer

client = chromadb.PersistentClient(path='data/chroma')
collection = client.get_or_create_collection('docs')

embedder = SentenceTransformer('all-MiniLM-L6-v2')
MODEL_NAME = 'mistralai/Mistral-7B-Instruct-v0.2'
MODEL_VERSION = 'v0.2'
PROMPT_VERSION = 'v1'
RETRIEVER_VERSION = 'chroma-minilm-v1'


def get_answer(query: str, k: int = 3) -> dict:
    cache_key = make_cache_key({'query': query, 'top_k': k})
    cached = get_cached_answer(cache_key)
    if cached:
        input_tokens = len(tokenizer(query, add_special_tokens=False)["input_ids"])
        output_tokens = len(tokenizer(cached, add_special_tokens=False)["input_ids"])
        return {
            "answer": cached,
            "cache_hit": True,
            "input_tokens": input_tokens,
            "output_tokens": output_tokens,
            "context_docs": [],
            "model_name": MODEL_NAME,
            "model_version": MODEL_VERSION,
            "prompt_version": PROMPT_VERSION,
            "retriever_version": RETRIEVER_VERSION,
        }

    qvec = embedder.encode([query])[0].tolist()
    results = collection.query(query_embeddings=[qvec], n_results=k)
    docs = results['documents'][0] if results['documents'] else []
    context = '\n'.join(docs)
    prompt = f'Context:\n{context}\n\nQuestion: {query}\nAnswer:'

    inputs = tokenizer(prompt, return_tensors='pt').to(model.device)
    output = model.generate(**inputs, max_new_tokens=256)
    input_tokens = int(inputs["input_ids"].shape[-1])
    output_tokens = int(output.shape[-1] - input_tokens)
    answer = tokenizer.decode(output[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()

    set_cached_answer(cache_key, answer, ttl_seconds=600)
    return {
        "answer": answer,
        "cache_hit": False,
        "input_tokens": input_tokens,
        "output_tokens": max(output_tokens, 0),
        "context_docs": docs,
        "model_name": MODEL_NAME,
        "model_version": MODEL_VERSION,
        "prompt_version": PROMPT_VERSION,
        "retriever_version": RETRIEVER_VERSION,
    }
