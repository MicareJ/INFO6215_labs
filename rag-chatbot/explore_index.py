import requests
import chromadb

CHROMA_PATH = "./chroma_db"
COLLECTION_NAME = "company_docs"
OLLAMA_URL = "http://localhost:11434/api/embeddings"
EMBED_MODEL = "nomic-embed-text"

def get_embedding(text):
    response = requests.post(OLLAMA_URL, json={
        "model": EMBED_MODEL,
        "prompt": text
    })
    return response.json()["embedding"]

def search(query, n_results=3):
    client = chromadb.PersistentClient(path=CHROMA_PATH)
    collection = client.get_collection(COLLECTION_NAME)
    query_embedding = get_embedding(query)
    results = collection.query(
        query_embeddings=[query_embedding],
        n_results=n_results,
        include=["documents", "metadatas", "distances"]
    )
    return results

queries = [
    "What is the PTO carryover limit?",
    "How do I submit an expense report?",
    "Can I use ChatGPT for work?",
    "Are there restrictions on external AI tools?",
    "What happens if I find a data breach?",
    "How far in advance do I need to request vacation?"
]

for query in queries:
    print(f"\n{'='*60}")
    print(f"QUERY: {query}")
    print(f"{'='*60}")
    results = search(query, n_results=2)

    for i, (doc, meta, dist) in enumerate(zip(
        results["documents"][0],
        results["metadatas"][0],
        results["distances"][0]
    )):
        print(f"\n  Result {i+1} | Source: {meta['source']} | Distance: {dist:.4f}")
        print(f"  {doc[:300]}...")