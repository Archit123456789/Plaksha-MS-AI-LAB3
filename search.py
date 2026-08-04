import json
import os
import numpy as np
import matplotlib.pyplot as plt
from embeddings import get_embedding

def load_documents(documents_path: str = "documents.json") -> list[dict]:
    with open(documents_path, 'r', encoding='utf-8') as f:
        documents = json.load(f)
    return documents

def build_embeddings_matrix(
        documents: list[dict],
        cache_dir: str = "embeddings_cache.json",
        mode: str = "offline"
):
    if os.path.exists(cache_dir):
        with open(cache_dir, 'r', encoding='utf-8') as f:
            embeddings_cache = json.load(f)
    else:
        embeddings_cache = {}

    embeddings_matrix = []
    for doc in documents:
        doc_id = doc['id']
        if doc_id in embeddings_cache:
            embedding = embeddings_cache[doc_id]
        else:
            embedding = get_embedding(doc['text'], mode=mode)
            embeddings_cache[doc_id] = embedding
        embeddings_matrix.append(embedding)

    with open(cache_dir, 'w', encoding='utf-8') as f:
        json.dump(embeddings_cache, f)

    return np.array(embeddings_matrix)

def cosine_similarity(vec_a: np.ndarray, vec_b: np.ndarray) -> np.ndarray:
    dot_product = np.dot(vec_a, vec_b)
    norm_a = np.linalg.norm(vec_a)
    norm_b = np.linalg.norm(vec_b)
    denominator = norm_a * norm_b
    if denominator == 0:
        return 0
    return dot_product / denominator

def search(
    query: str, 
    embedding_matrix: np.ndarray, 
    documents: list[dict], 
    top_k: int = 3, 
    mode: str = "offline"
) -> list[tuple[dict, float]]:
    """
    Executes semantic similarity search for a query against the embedding matrix.
    """
    # 1. Fetch embedding as a list
    raw_query_emb = get_embedding(query, input_type="query", mode=mode)
    
    # 2. Convert to numpy array with float32 dtype
    query_embedding = np.array(raw_query_emb, dtype=np.float32)
    
    # 3. Compute cosine similarities
    similarities = cosine_similarity(embedding_matrix, query_embedding)
    
    # 4. Get top_k indices sorted descending
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 5. Return list of tuples: (document, score)
    return [(documents[idx], float(similarities[idx])) for idx in top_indices]

def run_pca_2d(matrix: np.ndarray) -> np.ndarray:
    #1. Mean-center the data along each feature (column).
    mean_centered = matrix - np.mean(matrix, axis=0)
    #2. Singular Vector Decomposition (SVD) to compute the principal components.
    u, s, vh = np.linalg.svd(mean_centered, full_matrices=False)
    #3. Project the data onto the first two principal components.
    components = vh[:2, :]
    projected_data = np.dot(mean_centered,components.T)
    return projected_data

def visualize_embeddings_2d(embeddings_2d: np.ndarray, documents: list[dict], query_embedding_2d: np.ndarray = None, query_text: str = None):
    coordinates = run_pca_2d(embeddings_2d)
    # 1. Extract all topics and get UNIQUE topics
    topics = [doc["topic"] for doc in documents]
    unique_topics = sorted(list(set(topics)))
    
    plt.figure(figsize=(10, 7))
    
    # 2. Loop over UNIQUE TOPICS (Not individual documents!)
    for topic in unique_topics:
        # Find all indices belonging to this topic
        indices = [i for i, t in enumerate(topics) if t == topic]
        
        # Plot all points for this topic together under ONE color
        plt.scatter(
            coordinates[indices, 0], 
            coordinates[indices, 1], 
            label=topic, 
            alpha=0.85, 
            edgecolors='k', 
            s=90
        )
    
    plt.title("2D Visualization of Document Embeddings", fontsize=14, fontweight='bold')
    plt.xlabel("Principal Component 1")
    plt.ylabel("Principal Component 2")
    plt.legend(title="Topics", bbox_to_anchor=(1.05, 1), loc='upper left')
    plt.grid(True, linestyle="--", alpha=0.4)
    plt.tight_layout()
    plt.show()



