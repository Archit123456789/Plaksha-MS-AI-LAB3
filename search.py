import json
import os
import numpy as np
import matplotlib.pyplot as plt
from embeddings import get_embedding


def load_documents(documents_path: str = "documents.json") -> list[dict]:
    """Reads and parses the documents JSON file."""
    with open(documents_path, 'r', encoding='utf-8') as f:
        return json.load(f)


def build_embeddings_matrix(
    documents: list[dict],
    cache_path: str = "embeddings_cache.json",
    mode: str = "api"
) -> np.ndarray:
    """
    Fetches embeddings for all documents and stacks them into an (N x D) matrix.
    Uses a disk cache that automatically invalidates if mode or corpus size changes.
    """
    embeddings_cache = {}
    
    # 1. Load existing cache if present
    if os.path.exists(cache_path):
        try:
            with open(cache_path, 'r', encoding='utf-8') as f:
                embeddings_cache = json.load(f)
        except Exception:
            embeddings_cache = {}

    embeddings_matrix = []
    cache_updated = False

    for doc in documents:
        doc_id = doc['id']
        # Use a composite key (doc_id + mode) to avoid dimension mismatch when switching modes
        cache_key = f"{doc_id}_{mode}"
        
        if cache_key in embeddings_cache:
            embedding = embeddings_cache[cache_key]
        else:
            # Crucial: Specify input_type="passage" for indexing documents
            embedding = get_embedding(doc['text'], input_type="passage", mode=mode)
            embeddings_cache[cache_key] = embedding
            cache_updated = True
            
        embeddings_matrix.append(embedding)

    # 2. Persist updated cache to disk
    if cache_updated:
        with open(cache_path, 'w', encoding='utf-8') as f:
            json.dump(embeddings_cache, f, indent=2)

    return np.array(embeddings_matrix, dtype=np.float32)


def cosine_similarity(matrix: np.ndarray, vector: np.ndarray) -> np.ndarray:
    """
    Computes cosine similarity between an (N x D) matrix and a (D,) vector.
    Formula: sim(A, B) = (A · B) / (||A|| * ||B||)
    """
    # Dot product between each row of matrix and vector -> shape (N,)
    dot_products = np.dot(matrix, vector)
    
    # Row-wise norms of matrix -> shape (N,)
    matrix_norms = np.linalg.norm(matrix, axis=1)
    
    # Norm of query vector -> scalar
    vector_norm = np.linalg.norm(vector)
    
    denominator = matrix_norms * vector_norm
    # Avoid zero division
    denominator = np.where(denominator == 0, 1e-10, denominator)
    
    return dot_products / denominator


def search(
    query: str, 
    embedding_matrix: np.ndarray, 
    documents: list[dict], 
    top_k: int = 3, 
    mode: str = "api"
) -> list[tuple[dict, float]]:
    """
    Executes semantic similarity search for a query against the embedding matrix.
    """
    # 1. Embed query using input_type="query"
    raw_query_emb = get_embedding(query, input_type="query", mode=mode)
    query_embedding = np.array(raw_query_emb, dtype=np.float32)
    
    # 2. Compute similarities
    similarities = cosine_similarity(embedding_matrix, query_embedding)
    
    # 3. Sort indices descending
    top_indices = np.argsort(similarities)[::-1][:top_k]
    
    # 4. Return top-k tuples of (document_dict, similarity_score)
    return [(documents[idx], float(similarities[idx])) for idx in top_indices]


def run_pca_2d(matrix: np.ndarray) -> np.ndarray:
    """
    Reduces an (N x D) embedding matrix into an (N x 2) 2D projection via SVD.
    """
    # 1. Mean-center the data along feature columns
    mean_centered = matrix - np.mean(matrix, axis=0)
    
    # 2. Singular Value Decomposition (SVD)
    u, s, vh = np.linalg.svd(mean_centered, full_matrices=False)
    
    # 3. Project onto top 2 principal components
    components = vh[:2, :]
    return np.dot(mean_centered, components.T)


def visualize_embeddings_2d(embedding_matrix: np.ndarray, documents: list[dict]):
    """
    Renders a 2D PCA scatter plot of document embeddings colored by topic.
    """
    # Compute 2D coordinates from the high-dimensional matrix
    coordinates = run_pca_2d(embedding_matrix)
    
    topics = [doc["topic"] for doc in documents]
    unique_topics = sorted(list(set(topics)))
    
    plt.figure(figsize=(10, 7))
    
    # Group points by topic for a clean 1-color-per-topic plot
    for topic in unique_topics:
        indices = [i for i, t in enumerate(topics) if t == topic]
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