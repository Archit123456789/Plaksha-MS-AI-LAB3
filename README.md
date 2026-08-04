Here is a complete, polished **`README.md`** file ready to put directly into your repository. It includes project overview, setup instructions, architecture details, instructions for running both modes, and the non-technical PCA explanation you requested.

---

```markdown
# Semantic Search Engine (Lab 3 — Capstone, Part 1)

A lightweight, end-to-end semantic search engine and visualization system built in Python. This project converts short text documents into high-dimensional vector embeddings, performs semantic similarity searches using cosine distance, persists vectors to a local disk cache, and maps the high-dimensional space into 2D using Principal Component Analysis (PCA via SVD).

---

## Table of Contents
- [Features](#-features)
- [Project Structure](#-project-structure)
- [Setup & Installation](#-setup--installation)
- [Configuration: Offline vs. API Mode](#-configuration-offline-vs-api-mode)
- [How to Run](#-how-to-run)
- [Embedding Cache Strategy](#-embedding-cache-strategy)
- [Understanding the PCA Scatter Plot](#-understanding-the-pca-scatter-plot)
- [License](#-license)

---

## 🚀 Features

- **Semantic Vector Retrieval**: Replaces basic keyword matching with meaning-based retrieval using vector embeddings.
- **Asymmetric Vector Processing**: Handles document indexing (`input_type="passage"`) and search queries (`input_type="query"`) asymmetrically to optimize retrieval accuracy.
- **Dual Execution Modes**:
  - **API Mode**: Connects to NVIDIA NIM endpoints using `nvidia/nv-embedqa-e5-v5` for state-of-the-art semantic representations.
  - **Offline Mode**: A fallback mechanism using deterministic hashed bag-of-words vectors that works with zero network connectivity or API keys.
- **Persistent Disk Caching**: Stores generated document embeddings in `embeddings_cache.json` to reduce latency and prevent API rate-limiting.
- **2D Dimensionality Reduction & Plotting**: Projects 1,024-dimensional embeddings to 2D via SVD-based PCA and plots them with clear color coding per topic.

---

##  Project Structure

```text
semantic_search/
├── .env.example                  # Template for environment variables
├── .gitignore                    # Ensures .env and cache files aren't committed
├── documents.json                # Custom corpus (20 documents, 5 distinct topics)
├── embeddings.py                 # Handles API calls & offline vector generation
├── search.py                     # Document loading, caching, cosine similarity, & PCA
├── semantic_search_starter.ipynb # Primary execution notebook
├── requirements.txt              # Required Python dependencies
└── README.md                     # Documentation

```

---

## Setup & Installation

1. **Clone the repository**:
```bash
git clone <YOUR_GITHUB_REPOSITORY_URL>
cd semantic_search

```


2. **Install dependencies**:
```bash
pip install -r requirements.txt

```


3. **Configure environment variables**:
Copy `.env.example` to `.env`:
```bash
cp .env.example .env

```


Open `.env` and paste your NVIDIA API key:
```env
NVIDIA_API_KEY=nvapi-your-actual-key-here

```



---

## Configuration: Offline vs. API Mode

This project supports two operational modes:

| Mode | Key Required? | Description |
| --- | --- | --- |
| **`"offline"`** | ❌ No | Uses local hashing for quick pipeline validation and testing without network requirements. |
| **`"api"`** | ✅ Yes | Uses NVIDIA NIM (`nvidia/nv-embedqa-e5-v5`) for true semantic context. |

You can switch modes in your notebook or script when calling `build_embeddings_matrix` and `search`:

```python
# Switch between "offline" and "api"
MODE = "api"

matrix = build_embeddings_matrix(docs, mode=MODE)
results = search("What is relativity?", matrix, docs, top_k=3, mode=MODE)

```

---

## How to Run

Launch the Jupyter Notebook:

```bash
jupyter notebook semantic_search_starter.ipynb

```

Follow the notebook cells sequentially:

1. **Load Corpus**: Reads `documents.json`.
2. **Build Matrix**: Generates or loads cached vector embeddings.
3. **Execute Search**: Tests queries across topics (e.g., Physics, Cybersecurity, History).
4. **PCA Visualization**: Generates the 2D cluster scatter plot.

---

## Embedding Cache Strategy

To prevent unnecessary API usage, rate-limiting (`429` errors), and slow re-runs, document embeddings are cached on disk in `embeddings_cache.json`.

* **Cache Hit**: On execution, `build_embeddings_matrix()` checks if `embeddings_cache.json` exists and matches the corpus length. If so, vectors are loaded directly from disk in milliseconds.
* **Cache Miss**: If the cache file does not exist or the corpus size has changed, the system fetches fresh embeddings and updates the disk cache file.

---

## Understanding the PCA Scatter Plot

### Explaining the Plot to a Non-Technical Audience

Imagine taking a massive 1,024-dimensional map of words and squishing it down onto a flat piece of paper. That is what **PCA (Principal Component Analysis)** does.

* **Dots**: Each dot on the plot represents a single document from our dataset.
* **Distances**: The space between dots shows how similar the documents are in meaning.
* Dots close together share similar concepts or belong to the same topic.
* Dots far apart represent completely different subject areas.



### Key Insights & Observations

1. **Clear Semantic Clustering in API Mode**:
When using real AI embeddings (`mode="api"`), documents naturally cluster into distinct "neighborhoods" based on subject matter. For example, all **Cybersecurity** entries (phishing, ransomware, encryption) gather on one side, while **Physics** entries (relativity, quantum mechanics) group on another. This proves that the search engine evaluates underlying *meaning*, not just shared words.
2. **Offline vs. Real AI Representations**:
In **Offline Mode**, dots are scattered semi-randomly because the hashed fallback only looks for exact word overlaps. In **API Mode**, dots snap into well-defined, color-coded clusters, visually demonstrating the power of deep learning embeddings in understanding context.

```

```
