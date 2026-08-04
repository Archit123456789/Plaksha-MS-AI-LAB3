"""Embedding fetching for your semantic search system.

Two modes, controlled by the LAB3_EMBEDDING_MODE environment variable:

    "offline" (default) - a hashed bag-of-words vector. Fully implemented
                below already - you don't need to write or even fully
                understand this part, it's provided so you can build and
                test your whole pipeline with no API key and no network.
    "api"     - calls NVIDIA NIM's embeddings endpoint. THIS is the part
                you implement - see get_embedding() below.

Both modes return a plain Python list of floats, so the rest of your code
never needs to know which one is in use.
"""

import hashlib
import math
import os
import re

import requests
from dotenv import load_dotenv

# Load environment variables from .env file
load_dotenv()

EMBEDDING_MODE = os.environ.get("LAB3_EMBEDDING_MODE", "api")  # "api" or "offline"

# Store the API key cleanly
API_KEY = os.environ.get("NVIDIA_API_KEY")
EMBEDDING_MODEL = "nvidia/nv-embedqa-e5-v5"
EMBEDDING_URL = "https://integrate.api.nvidia.com/v1/embeddings"

OFFLINE_DIM = 64
_WORD_RE = re.compile(r"[a-z']+")


# --- Provided for you: don't need to modify this ---------------------------
def _get_embedding_offline(text, dim=OFFLINE_DIM):
    """A deterministic hashed bag-of-words vector - no model, no network.

    Each word hashes into one of `dim` buckets and increments a count there.
    The result is L2-normalized so cosine similarity behaves sensibly.
    Documents that share vocabulary land close together, which is enough to
    test the full search + visualization pipeline offline.
    """
    vec = [0.0] * dim
    words = _WORD_RE.findall(text.lower())
    for word in words:
        bucket = int(hashlib.md5(word.encode("utf-8")).hexdigest(), 16) % dim
        vec[bucket] += 1.0
    norm = math.sqrt(sum(v * v for v in vec))
    if norm > 0:
        vec = [v / norm for v in vec]
    return vec
# -----------------------------------------------------------------------


def _get_embedding_api(text: str, input_type: str = "passage") -> list[float]:
    """Fetches real semantic embeddings from NVIDIA NIM API."""
    if not API_KEY:
        raise ValueError(
            "NVIDIA_API_KEY environment variable not set. "
            "Get a free key at https://build.nvidia.com/settings/api-keys "
            "and add it to your .env file."
        )

    headers = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json",
        "Accept": "application/json",
    }

    payload = {
        "model": EMBEDDING_MODEL,
        "input": [text],              # Input must be wrapped in a list
        "input_type": input_type,      # Asymmetric parameter: "passage" or "query"
        "encoding_format": "float",
    }

    response = requests.post(EMBEDDING_URL, headers=headers, json=payload)

    if response.status_code != 200:
        raise RuntimeError(
            f"Error fetching embedding from NVIDIA NIM API [{response.status_code}]: {response.text}"
        )

    data = response.json()
    embedding = data.get("data", [])[0].get("embedding", [])
    return embedding


def get_embedding(
    text: str, 
    input_type: str = "passage", 
    mode: str = None
) -> list[float]:
    """
    Public entry point for vector extraction.
    
    Parameters:
        text (str): String to embed.
        input_type (str): "passage" (for index documents) or "query" (for searches).
        mode (str): "api" or "offline". Defaults to LAB3_EMBEDDING_MODE env variable if None.
    """
    # Fall back to env variable setting if mode is not explicitly passed
    selected_mode = mode if mode is not None else EMBEDDING_MODE
    
    if selected_mode == "offline":
        return _get_embedding_offline(text)
    elif selected_mode == "api":
        return _get_embedding_api(text, input_type=input_type)
    else:
        raise ValueError(f"Unknown mode '{selected_mode}'. Choose 'api' or 'offline'.")
