"""Download the BGE-large-en-v1.5 embedding model into models/."""

from pathlib import Path

import torch
from sentence_transformers import SentenceTransformer

MODEL_ID = "BAAI/bge-large-en-v1.5"
ROOT = Path(__file__).resolve().parent
MODEL_DIR = ROOT / "models" / "bge-large-en-v1.5"


def main() -> None:
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print("device:", device)
    if device == "cuda":
        print("gpu:", torch.cuda.get_device_name(0))

    model = SentenceTransformer(MODEL_ID, device=device)
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(str(MODEL_DIR))

    emb = model.encode(["test sentence"], normalize_embeddings=True)
    print("saved:", MODEL_DIR)
    print("embedding shape:", emb.shape)


if __name__ == "__main__":
    main()
