# build_index.py

import os
import json
import faiss
import numpy as np
from tqdm import tqdm
from PIL import Image
from sentence_transformers import SentenceTransformer

from config import IMAGE_DIR, INDEX_FILE, META_FILE, MODEL_NAME, EMBED_DIM, NORMALIZE_EMBEDDINGS


def load_model():
    print(f"Loading model from: {MODEL_NAME}")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded!")
    return model


def load_image_paths(folder):
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    paths = []
    for root, _, files in os.walk(folder):
        for f in files:
            if f.lower().endswith(exts):
                paths.append(os.path.join(root, f))
    return paths


def embed_image(model, img_path):
    img = Image.open(img_path).convert("RGB")
    embedding = model.encode(
        [img],
        batch_size=1,
        convert_to_numpy=True,
        normalize_embeddings=False   # normalize sau bằng FAISS
    )
    return embedding.astype("float32")


def main():
    model = load_model()

    print("Loading image paths...")
    image_paths = load_image_paths(IMAGE_DIR)
    print(f"Found {len(image_paths)} images.")

    embeddings = []
    metadata = []

    print("Embedding images...")
    for idx, img_path in enumerate(tqdm(image_paths)):
        emb = embed_image(model, img_path)
        embeddings.append(emb)
        metadata.append({"id": idx, "path": img_path})

    embeddings = np.vstack(embeddings)

    if NORMALIZE_EMBEDDINGS:
        faiss.normalize_L2(embeddings)

    print("Building FAISS index...")
    index = faiss.IndexFlatIP(EMBED_DIM)   # cosine similarity
    index.add(embeddings)

    faiss.write_index(index, INDEX_FILE)
    json.dump(metadata, open(META_FILE, "w"), indent=2)

    print("\n✅ DONE — Index and metadata saved!")
    print("Index:", INDEX_FILE)
    print("Meta :", META_FILE)


if __name__ == "__main__":
    main()
