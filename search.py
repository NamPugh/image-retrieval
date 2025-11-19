# search.py

import json
import faiss
import numpy as np
from PIL import Image
from sentence_transformers import SentenceTransformer

from config import INDEX_FILE, META_FILE, MODEL_NAME, NORMALIZE_EMBEDDINGS


class SearchEngine:
    def __init__(self):
        print("Loading model...")
        self.model = SentenceTransformer(MODEL_NAME)
        print("Model loaded.")

        print("Loading FAISS index...")
        self.index = faiss.read_index(INDEX_FILE)

        print("Loading metadata...")
        with open(META_FILE, "r") as f:
            self.metadata = json.load(f)

        print("Ready!")

    def encode_text(self, text):
        emb = self.model.encode(
            [text],
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype("float32")

        if NORMALIZE_EMBEDDINGS:
            faiss.normalize_L2(emb)

        return emb

    def encode_image(self, img_path):
        img = Image.open(img_path).convert("RGB")
        emb = self.model.encode(
            [img],
            convert_to_numpy=True,
            normalize_embeddings=False
        ).astype("float32")

        if NORMALIZE_EMBEDDINGS:
            faiss.normalize_L2(emb)

        return emb

    def search(self, emb, k=5):
        D, I = self.index.search(emb, k)
        results = []
        for score, idx in zip(D[0], I[0]):
            results.append({
                "path": self.metadata[idx]["path"],
                "score": float(score)
            })
        return results


def demo():
    engine = SearchEngine()

    while True:
        print("\n===== MENU =====")
        print("1. Search by text")
        print("2. Search by image")
        print("0. Exit")
        choice = input("> ")

        if choice == "1":
            q = input("Enter text: ")
            emb = engine.encode_text(q)
            res = engine.search(emb)
            print("\nResults:")
            for r in res:
                print(r)

        elif choice == "2":
            p = input("Image path: ")
            emb = engine.encode_image(p)
            res = engine.search(emb)
            print("\nResults:")
            for r in res:
                print(r)

        elif choice == "0":
            break
        else:
            print("Invalid choice!")

if __name__ == "__main__":
    demo()
