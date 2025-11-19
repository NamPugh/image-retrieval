import os
from sentence_transformers import SentenceTransformer
from PIL import Image
import numpy as np
from config import MODEL_NAME, IMAGE_DIR  # lấy luôn từ config.py

def find_any_image():
    exts = (".png", ".jpg", ".jpeg", ".bmp")
    for root, _, files in os.walk(IMAGE_DIR):
        for f in files:
            if f.lower().endswith(exts):
                return os.path.join(root, f)
    return None

def main():
    print("Loading model...")
    model = SentenceTransformer(MODEL_NAME)
    print("Model loaded!")

    img_path = find_any_image()
    if img_path is None:
        print(f"Không tìm thấy ảnh nào trong thư mục: {IMAGE_DIR}")
        return

    print("Using image:", img_path)

    img = Image.open(img_path).convert("RGB")
    emb = model.encode([img], convert_to_numpy=True)
    print("Embedding shape:", emb.shape)
    print("First 5 values:", emb[0][:5])

if __name__ == "__main__":
    main()
