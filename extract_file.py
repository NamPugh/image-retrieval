import os
import pickle
import numpy as np
from PIL import Image

# ==== CHỈNH ĐÚNG ĐƯỜNG DẪN NÀY ====
CIFAR_DIR = r"D:\Downloads\cifar-100-python\cifar-100-python"
OUT_DIR   = r"D:\Code\image_retrieval\data\images"
# ==================================

def load_pickle(path):
    with open(path, "rb") as f:
        return pickle.load(f, encoding="latin1")

def save_split(images, labels, label_names, split_name):
    count = images.shape[0]
    print(f"[{split_name}] Saving {count} images...")

    for i in range(count):
        img_flat = images[i]  # (3072,)
        label_id = labels[i]
        label_name = label_names[label_id]

        # 3072 -> (3,32,32) -> (32,32,3)
        img = img_flat.reshape(3, 32, 32).transpose(1, 2, 0)
        img = Image.fromarray(img)

        out_dir = os.path.join(OUT_DIR, split_name, label_name)
        os.makedirs(out_dir, exist_ok=True)

        img.save(os.path.join(out_dir, f"{i}.png"))

        if (i + 1) % 1000 == 0:
            print(f"  Saved {i + 1}/{count}")

def main():
    os.makedirs(OUT_DIR, exist_ok=True)

    print("Loading meta...")
    meta = load_pickle(os.path.join(CIFAR_DIR, "meta"))
    label_names = meta["fine_label_names"]
    print("Classes:", len(label_names))

    print("\nExtracting TRAIN images...")
    train = load_pickle(os.path.join(CIFAR_DIR, "train"))
    save_split(train["data"], train["fine_labels"], label_names, "train")

    print("\nExtracting TEST images...")
    test = load_pickle(os.path.join(CIFAR_DIR, "test"))
    save_split(test["data"], test["fine_labels"], label_names, "test")

    print("\n✅ DONE — All images extracted to:")
    print(OUT_DIR)

if __name__ == "__main__":
    main()
