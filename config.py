# config.py

MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
#MODEL_NAME = r"D:\Model\clip-ViT-B-32"
IMAGE_DIR  = r"data\images"   # thư mục ảnh đã extract CIFAR100
INDEX_FILE = r"data\index.faiss"
META_FILE  = r"data\meta.json"

EMBED_DIM = 512
NORMALIZE_EMBEDDINGS = True
