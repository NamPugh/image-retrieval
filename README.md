# 🔍 Image Retrieval using CLIP + FAISS + Streamlit

Hệ thống truy vấn ảnh thông minh sử dụng: - **CLIP ViT-B/32
(HuggingFace)** - **FAISS (IndexFlatIP -- cosine similarity)** -
**Dataset: CIFAR-100 (Python version)** - **Streamlit UI**

Dự án cho phép: - **Tìm ảnh bằng văn bản (Text → Image)** - **Tìm ảnh
bằng ảnh truy vấn (Image → Image)**

------------------------------------------------------------------------

# 📂 1. Clone dự án

``` bash
git clone https://github.com/NamPugh/image-retrieval.git
cd image-retrieval
```

------------------------------------------------------------------------

# 🐍 2. Tạo môi trường & cài thư viện

### (Khuyến nghị) tạo virtual environment

``` bash
python -m venv .venv
.\.venv\Scripts\activate
```

### 2.1 Cài PyTorch

``` bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cpu
```

### 2.2 Cài thư viện còn lại

``` bash
pip install -r requirements.txt
```

------------------------------------------------------------------------

# 📥 3. Tải dataset CIFAR-100

Tải từ: https://www.cs.toronto.edu/~kriz/cifar.html
Chọn **CIFAR-100 python version**

------------------------------------------------------------------------

# 🖼️ 4. Extract ảnh

Chỉnh đường dẫn trong file `extract_cifar100.py`:

``` python
CIFAR_DIR = r"C:\path\to\cifar-100-python"
OUT_DIR   = "data/images"
```

Chạy extract:

``` bash
python extract_cifar100.py
```

------------------------------------------------------------------------

# 🧱 5. Build FAISS index

``` bash
python build_index.py
```

------------------------------------------------------------------------

# 🌐 6. Chạy UI

``` bash
streamlit run app.py
```

------------------------------------------------------------------------

# ⚙️ 7. Cấu hình (config.py)

``` python
MODEL_NAME = "sentence-transformers/clip-ViT-B-32"
IMAGE_DIR = "data/images"
INDEX_FILE = "data/index.faiss"
META_FILE  = "data/meta.json"
EMBED_DIM = 512
NORMALIZE_EMBEDDINGS = True
```

------------------------------------------------------------------------

# 📁 8. Cấu trúc thư mục dự án

    image-retrieval/
    │
    ├── app.py                 # UI Streamlit
    ├── build_index.py         # Tạo FAISS index
    ├── search.py              # Hàm tìm kiếm ảnh
    ├── extract_cifar100.py    # Extract CIFAR-100 → ảnh PNG
    ├── config.py              # Config chung
    ├── requirements.txt       # Danh sách thư viện
    ├── README.md              # Hướng dẫn
    ├── .gitignore             # Ignore các file không đẩy lên GitHub
    │
    └── data/
        ├── images/            # 60.000 ảnh sau khi extract (train/test)
        │    ├── train/
        │    │     └── <class_name>/*.png
        │    └── test/
        │          └── <class_name>/*.png
        │
        ├── index.faiss        # FAISS index (tạo tự động)
        └── meta.json          # Metadata ảnh (tạo tự động)

------------------------------------------------------------------------

# 🎉 Hoàn tất!

Ứng dụng đã sẵn sàng để chạy trên mọi máy. Nếu cần viết báo cáo, hướng
dẫn nâng cao, hoặc triển khai lên HuggingFace Spaces --- chỉ cần nói!
