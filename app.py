import streamlit as st
from PIL import Image
import numpy as np
import faiss

from search import SearchEngine
from config import NORMALIZE_EMBEDDINGS


# ---------- Singleton cho SearchEngine ----------

@st.cache_resource
def get_engine():
    return SearchEngine()


# ---------- UI ----------

st.set_page_config(page_title="Image & Text Retrieval", layout="wide")

st.title("🔍 Image & Text Retrieval với CLIP (HuggingFace + FAISS)")
st.markdown(
    """
    - **Text → Image**: nhập câu mô tả, hệ thống sẽ trả về ảnh phù hợp nhất.  
    - **Image → Image**: upload 1 ảnh, hệ thống tìm ảnh giống nó trong CIFAR-100.  
    """
)

engine = get_engine()

tab_text, tab_image = st.tabs(["🔤 Search by Text", "🖼️ Search by Image"])


# ---------- TAB 1: TEXT QUERY ----------
with tab_text:
    st.subheader("Search by Text")

    query = st.text_input("Nhập câu truy vấn (tiếng Anh càng tốt):", "")
    top_k = st.slider("Số lượng kết quả (top-k):", 1, 20, 5)

    if st.button("Tìm với text", type="primary"):
        if not query.strip():
            st.warning("Hãy nhập câu truy vấn.")
        else:
            with st.spinner("Đang encode text và truy vấn FAISS..."):
                emb = engine.encode_text(query)
                results = engine.search(emb, k=top_k)

            st.markdown("### Kết quả")
            cols = st.columns(min(5, top_k))
            for i, r in enumerate(results):
                col = cols[i % len(cols)]
                with col:
                    st.image(r["path"], caption=f"score={r['score']:.4f}", use_container_width=True)
                    st.caption(r["path"])


# ---------- TAB 2: IMAGE QUERY ----------
with tab_image:
    st.subheader("Search by Image")

    uploaded = st.file_uploader("Upload một ảnh (png/jpg/jpeg):", type=["png", "jpg", "jpeg"])
    top_k_img = st.slider("Số lượng kết quả (top-k):", 1, 20, 5, key="top_k_img")

    if uploaded is not None:
        # Hiển thị ảnh query
        img = Image.open(uploaded).convert("RGB")
        st.image(img, caption="Ảnh query", use_container_width=False)

        if st.button("Tìm ảnh giống", type="primary"):
            with st.spinner("Đang encode image và truy vấn FAISS..."):
                # Encode ảnh trực tiếp bằng model trong engine
                emb = engine.model.encode(
                    [img],
                    convert_to_numpy=True,
                    normalize_embeddings=False
                ).astype("float32")

                if NORMALIZE_EMBEDDINGS:
                    faiss.normalize_L2(emb)

                results = engine.search(emb, k=top_k_img)

            st.markdown("### Kết quả")
            cols = st.columns(min(5, top_k_img))
            for i, r in enumerate(results):
                col = cols[i % len(cols)]
                with col:
                    st.image(r["path"], caption=f"score={r['score']:.4f}", use_container_width=True)
                    st.caption(r["path"])
    else:
        st.info("Hãy upload một ảnh để bắt đầu tìm kiếm.")
