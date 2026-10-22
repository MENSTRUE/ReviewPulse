from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# Supports: streamlit run app/streamlit_app.py from repository root.
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from reviewpulse.analytics import overview
from reviewpulse.inference import load_models, predict_review, predict_reviews
from reviewpulse.preprocessing import clean_review

st.set_page_config(page_title="ReviewPulse", page_icon="💬", layout="wide")
st.title("ReviewPulse 💬")
st.caption("Analisis sentimen dan emosi ulasan produk berbahasa Indonesia menggunakan model hasil training.")

with st.sidebar:
    st.header("Tentang model")
    st.write("Klasifikasi Sentiment (Positive/Negative) dan Emotion (5 kelas PRDECT-ID).")
    st.info("Prediksi dapat salah pada ulasan baru, sarkasme, dan bahasa informal. Ini bukan diagnosis emosi seseorang.")
    if st.button("Periksa model"):
        try:
            load_models()
            st.success("Kedua model berhasil dimuat")
        except Exception as e:
            st.error(str(e))

single, bulk = st.tabs(["Satu ulasan", "Analisis CSV"])

with single:
    text = st.text_area("Tulis ulasan", placeholder="Barang sesuai foto, pengiriman cepat dan packing aman!", height=135)
    if st.button("Analisis ulasan", type="primary", use_container_width=True):
        if not clean_review(text):
            st.warning("Masukkan teks ulasan terlebih dahulu.")
        else:
            try:
                result = predict_review(text)
                c1, c2 = st.columns(2)
                with c1:
                    st.metric("Sentiment", result["sentiment"])
                    confidence = result["sentiment_confidence"]
                    st.caption(f"Probabilitas prediksi: {confidence:.1%}" if confidence is not None else "Model tidak menyediakan probabilitas terkalibrasi")
                with c2:
                    st.metric("Emotion", result["emotion"])
                    confidence = result["emotion_confidence"]
                    st.caption(f"Probabilitas prediksi: {confidence:.1%}" if confidence is not None else "Model tidak menyediakan probabilitas terkalibrasi")
                st.caption("Label adalah hasil prediksi model, bukan fakta terverifikasi.")
            except Exception as exc:
                st.error(f"Inferensi gagal: {exc}")

with bulk:
    uploaded = st.file_uploader("Upload CSV ulasan", type=["csv"])
    if uploaded:
        try:
            table = pd.read_csv(uploaded, encoding="utf-8-sig", low_memory=False)
            if table.empty:
                st.warning("CSV kosong.")
            else:
                st.caption(f"{len(table):,} baris, {len(table.columns)} kolom")
                default = next((i for i,c in enumerate(table.columns) if c.lower().strip() in {"customer review","review","review_text","ulasan","text"}), 0)
                col = st.selectbox("Kolom teks ulasan", list(table.columns), index=default)
                max_rows = st.number_input("Maksimum baris untuk analisis", min_value=1, max_value=100000, value=min(1000,len(table)), step=100)
                if st.button("Analisis CSV", type="primary"):
                    selected = table.head(int(max_rows)).copy()
                    with st.spinner("Menjalankan kedua model..."):
                        preds = predict_reviews(selected[col].tolist())
                    preds = preds.rename(columns={"review": "review_cleaned"}).reset_index(drop=True)
                    result = pd.concat([selected.reset_index(drop=True), preds], axis=1)
                    summary = overview(preds)
                    a,b,c = st.columns(3)
                    a.metric("Total baris",summary["total"])
                    b.metric("Diproses",summary["analyzed"])
                    c.metric("Dilewati",summary["skipped"])
                    c1,c2 = st.columns(2)
                    with c1:
                        st.subheader("Distribusi sentiment")
                        st.bar_chart(pd.Series(summary["sentiments"], name="Jumlah"))
                    with c2:
                        st.subheader("Distribusi emotion")
                        st.bar_chart(pd.Series(summary["emotions"], name="Jumlah"))
                    st.dataframe(result, use_container_width=True, hide_index=True)
                    st.download_button("Unduh hasil CSV", data=result.to_csv(index=False).encode("utf-8-sig"),
                                       file_name="reviewpulse_predictions.csv", mime="text/csv")
        except Exception as exc:
            st.error(f"Gagal memproses CSV: {exc}")
