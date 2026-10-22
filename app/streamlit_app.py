
from pathlib import Path
import sys

import pandas as pd
import streamlit as st

# ============================================================
# ReviewPulse — Streamlit Application
# ============================================================

ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"

if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))

from reviewpulse.analytics import overview
from reviewpulse.inference import (
    load_models,
    predict_review,
    predict_reviews,
)
from reviewpulse.preprocessing import clean_review
from reviewpulse.visualization import (
    sentiment_distribution,
    emotion_distribution,
    sentiment_donut,
    emotion_donut,
    keyword_chart,
)

st.set_page_config(
    page_title="ReviewPulse | Review Intelligence",
    page_icon="💬",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ============================================================
# Style
# ============================================================

st.markdown(
    """
    <style>
    .block-container {
        padding-top: 1.7rem;
        padding-bottom: 3rem;
        max-width: 1350px;
    }

    h1, h2, h3 {
        letter-spacing: -0.025em;
    }

    div[data-testid="stMetric"] {
        border: 1px solid rgba(128, 128, 128, 0.20);
        border-radius: 14px;
        padding: 18px;
    }

    div[data-testid="stMetricLabel"] {
        font-weight: 500;
    }

    div.stButton > button {
        border-radius: 10px;
    }

    div[data-testid="stFileUploader"] {
        border-radius: 12px;
    }
    </style>
    """,
    unsafe_allow_html=True,
)


# ============================================================
# Helpers
# ============================================================

def render_chart(fig):
    """Render a Plotly figure with responsive width."""
    st.plotly_chart(
        fig,
        use_container_width=True,
        config={"displayModeBar": False},
    )


def render_confidence(value, label):
    """Display model confidence without inventing probabilities."""
    if value is None or pd.isna(value):
        st.caption(
            f"{label}: model tidak menyediakan probabilitas "
            "prediksi yang dapat ditampilkan."
        )
        return

    try:
        score = float(value)
    except (TypeError, ValueError):
        st.caption(f"{label}: nilai confidence tidak valid.")
        return

    if 0 <= score <= 1:
        st.caption(
            f"{label}: {score:.1%} "
            "(skor model, bukan jaminan kebenaran)."
        )
    else:
        st.caption(f"{label}: confidence di luar rentang 0–1.")


def predictions_frame(preds):
    """Accept the DataFrame contract returned by predict_reviews."""
    if not isinstance(preds, pd.DataFrame):
        raise TypeError(
            "predict_reviews() harus mengembalikan pandas DataFrame."
        )

    required = {"sentiment", "emotion"}
    missing = required - set(preds.columns)

    if missing:
        raise ValueError(
            f"Kolom prediksi tidak ditemukan: {sorted(missing)}"
        )

    return preds.copy().reset_index(drop=True)


def render_distributions(preds):
    """Show prediction distributions only when data exists."""
    valid = preds.copy()

    if valid.empty:
        st.info("Belum ada prediksi yang dapat divisualisasikan.")
        return

    st.subheader("Ringkasan prediksi")

    left, right = st.columns(2)

    with left:
        render_chart(
            sentiment_donut(valid, column="sentiment")
        )

    with right:
        render_chart(
            emotion_donut(valid, column="emotion")
        )

    left, right = st.columns(2)

    with left:
        render_chart(
            sentiment_distribution(
                valid,
                column="sentiment",
            )
        )

    with right:
        render_chart(
            emotion_distribution(
                valid,
                column="emotion",
            )
        )


# ============================================================
# Sidebar
# ============================================================

with st.sidebar:
    st.title("ReviewPulse")
    st.caption("Indonesian E-commerce Review Intelligence")

    st.divider()

    st.subheader("Model")

    st.write(
        "**Sentiment:** Positive / Negative"
    )
    st.write(
        "**Emotion:** Happy, Love, Sadness, Anger, Fear"
    )

    if st.button(
        "Periksa model",
        use_container_width=True,
    ):
        try:
            load_models()
            st.success("Kedua model berhasil dimuat.")
        except Exception as exc:
            st.error(f"Gagal memuat model: {exc}")

    st.divider()

    st.subheader("Tentang")

    st.write(
        "ReviewPulse membantu seller menganalisis "
        "ulasan pelanggan dengan Machine Learning."
    )

    st.info(
        "Prediksi dapat salah pada sarkasme, bahasa informal, "
        "dan ulasan di luar distribusi data training. "
        "Label emosi bukan diagnosis psikologis."
    )

    st.caption("Portfolio Project — Day 15/100")


# ============================================================
# Header
# ============================================================

st.title("ReviewPulse 💬")

st.write(
    "Analisis sentimen dan emosi ulasan pelanggan "
    "untuk membantu memahami pengalaman berbelanja."
)

st.caption(
    "Prediksi berasal dari model Machine Learning hasil training. "
    "Analisis kata kunci bersifat deskriptif, bukan diagnosis "
    "otomatis akar masalah."
)

st.divider()


# ============================================================
# Navigation
# ============================================================

tab_single, tab_bulk, tab_about = st.tabs(
    [
        "💬 Analisis satu ulasan",
        "📊 Analisis CSV",
        "ℹ️ Metodologi",
    ]
)


# ============================================================
# Single Review
# ============================================================

with tab_single:

    st.header("Analisis satu ulasan")

    st.write(
        "Masukkan ulasan produk berbahasa Indonesia "
        "untuk mengetahui prediksi sentiment dan emotion."
    )

    review_text = st.text_area(
        "Ulasan pelanggan",
        placeholder=(
            "Contoh: Barang sesuai foto, pengiriman cepat, "
            "dan kualitasnya bagus sekali!"
        ),
        height=155,
        key="single_review_text",
    )

    if st.button(
        "Analisis ulasan",
        type="primary",
        use_container_width=True,
        key="single_analyze",
    ):

        if not clean_review(review_text):
            st.warning("Masukkan teks ulasan terlebih dahulu.")

        else:
            try:
                with st.spinner("Menganalisis ulasan..."):
                    result = predict_review(review_text)

                sentiment = result.get("sentiment")
                emotion = result.get("emotion")

                sentiment_confidence = result.get(
                    "sentiment_confidence"
                )
                emotion_confidence = result.get(
                    "emotion_confidence"
                )

                st.success("Analisis selesai.")

                col1, col2 = st.columns(2)

                with col1:
                    st.metric(
                        "Prediksi Sentiment",
                        sentiment or "Tidak tersedia",
                    )
                    render_confidence(
                        sentiment_confidence,
                        "Confidence sentiment",
                    )

                with col2:
                    st.metric(
                        "Prediksi Emotion",
                        emotion or "Tidak tersedia",
                    )
                    render_confidence(
                        emotion_confidence,
                        "Confidence emotion",
                    )

                st.subheader("Detail ulasan")

                st.write(review_text)

                st.caption(
                    "Hasil merupakan prediksi model dan "
                    "tidak selalu sesuai dengan maksud pelanggan."
                )

            except Exception as exc:
                st.error(f"Inferensi gagal: {exc}")


# ============================================================
# Bulk CSV Analysis
# ============================================================

with tab_bulk:

    st.header("Analisis banyak ulasan")

    st.write(
        "Unggah file CSV berisi ulasan pelanggan. "
        "Pilih kolom teks, jalankan klasifikasi, "
        "dan unduh hasil analisis."
    )

    uploaded = st.file_uploader(
        "Upload CSV",
        type=["csv"],
        key="review_csv",
    )

    if uploaded is not None:

        try:
            table = pd.read_csv(
                uploaded,
                encoding="utf-8-sig",
                low_memory=False,
            )

            if table.empty:
                st.warning("File CSV tidak berisi baris data.")

            else:

                st.success("CSV berhasil dibaca.")

                c1, c2 = st.columns(2)

                c1.metric(
                    "Jumlah baris",
                    f"{len(table):,}",
                )

                c2.metric(
                    "Jumlah kolom",
                    len(table.columns),
                )

                with st.expander(
                    "Preview dataset",
                    expanded=False,
                ):
                    st.dataframe(
                        table.head(20),
                        use_container_width=True,
                        hide_index=True,
                    )

                possible_columns = {
                    "customer review",
                    "review",
                    "review_text",
                    "ulasan",
                    "text",
                    "komentar",
                }

                default_index = next(
                    (
                        index
                        for index, column_name
                        in enumerate(table.columns)
                        if str(column_name).lower().strip()
                        in possible_columns
                    ),
                    0,
                )

                review_column = st.selectbox(
                    "Pilih kolom ulasan",
                    options=list(table.columns),
                    index=default_index,
                )

                max_rows = st.number_input(
                    "Maksimum baris yang dianalisis",
                    min_value=1,
                    max_value=min(100000, len(table)),
                    value=min(1000, len(table)),
                    step=1,
                )

                if st.button(
                    "Mulai analisis CSV",
                    type="primary",
                    use_container_width=True,
                    key="bulk_analyze",
                ):

                    selected = (
                        table.head(int(max_rows))
                        .copy()
                        .reset_index(drop=True)
                    )

                    with st.spinner(
                        "Menjalankan model sentiment dan emotion..."
                    ):
                        predictions = predict_reviews(
                            selected[review_column].tolist()
                        )

                    predictions = predictions_frame(predictions)

                    if len(predictions) != len(selected):
                        raise ValueError(
                            "Jumlah hasil prediksi berbeda "
                            "dari jumlah baris input."
                        )

                    # Prevent duplicate column names between
                    # the uploaded CSV and prediction output.
                    renamed = {
                        col: f"pred_{col}"
                        for col in predictions.columns
                        if col in selected.columns
                    }

                    export_predictions = predictions.rename(
                        columns=renamed
                    )

                    result_df = pd.concat(
                        [
                            selected,
                            export_predictions,
                        ],
                        axis=1,
                    )

                    summary = overview(predictions)

                    st.success(
                        "Analisis CSV berhasil diselesaikan."
                    )

                    st.subheader("Statistik analisis")

                    m1, m2, m3 = st.columns(3)

                    m1.metric(
                        "Total baris",
                        summary["total"],
                    )

                    m2.metric(
                        "Diproses",
                        summary["analyzed"],
                    )

                    m3.metric(
                        "Dilewati",
                        summary["skipped"],
                    )

                    # Visualization functions use the actual
                    # inference output before export renaming.
                    render_distributions(predictions)

                    st.subheader("Kata yang sering muncul")

                    cleaned_texts = selected[
                        review_column
                    ].fillna("").astype(str)

                    render_chart(
                        keyword_chart(
                            cleaned_texts,
                            top_n=15,
                        )
                    )

                    st.subheader("Hasil klasifikasi")

                    st.dataframe(
                        result_df,
                        use_container_width=True,
                        hide_index=True,
                    )

                    st.download_button(
                        label="Unduh hasil CSV",
                        data=result_df.to_csv(
                            index=False
                        ).encode("utf-8-sig"),
                        file_name="reviewpulse_predictions.csv",
                        mime="text/csv",
                        use_container_width=True,
                    )

                    st.caption(
                        "Grafik menampilkan distribusi prediksi "
                        "model. Kata kunci dihitung berdasarkan "
                        "frekuensi kemunculan dalam teks input."
                    )

        except UnicodeDecodeError:
            st.error(
                "Encoding CSV tidak terbaca sebagai UTF-8. "
                "Simpan ulang file sebagai UTF-8 lalu upload lagi."
            )

        except Exception as exc:
            st.error(f"Gagal memproses CSV: {exc}")


# ============================================================
# Methodology
# ============================================================

with tab_about:

    st.header("Metodologi ReviewPulse")

    st.write(
        "ReviewPulse menggunakan model klasifikasi teks "
        "yang dilatih untuk mengenali sentimen serta "
        "kategori emosi ulasan produk berbahasa Indonesia."
    )

    st.subheader("Dataset penelitian")

    st.write(
        "PRDECT-ID — Product Reviews Dataset "
        "for Emotions Classification Tasks, Indonesian."
    )

    st.write(
        "Dataset penelitian yang digunakan terdiri dari "
        "5.400 ulasan dengan label sentiment dan emotion."
    )

    st.subheader("Hasil evaluasi penelitian")

    metric1, metric2 = st.columns(2)

    metric1.metric(
        "Sentiment Macro-F1",
        "95.07%",
    )

    metric2.metric(
        "Emotion Macro-F1",
        "58.86%",
    )

    st.caption(
        "Angka di atas berasal dari pengujian internal "
        "notebook penelitian pada 811 ulasan test. "
        "Bukan hasil evaluasi setiap CSV pengguna."
    )

    st.subheader("Batasan")

    st.warning(
        "Hasil penelitian belum membuktikan akurasi "
        "yang sama pada seluruh marketplace, "
        "domain produk, atau bahasa informal. "
        "Emotion classification memiliki performa "
        "lebih rendah dibanding sentiment classification."
    )

    st.write(
        "Model yang ditampilkan aplikasi harus berasal "
        "dari artifact training yang sama dengan "
        "laporan evaluasi penelitian."
    )
