
"""Reusable Plotly visualizations for ReviewPulse.

This module only visualizes prediction results. It never fabricates
predictions, confidence scores, or evaluation metrics.
"""

from __future__ import annotations

from collections import Counter
from collections.abc import Iterable
import re

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go


SENTIMENT_COLORS = {
    "Positive": "#22A06B",
    "Negative": "#EF5350",
    "Neutral": "#94A3B8",
}

EMOTION_COLORS = {
    "Happy": "#22A06B",
    "Love": "#EC4899",
    "Sadness": "#3B82F6",
    "Anger": "#EF4444",
    "Fear": "#F59E0B",
}

DEFAULT_COLOR = "#64748B"

STOPWORDS = {
    "yang", "dan", "di", "ke", "dari", "ini", "itu", "untuk",
    "dengan", "atau", "ada", "saya", "aku", "kami", "kita",
    "nya", "juga", "sudah", "akan", "pada", "dalam", "karena",
    "jadi", "kalau", "tapi", "tidak", "nggak", "ga", "gak",
    "yg", "aja", "sih", "deh", "dong", "banget",
    "the", "and", "is", "are", "of", "to", "in",
}


def _validate_column(df: pd.DataFrame, column: str) -> None:
    if not isinstance(df, pd.DataFrame):
        raise TypeError("Input must be a pandas DataFrame.")

    if column not in df.columns:
        raise ValueError(
            f"Column '{column}' is missing. "
            f"Available columns: {list(df.columns)}"
        )


def _empty_chart(message: str) -> go.Figure:
    fig = go.Figure()

    fig.add_annotation(
        text=message,
        x=0.5,
        y=0.5,
        xref="paper",
        yref="paper",
        showarrow=False,
        font=dict(size=14),
    )

    fig.update_layout(
        template="plotly_white",
        height=340,
        margin=dict(l=30, r=30, t=40, b=30),
        xaxis=dict(visible=False),
        yaxis=dict(visible=False),
    )

    return fig


def sentiment_distribution(
    df: pd.DataFrame,
    column: str = "sentiment",
) -> go.Figure:
    """Plot predicted sentiment distribution."""

    _validate_column(df, column)

    counts = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    counts = counts[counts.ne("")].value_counts()

    if counts.empty:
        return _empty_chart("No sentiment predictions available.")

    plot_df = counts.rename_axis("Sentiment").reset_index(name="Count")

    fig = px.bar(
        plot_df,
        x="Sentiment",
        y="Count",
        color="Sentiment",
        color_discrete_map=SENTIMENT_COLORS,
        text="Count",
        title="Sentiment Distribution",
    )

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        template="plotly_white",
        height=370,
        showlegend=False,
        margin=dict(l=30, r=30, t=65, b=40),
        yaxis_title="Number of Reviews",
        xaxis_title=None,
    )

    return fig


def emotion_distribution(
    df: pd.DataFrame,
    column: str = "emotion",
) -> go.Figure:
    """Plot predicted emotion distribution."""

    _validate_column(df, column)

    counts = (
        df[column]
        .dropna()
        .astype(str)
        .str.strip()
    )
    counts = counts[counts.ne("")].value_counts()

    if counts.empty:
        return _empty_chart("No emotion predictions available.")

    plot_df = counts.rename_axis("Emotion").reset_index(name="Count")

    fig = px.bar(
        plot_df,
        x="Emotion",
        y="Count",
        color="Emotion",
        color_discrete_map=EMOTION_COLORS,
        text="Count",
        title="Emotion Distribution",
    )

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        template="plotly_white",
        height=370,
        showlegend=False,
        margin=dict(l=30, r=30, t=65, b=40),
        yaxis_title="Number of Reviews",
        xaxis_title=None,
    )

    return fig


def sentiment_donut(
    df: pd.DataFrame,
    column: str = "sentiment",
) -> go.Figure:
    """Plot predicted sentiment proportions as a donut chart."""

    _validate_column(df, column)

    counts = df[column].dropna().astype(str).str.strip()
    counts = counts[counts.ne("")].value_counts()

    if counts.empty:
        return _empty_chart("No sentiment data available.")

    labels = counts.index.tolist()

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=counts.values.tolist(),
            hole=0.62,
            marker=dict(
                colors=[
                    SENTIMENT_COLORS.get(label, DEFAULT_COLOR)
                    for label in labels
                ]
            ),
            textinfo="label+percent",
            sort=False,
        )
    )

    fig.update_layout(
        title="Sentiment Breakdown",
        template="plotly_white",
        height=380,
        margin=dict(l=25, r=25, t=65, b=25),
    )

    return fig


def emotion_donut(
    df: pd.DataFrame,
    column: str = "emotion",
) -> go.Figure:
    """Plot predicted emotion proportions as a donut chart."""

    _validate_column(df, column)

    counts = df[column].dropna().astype(str).str.strip()
    counts = counts[counts.ne("")].value_counts()

    if counts.empty:
        return _empty_chart("No emotion data available.")

    labels = counts.index.tolist()

    fig = go.Figure(
        go.Pie(
            labels=labels,
            values=counts.values.tolist(),
            hole=0.62,
            marker=dict(
                colors=[
                    EMOTION_COLORS.get(label, DEFAULT_COLOR)
                    for label in labels
                ]
            ),
            textinfo="label+percent",
            sort=False,
        )
    )

    fig.update_layout(
        title="Emotion Breakdown",
        template="plotly_white",
        height=380,
        margin=dict(l=25, r=25, t=65, b=25),
    )

    return fig


def top_keywords(
    texts: Iterable[str],
    top_n: int = 15,
    min_length: int = 3,
) -> pd.DataFrame:
    """Extract frequent tokens; descriptive counts, not topic modeling."""

    if top_n < 1:
        raise ValueError("top_n must be greater than zero.")

    counter: Counter[str] = Counter()

    for text in texts:
        if pd.isna(text):
            continue

        words = re.findall(r"[a-zA-ZÀ-ÿ]+", str(text).lower())

        counter.update(
            word
            for word in words
            if len(word) >= min_length and word not in STOPWORDS
        )

    return pd.DataFrame(
        counter.most_common(top_n),
        columns=["keyword", "frequency"],
    )


def keyword_chart(
    texts: Iterable[str],
    top_n: int = 15,
) -> go.Figure:
    """Create a horizontal bar chart of frequent review words."""

    keywords = top_keywords(texts, top_n=top_n)

    if keywords.empty:
        return _empty_chart("No keywords available.")

    fig = px.bar(
        keywords.sort_values("frequency"),
        x="frequency",
        y="keyword",
        orientation="h",
        text="frequency",
        title="Most Frequent Review Keywords",
        color_discrete_sequence=["#64748B"],
    )

    fig.update_traces(textposition="outside", cliponaxis=False)
    fig.update_layout(
        template="plotly_white",
        height=max(350, 38 * len(keywords) + 110),
        showlegend=False,
        margin=dict(l=40, r=45, t=65, b=35),
        xaxis_title="Frequency",
        yaxis_title=None,
    )

    return fig


def confidence_distribution(
    df: pd.DataFrame,
    column: str = "sentiment_confidence",
) -> go.Figure:
    """Show prediction confidence only when scores are available.

    Confidence is a model-generated value, not a guarantee of correctness.
    """

    _validate_column(df, column)

    scores = pd.to_numeric(df[column], errors="coerce").dropna()
    scores = scores[scores.between(0, 1)]

    if scores.empty:
        return _empty_chart("Confidence scores are not available.")

    fig = px.histogram(
        x=scores,
        nbins=20,
        title="Prediction Confidence Distribution",
        labels={"x": "Model Confidence"},
        color_discrete_sequence=["#64748B"],
    )

    fig.update_layout(
        template="plotly_white",
        height=350,
        showlegend=False,
        margin=dict(l=30, r=30, t=65, b=40),
        xaxis_range=[0, 1],
        yaxis_title="Number of Predictions",
    )

    return fig
