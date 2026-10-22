"""Descriptive counts only. Do not mislabel frequent tokens as verified issues."""
import pandas as pd


def overview(df: pd.DataFrame) -> dict:
    valid = df.loc[df["status"].eq("ok")].copy()
    return {
        "total": int(len(df)),
        "analyzed": int(len(valid)),
        "skipped": int(len(df) - len(valid)),
        "sentiments": valid["sentiment"].value_counts().to_dict(),
        "emotions": valid["emotion"].value_counts().to_dict(),
    }
