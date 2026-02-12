"""
Treasury Select Committee — topic modelling over oral evidence corpus.

Reads the plain-text transcripts produced by tsc_extract.py and runs
BERTopic to discover latent themes across ~680 sessions (2012-2026).

Outputs:
    tsc_data/topics_by_session.csv   – per-session topic assignments
    tsc_data/topic_labels.csv        – topic id -> top words / label
    tsc_data/topics_over_time.png    – topic prevalence timeline chart

Requirements (install once):
    pip install bertopic[visualization] sentence-transformers pandas matplotlib
"""

import csv
import re
from pathlib import Path

import pandas as pd
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from bertopic import BERTopic

DATA_DIR = Path(__file__).parent / "tsc_data"
TRANSCRIPT_DIR = DATA_DIR / "transcripts"

# ── 1. load corpus ──────────────────────────────────────────────────

def load_corpus() -> pd.DataFrame:
    """Merge metadata with transcript text."""
    meta = pd.read_csv(DATA_DIR / "metadata.csv", dtype=str)
    meta["date"] = pd.to_datetime(meta["date"], errors="coerce")

    docs, dates, inquiries, eids = [], [], [], []
    for _, row in meta.iterrows():
        path = TRANSCRIPT_DIR / row["file"]
        if not path.exists():
            continue
        text = path.read_text(encoding="utf-8")
        if len(text) < 500:
            continue
        docs.append(text)
        dates.append(row["date"])
        inquiries.append(row["inquiry"])
        eids.append(row["evidence_id"])

    df = pd.DataFrame({
        "evidence_id": eids,
        "date": dates,
        "inquiry": inquiries,
        "text": docs,
    })
    df["date"] = pd.to_datetime(df["date"], errors="coerce")
    df = df.dropna(subset=["date"]).sort_values("date").reset_index(drop=True)
    print(f"Loaded {len(df)} transcripts  "
          f"({df['date'].min():%Y-%m} -> {df['date'].max():%Y-%m})")
    return df


# ── 2. fit topic model ──────────────────────────────────────────────

def fit_model(docs: list[str]) -> BERTopic:
    """Fit BERTopic with sensible defaults for long parliamentary text."""
    from sklearn.feature_extraction.text import CountVectorizer

    vectorizer = CountVectorizer(
        stop_words="english",
        ngram_range=(1, 3),
        min_df=3,
        max_df=0.85,
    )

    model = BERTopic(
        language="english",
        min_topic_size=5,
        n_gram_range=(1, 3),
        vectorizer_model=vectorizer,
        verbose=True,
    )
    topics, probs = model.fit_transform(docs)
    info = model.get_topic_info()
    print(f"\nDiscovered {len(info) - 1} topics  "
          f"(+ outlier topic -1 with {(pd.Series(topics) == -1).sum()} docs)\n")
    return model


# ── 3. save outputs ─────────────────────────────────────────────────

def save_results(model: BERTopic, df: pd.DataFrame):
    topics = model.topics_

    # per-session assignments
    df_out = df[["evidence_id", "date", "inquiry"]].copy()
    df_out["topic_id"] = topics
    topic_info = model.get_topic_info()
    label_map = dict(zip(topic_info["Topic"], topic_info["Name"]))
    df_out["topic_label"] = [label_map.get(t, "") for t in topics]
    df_out.to_csv(DATA_DIR / "topics_by_session.csv", index=False)

    # topic dictionary
    rows = []
    for _, row in topic_info.iterrows():
        tid = row["Topic"]
        if tid == -1:
            continue
        words = model.get_topic(tid)
        top_words = ", ".join(w for w, _ in words[:10])
        rows.append({"topic_id": tid, "label": row["Name"],
                      "count": row["Count"], "top_words": top_words})
    pd.DataFrame(rows).to_csv(DATA_DIR / "topic_labels.csv", index=False)

    print(f"Saved topics_by_session.csv  ({len(df_out)} rows)")
    print(f"Saved topic_labels.csv       ({len(rows)} topics)")


# ── 4. timeline chart ───────────────────────────────────────────────

def plot_timeline(model: BERTopic, df: pd.DataFrame):
    """Topic prevalence per year as a stacked area chart."""
    topics = model.topics_
    df_plot = df[["date"]].copy()
    df_plot["topic"] = topics
    df_plot["year"] = df_plot["date"].dt.year

    # keep top 10 topics (by count) + lump rest into "Other"
    topic_info = model.get_topic_info()
    top_topics = (
        topic_info[topic_info["Topic"] != -1]
        .nlargest(10, "Count")["Topic"]
        .tolist()
    )
    label_map = dict(zip(topic_info["Topic"], topic_info["Name"]))

    df_plot["topic_clean"] = df_plot["topic"].apply(
        lambda t: label_map.get(t, "Other") if t in top_topics else "Other"
    )

    pivot = (
        df_plot.groupby(["year", "topic_clean"]).size()
        .unstack(fill_value=0)
    )

    # sort columns by total count descending, but keep "Other" last
    col_order = pivot.drop(columns="Other", errors="ignore").sum().sort_values(ascending=False).index.tolist()
    if "Other" in pivot.columns:
        col_order.append("Other")
    pivot = pivot[col_order]

    fig, ax = plt.subplots(figsize=(14, 6))
    pivot.plot.bar(stacked=True, ax=ax, width=0.85)

    ax.set_xlabel("")
    ax.set_ylabel("Sessions")
    ax.set_title("Treasury Select Committee — Topic prevalence over time")
    ax.legend(bbox_to_anchor=(1.02, 1), loc="upper left", fontsize=8)
    fig.tight_layout()

    out = DATA_DIR / "topics_over_time.png"
    fig.savefig(out, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Saved {out}")


# ── main ─────────────────────────────────────────────────────────────

def main():
    df = load_corpus()
    model = fit_model(df["text"].tolist())
    save_results(model, df)
    plot_timeline(model, df)

    # print topic summary
    print("\n-- Top topics --")
    info = model.get_topic_info()
    for _, row in info.head(16).iterrows():
        if row["Topic"] == -1:
            continue
        words = model.get_topic(row["Topic"])
        top3 = ", ".join(w for w, _ in words[:5])
        print(f"  Topic {row['Topic']:>2}  ({row['Count']:>3} sessions)  {top3}")


if __name__ == "__main__":
    main()
