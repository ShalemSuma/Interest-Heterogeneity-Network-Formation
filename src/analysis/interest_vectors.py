"""
Interest Vector Pipeline

1. Build a TF-IDF model from YouTube video titles.
2. Generate per-channel interest vectors (703-dimensional).
3. Map to consolidated 30-dimensional vectors via community mapping.
"""

import os
import re
import csv
import unicodedata
import warnings

import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from ..utils.config import get_data_path

warnings.filterwarnings("ignore")


# ------------------------------------------------------------------
# Text cleaning
# ------------------------------------------------------------------
def _normalize_unicode(text):
    if not isinstance(text, str):
        return ""
    return unicodedata.normalize("NFKD", text).encode("ascii", "ignore").decode("ascii")


def clean_text(text, preserve_unicode=True):
    """Clean and preprocess text data."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r"https?://\S+|www\.\S+", "", text)
    text = re.sub(r"[^\w\s]", " ", text)
    text = re.sub(r"\d+", " ", text)
    text = re.sub(r"\s+", " ", text).strip()
    tokens = [t for t in text.split() if len(t) > 1]
    return " ".join(tokens)


def _read_video_data(file_path):
    """Read video titles from a channel CSV file."""
    try:
        df = pd.read_csv(file_path, encoding="utf-8")
    except Exception:
        try:
            df = pd.read_csv(file_path, encoding="latin1")
        except Exception:
            return ""
    if df is not None and "title" in df.columns:
        return " ".join(df["title"].fillna("").apply(clean_text))
    return ""


# ------------------------------------------------------------------
# TF-IDF pipeline
# ------------------------------------------------------------------
def build_tfidf_model(
    nodes_path=None,
    video_data_path=None,
    max_features=3000,
):
    """
    Build a TF-IDF vectoriser from channel video titles.

    Returns
    -------
    tfidf : TfidfVectorizer (fitted)
    class_vectors : dict mapping modularity_class -> sparse vector
    class_to_index : dict mapping class_id -> int
    modularity_classes : sorted list of class ids
    """
    nodes_path = nodes_path or str(get_data_path("raw") / "YoutubeNodes.csv")
    video_data_path = video_data_path or str(get_data_path("raw") / "video_data")

    nodes_df = pd.read_csv(nodes_path, encoding="utf-8")
    modularity_classes = sorted(nodes_df["modularity_class"].unique())
    class_to_index = {c: i for i, c in enumerate(modularity_classes)}

    # Map channel names -> modularity class
    ch2cls = {}
    for _, row in nodes_df.iterrows():
        name = row["Label"]
        cls = row["modularity_class"]
        ch2cls[name] = cls
        ch2cls[name.replace(" ", "_")] = cls
        norm = _normalize_unicode(name)
        if norm != name:
            ch2cls[norm] = cls
            ch2cls[norm.replace(" ", "_")] = cls

    # Build corpus
    class_corpus = {c: [] for c in modularity_classes}
    csv_files = [f for f in os.listdir(video_data_path) if f.endswith("_videos.csv")]

    for fname in csv_files:
        ch_name = fname[:-11] if fname.endswith("_videos.csv") else None
        if not ch_name:
            continue
        cls = ch2cls.get(ch_name) or ch2cls.get(_normalize_unicode(ch_name)) or modularity_classes[0]
        text = _read_video_data(os.path.join(video_data_path, fname))
        if text and len(text.strip()) > 10:
            class_corpus[cls].append(text)

    all_texts = [t for texts in class_corpus.values() for t in texts]
    if not all_texts:
        raise ValueError("No text data found — check file paths.")

    tfidf = TfidfVectorizer(
        max_features=max_features, min_df=1, max_df=0.95,
        ngram_range=(1, 2), token_pattern=r"(?u)\b\w\w+\b",
    )
    tfidf.fit(all_texts)

    class_vectors = {}
    for cls, texts in class_corpus.items():
        if texts:
            class_vectors[cls] = tfidf.transform([" ".join(texts)])

    return tfidf, class_vectors, class_to_index, modularity_classes


def generate_interest_vectors(
    tfidf,
    class_vectors,
    class_to_index,
    modularity_classes,
    video_data_path=None,
    nodes_path=None,
):
    """
    Generate per-channel interest vectors.

    Returns a DataFrame with columns [channel_name, interest_vector].
    """
    video_data_path = video_data_path or str(get_data_path("raw") / "video_data")
    nodes_path = nodes_path or str(get_data_path("raw") / "YoutubeNodes.csv")

    nodes_df = pd.read_csv(nodes_path, encoding="utf-8")
    ch2cls = {}
    for _, row in nodes_df.iterrows():
        ch2cls[row["Label"]] = row["modularity_class"]
        ch2cls[row["Label"].replace(" ", "_")] = row["modularity_class"]

    num_classes = len(modularity_classes)
    results = []
    csv_files = [f for f in os.listdir(video_data_path) if f.endswith("_videos.csv")]

    for fname in csv_files:
        ch_name = fname[:-11] if fname.endswith("_videos.csv") else None
        if not ch_name:
            continue

        cls = ch2cls.get(ch_name, modularity_classes[0])
        text = _read_video_data(os.path.join(video_data_path, fname))
        vec = np.zeros(num_classes)

        if text and len(text.strip()) > 10:
            try:
                ch_vec = tfidf.transform([text])
                for c_id, c_vec in class_vectors.items():
                    vec[class_to_index[c_id]] = max(0, cosine_similarity(ch_vec, c_vec)[0][0])
            except Exception:
                vec[class_to_index.get(cls, 0)] = 1.0
        else:
            vec[class_to_index.get(cls, 0)] = 1.0

        norm = np.linalg.norm(vec)
        if norm > 0:
            vec /= norm

        results.append({"channel_name": ch_name, "interest_vector": ",".join(f"{v:.10f}" for v in vec)})

    return pd.DataFrame(results)


# ------------------------------------------------------------------
# Dimension reduction  703 -> 30
# ------------------------------------------------------------------
def transform_to_30d(vectors_df, mapping_df):
    """
    Map original interest vectors to consolidated 30-dimensional vectors.

    Parameters
    ----------
    vectors_df : DataFrame with columns [channel_name, interest_vector]
    mapping_df : DataFrame with columns [original_class, new_class]

    Returns
    -------
    DataFrame with columns [channel_name, interest_vector]
    """
    original_to_new = dict(zip(mapping_df["original_class"], mapping_df["new_class"]))
    num_new = len(set(original_to_new.values()))
    orig_classes = sorted(mapping_df["original_class"].unique())

    output = []
    for _, row in vectors_df.iterrows():
        orig_vec = np.array([float(x) for x in row["interest_vector"].split(",")])
        new_vec = np.zeros(num_new)
        for i, val in enumerate(orig_vec):
            if i < len(orig_classes):
                oc = orig_classes[i]
                if oc in original_to_new:
                    new_vec[original_to_new[oc]] += val

        norm = np.linalg.norm(new_vec)
        if norm > 0:
            new_vec /= norm

        output.append({
            "channel_name": row["channel_name"],
            "interest_vector": ",".join(f"{v:.10f}" for v in new_vec),
        })

    return pd.DataFrame(output)
