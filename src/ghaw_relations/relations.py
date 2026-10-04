"""Text normalization and similarity measures for GH-AW instruction files."""

import hashlib
import re

import numpy as np
import pandas as pd
import yaml
from sklearn.feature_extraction.text import TfidfVectorizer

MIN_FRAGMENT_WORDS = 8  # fragments shorter than this (headings, one-liners) are ignored
STUB_MAX_WORDS = 20  # bodies shorter than this are stubs (title, placeholder); instructions come from `imports:`


def is_stub(body: str) -> bool:
    return len(body.split()) < STUB_MAX_WORDS


def frontmatter_imports(frontmatter: str) -> tuple[str, ...]:
    """Import targets listed under `imports:` (strings, or the `path`/first key of mapping entries)."""
    try:
        data = yaml.load(frontmatter, Loader=yaml.BaseLoader)
    except yaml.YAMLError:
        return ()
    imports = data.get("imports") if isinstance(data, dict) else None
    if not isinstance(imports, list):
        return ()
    out = []
    for item in imports:
        if isinstance(item, str):
            out.append(item)
        elif isinstance(item, dict) and item:
            out.append(item.get("path") or item.get("uses") or next(iter(item)))
    return tuple(out)


def normalize_text(text: str) -> str:
    """Whitespace-insensitive form used for exact matching: trailing spaces, line endings and blank-line runs."""
    text = text.replace("\r\n", "\n")
    text = "\n".join(line.rstrip() for line in text.split("\n"))
    return re.sub(r"\n{3,}", "\n\n", text).strip()


def text_hash(text: str) -> str:
    return hashlib.sha1(normalize_text(text).encode()).hexdigest()


def fragments(body: str) -> list[str]:
    """Split a Markdown body into paragraph-level fragments (blank-line separated), lowercased and whitespace-collapsed."""
    out = []
    for block in re.split(r"\n\s*\n", body):
        frag = " ".join(block.lower().split())
        if len(frag.split()) >= MIN_FRAGMENT_WORDS:
            out.append(frag)
    return out


def frontmatter_keys(frontmatter: str) -> frozenset[str] | None:
    """Top-level frontmatter keys. BaseLoader keeps `on` as a string instead of YAML 1.1 boolean True."""
    try:
        data = yaml.load(frontmatter, Loader=yaml.BaseLoader)
    except yaml.YAMLError:
        return None
    return frozenset(data) if isinstance(data, dict) else None


def frontmatter_source(frontmatter: str) -> str | None:
    """The `source:` field gh-aw writes when a workflow is added from another repo (e.g. `owner/repo/path.md@ref`)."""
    try:
        data = yaml.load(frontmatter, Loader=yaml.BaseLoader)
    except yaml.YAMLError:
        return None
    return data.get("source") if isinstance(data, dict) else None


def fit_tfidf(bodies: pd.Series) -> TfidfVectorizer:
    """TF-IDF fitted on the distinct bodies, so repeated versions don't skew the idf weights."""
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True)
    return vec.fit(bodies.drop_duplicates())


def body_cosine_matrix(bodies: pd.Series) -> np.ndarray:
    """Pairwise TF-IDF cosine similarity (word 1-2 grams, sublinear tf) between bodies."""
    vec = TfidfVectorizer(ngram_range=(1, 2), sublinear_tf=True, min_df=1)
    tfidf = vec.fit_transform(bodies)
    return (tfidf @ tfidf.T).toarray()


def containment(a: set, b: set) -> float:
    """Share of the smaller fragment set that also appears in the other one."""
    if not a or not b:
        return 0.0
    return len(a & b) / min(len(a), len(b))


def jaccard(a: frozenset | None, b: frozenset | None) -> float:
    if not a or not b:
        return float("nan")
    return len(a & b) / len(a | b)
