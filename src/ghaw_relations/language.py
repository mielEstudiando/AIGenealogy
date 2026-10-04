"""Language of instruction bodies, used to keep only repositories the team can read and check."""

import re

import pandas as pd
import py3langid as langid

from ghaw_relations.relations import is_stub

READABLE = ("en", "es")  # languages the collaborators can read for manual validation
MAX_OTHER_SHARE = 0.5  # a repo is excluded when more than this share of its non-stub body words is in other languages


def prose(body: str) -> str:
    """Body without code blocks, inline code, `${{ ... }}` expressions, URLs and HTML tags or comments."""
    body = re.sub(r"(?ms)^(```|~~~).*?^\1", " ", body)
    body = re.sub(r"<!--.*?-->", " ", body, flags=re.S)
    return re.sub(r"\$\{\{.*?\}\}|`[^`]*`|https?://\S+|<[^>]+>", " ", body)


def body_language(body: str) -> str:
    """ISO 639-1 code from py3langid (deterministic, no model download)."""
    return langid.classify(prose(body))[0]


def repo_languages(df: pd.DataFrame) -> pd.DataFrame:
    """Per repository: body words by language over non-stub versions, and whether it is kept.
    Stub bodies (titles, `${{ ... }}`, comments) have no prose to classify, so they are ignored,
    and a repository with only stubs is kept."""
    ns = df[~df.body.map(is_stub)]
    words = (ns.assign(lang=ns.body.map(body_language), words=ns.body.str.split().str.len())
             .pivot_table(index="repo_full_name", columns="lang", values="words", aggfunc="sum", fill_value=0))
    out = pd.DataFrame(index=pd.Index(sorted(df.repo_full_name.unique()), name="repo_full_name"))
    out["nonstub_words"] = words.sum(axis=1)
    readable = words[[c for c in READABLE if c in words]].sum(axis=1)
    out["other_share"] = (1 - readable / out.nonstub_words).round(3)
    out["main_language"] = words.idxmax(axis=1)
    out["languages"] = words.apply(lambda r: "; ".join(f"{k}={v}" for k, v in r[r > 0].sort_values(ascending=False).items()), axis=1)
    out["excluded"] = out.other_share.fillna(0) > MAX_OTHER_SHARE
    return out.reset_index()
