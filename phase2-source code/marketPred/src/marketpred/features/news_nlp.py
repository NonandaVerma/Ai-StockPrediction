"""News event-extraction pipeline for user-supplied, licensed article archives.

Status: implemented and unit-tested on synthetic headlines, but NOT used in the evaluated models:
no licensed historical news archive with original publication timestamps is available here.

Pipeline: normalise -> timestamp to UTC (original zone kept) -> near-duplicate clustering
(syndicated copies) -> novelty -> rule-based event typing -> lexicon sentiment -> daily aggregation
with point-in-time availability (an article is usable from its publication time onward).

The bundled lexicons are small illustrative lists. For research use, supply a validated
finance lexicon (e.g. Loughran & McDonald 2011, subject to its licence) via `load_lexicon`.
"""

from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass
from pathlib import Path

import pandas as pd

EVENT_PATTERNS = {
    "monetary_policy": r"\b(rate (hike|cut)|fomc|federal reserve|central bank|monetary policy|basis points)\b",
    "trade_tariffs": r"\b(tariffs?|trade war|export (ban|controls?)|sanctions?|embargo)\b",
    "conflict": r"\b(war|invasion|missile|airstrikes?|military|attack|ceasefire)\b",
    "election_politics": r"\b(election|ballot|parliament|congress|shutdown|impeach\w*|budget)\b",
    "regulation": r"\b(regulator|antitrust|sec charges|fined?|probe|lawsuit|ruling)\b",
    "supply_chain": r"\b(supply chain|shortage|port (closure|strike)|shipping|canal|blockade)\b",
    "earnings_corporate": r"\b(earnings|guidance|profit warning|merger|acquisition|buyback|dividend)\b",
    "disaster": r"\b(hurricane|earthquake|flood|wildfire|cyclone|drought|pandemic)\b",
}
POSITIVE = {"beat", "beats", "surge", "surges", "rally", "gain", "gains", "growth", "upgrade", "record",
            "strong", "eases", "agreement", "deal", "recovery", "optimism"}
NEGATIVE = {"miss", "misses", "plunge", "plunges", "slump", "loss", "losses", "downgrade", "weak", "fears",
            "crisis", "default", "recession", "war", "sanctions", "tariffs", "cut", "cuts", "warning", "probe"}


@dataclass
class Lexicon:
    positive: set[str]
    negative: set[str]


def load_lexicon(path: str | Path) -> Lexicon:
    """CSV with columns word,polarity (polarity in {positive, negative})."""
    df = pd.read_csv(path)
    df["word"] = df["word"].str.lower()
    return Lexicon(set(df.loc[df.polarity == "positive", "word"]), set(df.loc[df.polarity == "negative", "word"]))


def tokens(text: str) -> list[str]:
    return re.findall(r"[a-z]+", text.lower())


def shingles(text: str, k: int = 3) -> set[str]:
    t = tokens(re.sub(r"\s*[-|]\s*(reuters|ap|bloomberg|afp)\s*$", "", text.lower()))
    return {" ".join(t[i:i + k]) for i in range(max(1, len(t) - k + 1))}


def jaccard(a: set[str], b: set[str]) -> float:
    return len(a & b) / len(a | b) if a and b else 0.0


def classify_events(text: str) -> list[str]:
    t = text.lower()
    return [name for name, pat in EVENT_PATTERNS.items() if re.search(pat, t)]


def sentiment(text: str, lex: Lexicon | None = None) -> float:
    lex = lex or Lexicon(POSITIVE, NEGATIVE)
    toks = tokens(text)
    pos = sum(t in lex.positive for t in toks)
    neg = sum(t in lex.negative for t in toks)
    return 0.0 if pos + neg == 0 else (pos - neg) / (pos + neg)


def process_articles(articles: pd.DataFrame, dup_threshold: float = 0.6, novelty_window_days: float = 3.0,
                     lex: Lexicon | None = None) -> pd.DataFrame:
    """Structure raw articles.

    Input columns: source, published (timestamp string or datetime), tz (IANA zone of `published` if naive),
    headline, and optionally body and entities (comma-separated).
    Output adds: published_utc, cluster_id, is_duplicate, novelty, event_types, sentiment.
    """
    df = articles.copy()
    ts = []
    for p, z in zip(df["published"], df.get("tz", pd.Series("UTC", index=df.index))):
        t = pd.Timestamp(p)
        ts.append((t.tz_localize(z) if t.tzinfo is None else t).tz_convert("UTC"))
    df["published_utc"] = ts
    df["original_tz"] = df.get("tz", "UTC")
    df = df.sort_values("published_utc").reset_index(drop=True)
    text = df["headline"].fillna("") + " " + df.get("body", pd.Series("", index=df.index)).fillna("")
    sh = [shingles(h) for h in df["headline"].fillna("")]
    cluster, novelty = [-1] * len(df), [1.0] * len(df)
    for i in range(len(df)):
        best, best_j = 0.0, None
        for j in range(i - 1, -1, -1):
            if (df.at[i, "published_utc"] - df.at[j, "published_utc"]).total_seconds() > novelty_window_days * 86400:
                break
            s = jaccard(sh[i], sh[j])
            if s > best:
                best, best_j = s, j
        novelty[i] = 1.0 - best
        cluster[i] = cluster[best_j] if (best_j is not None and best >= dup_threshold) else i
    df["cluster_id"] = cluster
    df["is_duplicate"] = df["cluster_id"] != df.index
    df["novelty"] = novelty
    df["event_types"] = [classify_events(t) for t in text]
    df["sentiment"] = [sentiment(t, lex) for t in text]
    df["article_id"] = [hashlib.sha1(f"{s}|{h}|{p}".encode()).hexdigest()[:12]
                        for s, h, p in zip(df["source"], df["headline"], df["published_utc"])]
    return df


def daily_event_features(processed: pd.DataFrame, decision_ts: pd.Series) -> pd.DataFrame:
    """Aggregate original (non-duplicate) articles published in (previous decision, this decision]."""
    orig = processed[~processed["is_duplicate"]]
    edges = decision_ts.sort_values()
    bins = pd.cut(orig["published_utc"].astype("int64"), bins=[-(2 ** 62)] + list(edges.astype("int64")),
                  labels=edges.index, right=True)
    g = orig.groupby(bins, observed=False)
    out = pd.DataFrame(index=edges.index)
    out["n_articles"] = g.size().reindex(edges.index).fillna(0)
    out["mean_sentiment"] = g["sentiment"].mean().reindex(edges.index)
    out["mean_novelty"] = g["novelty"].mean().reindex(edges.index)
    for ev in EVENT_PATTERNS:
        out[f"n_{ev}"] = orig.assign(hit=orig["event_types"].apply(lambda e, ev=ev: ev in e)) \
            .groupby(bins, observed=False)["hit"].sum().reindex(edges.index).fillna(0)
    return out
