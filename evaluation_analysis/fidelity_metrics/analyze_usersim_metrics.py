"""
User Simulator Metric Analysis
==============================
Computes 10 metrics comparing UserLM and sysPrompt simulated user turns
to gold WildBench turns, then optionally correlates with downstream agent
WB-Score/WB-Reward from wildbench_pairwise_eval.py.

Metrics (all are distance-from-GT, lower = more faithful):
  Scalar (L1-norm):        utterance_length, avg_word_length, typo_rate
  Distribution (JSD²):     pos, punctuation, capitalization, sentiment, liwc (optional)
  Vector (1-cosine):       sbert
  Classification (F1):     terminal_f1 (dataset-level)

For GT row in the summary table: raw metric values (scalars), 0.0 (distributions/SBERT), 1.0 (F1).

Output:
  summary_table.csv     3-row × N-metric table (userlm | sysprompt | gt)
  per_session.csv       per-session metric values (for correlation)
  correlation.csv       Spearman + Pearson ρ per metric vs WB-score/WB-reward gaps
  barplot.png           Spearman ρ bar chart

Usage:
    python analyze_usersim_metrics.py \\
        --sim_turns  /path/to/wildbench_simulation.jsonl \\
        --output_dir /path/to/output \\
        --device     cuda \\
        --wb_details /path/to/agent-rp2_vs_agent-userlm_details.jsonl  # optional

    # With LIWC (requires a LIWC .dic file):
        --liwc_dict  /path/to/LIWC2015.dic
"""

from __future__ import annotations

import argparse
import json
import logging
import string
from pathlib import Path
from typing import Dict, List, Optional

import numpy as np
import pandas as pd
from scipy.spatial.distance import jensenshannon
from scipy.stats import spearmanr, pearsonr
from sklearn.metrics import f1_score

logging.basicConfig(level=logging.INFO, format="%(asctime)s [%(levelname)s] %(message)s")
logger = logging.getLogger(__name__)

# Fixed vocabularies for distribution metrics
_ALL_POS = ["ADJ", "ADP", "ADV", "AUX", "CCONJ", "DET", "INTJ", "NOUN",
            "NUM", "PART", "PRON", "PROPN", "PUNCT", "SCONJ", "SYM", "VERB", "X"]
_PUNCT_CHARS = list(string.punctuation)
_SENT_LABELS = ["positive", "neutral", "negative"]


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _jsd(p: np.ndarray, q: np.ndarray) -> float:
    """Squared Jensen-Shannon divergence in [0, 1]."""
    p = np.asarray(p, float) + 1e-10
    q = np.asarray(q, float) + 1e-10
    p /= p.sum()
    q /= q.sum()
    return float(jensenshannon(p, q) ** 2)


def _pos_vec(doc) -> np.ndarray:
    counts = {p: 0 for p in _ALL_POS}
    for tok in doc:
        if tok.pos_ in counts:
            counts[tok.pos_] += 1
    vec = np.array([counts[p] for p in _ALL_POS], float)
    s = vec.sum()
    return vec / s if s > 0 else np.ones(len(_ALL_POS)) / len(_ALL_POS)


def _punct_vec(text: str) -> np.ndarray:
    vec = np.array([float(text.count(c)) for c in _PUNCT_CHARS])
    s = vec.sum()
    return vec / s if s > 0 else np.ones(len(_PUNCT_CHARS)) / len(_PUNCT_CHARS)


def _cap_vec(text: str) -> np.ndarray:
    alpha = [c for c in text if c.isalpha()]
    if not alpha:
        return np.array([0.5, 0.5])
    n = len(alpha)
    u = sum(c.isupper() for c in alpha) / n
    return np.array([u, 1.0 - u])


def _sent_vec(scores: Dict[str, float]) -> np.ndarray:
    vec = np.array([scores.get(l, 0.0) for l in _SENT_LABELS])
    s = vec.sum()
    return vec / s if s > 0 else np.ones(3) / 3.0


def _typo_frac(text: str, spell) -> float:
    words = [w.strip(string.punctuation).lower() for w in text.split()]
    words = [w for w in words if w]
    return len(spell.unknown(words)) / len(words) if words else 0.0


def _awl(text: str) -> float:
    words = text.split()
    return sum(len(w) for w in words) / len(words) if words else 0.0


# ---------------------------------------------------------------------------
# Feature extraction (batch, returns per-turn feature DataFrame)
# ---------------------------------------------------------------------------

def extract_features(sessions: List[Dict], device: str, liwc_parse=None, liwc_cats: List[str] = None) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per (session_id, turn_idx) and columns:
      gt, ul, sp               — raw text
      gt_terminal              — bool (last turn in conversation)
      ul_terminal, sp_terminal — bool (from simulation output)
      + all precomputed feature vectors / scalars
    Heavy models (spaCy, SBERT, sentiment) are batch-processed for efficiency.
    """
    rows = []
    for sess in sessions:
        # New schema: a flat `conversation` list of {role, gt, [userlm, userlm_terminal,
        # sysprompt, sysprompt_terminal]}. Sim entries are user turns from the 2nd user
        # onward; the final user entry has gt=None and is the end-of-conversation check.
        sim_user_entries = [
            e for e in sess["conversation"]
            if e.get("role") == "user" and "userlm" in e
        ]
        for k_offset, e in enumerate(sim_user_entries, start=2):
            rows.append({
                "session_id": sess["session_id"],
                "primary_tag": sess.get("primary_tag", ""),
                "turn_idx": k_offset,
                "gt": e.get("gt") or "",
                "ul": e.get("userlm") or "",
                "sp": e.get("sysprompt") or "",
                "ul_terminal": bool(e.get("userlm_terminal", False)),
                "sp_terminal": bool(e.get("sysprompt_terminal", False)),
                "gt_terminal": e.get("gt") is None,
            })
    df = pd.DataFrame(rows)
    n = len(df)
    logger.info("Extracting features for %d turns...", n)

    # --- Scalar features ---
    df["len_gt"] = df["gt"].str.split().str.len()
    df["len_ul"] = df["ul"].str.split().str.len()
    df["len_sp"] = df["sp"].str.split().str.len()
    df["awl_gt"] = df["gt"].apply(_awl)
    df["awl_ul"] = df["ul"].apply(_awl)
    df["awl_sp"] = df["sp"].apply(_awl)

    logger.info("  Typo rate...")
    from spellchecker import SpellChecker
    spell = SpellChecker()
    df["typo_gt"] = df["gt"].apply(lambda t: _typo_frac(t, spell))
    df["typo_ul"] = df["ul"].apply(lambda t: _typo_frac(t, spell))
    df["typo_sp"] = df["sp"].apply(lambda t: _typo_frac(t, spell))

    # --- POS (spaCy batch) ---
    logger.info("  POS distribution (spaCy)...")
    import spacy
    nlp = spacy.load("en_core_web_sm", disable=["ner", "parser"])
    all_texts = list(df["gt"]) + list(df["ul"]) + list(df["sp"])
    safe_texts = [t if t.strip() else "." for t in all_texts]
    all_pos = [_pos_vec(doc) for doc in nlp.pipe(safe_texts, batch_size=128)]
    df["pos_gt"] = all_pos[:n]
    df["pos_ul"] = all_pos[n:2 * n]
    df["pos_sp"] = all_pos[2 * n:]

    # --- SBERT (batch) ---
    logger.info("  SBERT embeddings...")
    from sentence_transformers import SentenceTransformer
    sbert = SentenceTransformer("all-MiniLM-L6-v2", device=device)
    safe_gt = [t if t.strip() else "." for t in df["gt"]]
    safe_ul = [t if t.strip() else "." for t in df["ul"]]
    safe_sp = [t if t.strip() else "." for t in df["sp"]]
    emb_gt = sbert.encode(safe_gt, batch_size=128, show_progress_bar=False)
    emb_ul = sbert.encode(safe_ul, batch_size=128, show_progress_bar=False)
    emb_sp = sbert.encode(safe_sp, batch_size=128, show_progress_bar=False)
    df["emb_gt"] = list(emb_gt)
    df["emb_ul"] = list(emb_ul)
    df["emb_sp"] = list(emb_sp)

    # --- Sentiment (batch) ---
    logger.info("  Sentiment (distilbert)...")
    import torch
    from transformers import pipeline as hf_pipeline
    dev_id = 0 if (device.startswith("cuda") and torch.cuda.is_available()) else -1
    sent_pipe = hf_pipeline(
        "text-classification",
        model="lxyuan/distilbert-base-multilingual-cased-sentiments-student",
        return_all_scores=True,
        device=dev_id,
    )
    all_safe = safe_gt + safe_ul + safe_sp
    batch_results = sent_pipe(all_safe, batch_size=64, truncation=True, max_length=512)
    def _to_dict(res): return {r["label"].lower(): r["score"] for r in res}
    df["sent_gt"] = [_to_dict(batch_results[i]) for i in range(n)]
    df["sent_ul"] = [_to_dict(batch_results[n + i]) for i in range(n)]
    df["sent_sp"] = [_to_dict(batch_results[2 * n + i]) for i in range(n)]

    # --- LIWC (optional) ---
    if liwc_parse is not None and liwc_cats:
        logger.info("  LIWC distribution...")
        nc = len(liwc_cats)
        cat_idx = {c: i for i, c in enumerate(liwc_cats)}

        def _liwc_vec(text: str) -> np.ndarray:
            vec = np.zeros(nc)
            for tok in text.lower().split():
                for cat in liwc_parse(tok):
                    if cat in cat_idx:
                        vec[cat_idx[cat]] += 1
            s = vec.sum()
            return vec / s if s > 0 else np.ones(nc) / nc

        df["liwc_gt"] = df["gt"].apply(_liwc_vec)
        df["liwc_ul"] = df["ul"].apply(_liwc_vec)
        df["liwc_sp"] = df["sp"].apply(_liwc_vec)

    return df


# ---------------------------------------------------------------------------
# Per-conversation metric functions
# Input:  group DataFrame (rows = turns from one session, with precomputed features)
# Output: dict with keys {userlm, sysprompt, gt} → scalar metric value
#   - userlm/sysprompt: distance from GT (lower = more faithful)
#   - gt: raw metric value (scalars), 0.0 (distribution/SBERT)
# ---------------------------------------------------------------------------

def _safe_mean(vals) -> float:
    """Mean of a list/Series, returning NaN when there is nothing to average."""
    arr = np.asarray(list(vals), dtype=float)
    arr = arr[~np.isnan(arr)]
    return float(arr.mean()) if arr.size else float("nan")


def metric_utterance_length(group: pd.DataFrame) -> Dict[str, float]:
    """Raw mean character count for each model (empty utterances excluded)."""
    return {
        "gt": _safe_mean(group.loc[group["gt"] != "", "len_gt"]),
        "userlm": _safe_mean(group.loc[group["ul"] != "", "len_ul"]),
        "sysprompt": _safe_mean(group.loc[group["sp"] != "", "len_sp"]),
    }


def metric_avg_word_length(group: pd.DataFrame) -> Dict[str, float]:
    """Raw mean chars-per-word for each model (empty utterances excluded)."""
    return {
        "gt": _safe_mean(group.loc[group["gt"] != "", "awl_gt"]),
        "userlm": _safe_mean(group.loc[group["ul"] != "", "awl_ul"]),
        "sysprompt": _safe_mean(group.loc[group["sp"] != "", "awl_sp"]),
    }


def metric_typo_rate(group: pd.DataFrame) -> Dict[str, float]:
    """Raw mean misspelled-word fraction for each model (empty utterances excluded)."""
    return {
        "gt": _safe_mean(group.loc[group["gt"] != "", "typo_gt"]),
        "userlm": _safe_mean(group.loc[group["ul"] != "", "typo_ul"]),
        "sysprompt": _safe_mean(group.loc[group["sp"] != "", "typo_sp"]),
    }


def metric_pos_jsd(group: pd.DataFrame) -> Dict[str, float]:
    """Squared JSD of POS tag distributions (skip rows where either side is empty)."""
    ul_jsds = [_jsd(r["pos_gt"], r["pos_ul"]) for _, r in group.iterrows() if r["gt"] and r["ul"]]
    sp_jsds = [_jsd(r["pos_gt"], r["pos_sp"]) for _, r in group.iterrows() if r["gt"] and r["sp"]]
    return {"gt": 0.0, "userlm": _safe_mean(ul_jsds), "sysprompt": _safe_mean(sp_jsds)}


def metric_sbert(group: pd.DataFrame) -> Dict[str, float]:
    """1 - cosine similarity of SBERT embeddings (skip rows where either side is empty)."""
    from numpy.linalg import norm
    ul_dists, sp_dists = [], []
    for _, r in group.iterrows():
        g = r["emb_gt"]
        if r["gt"] and r["ul"]:
            u = r["emb_ul"]
            ul_dists.append(1.0 - float(np.dot(g, u) / (norm(g) * norm(u) + 1e-10)))
        if r["gt"] and r["sp"]:
            s = r["emb_sp"]
            sp_dists.append(1.0 - float(np.dot(g, s) / (norm(g) * norm(s) + 1e-10)))
    return {"gt": 0.0, "userlm": _safe_mean(ul_dists), "sysprompt": _safe_mean(sp_dists)}


def metric_punctuation_jsd(group: pd.DataFrame) -> Dict[str, float]:
    """Squared JSD of punctuation character distributions (skip empty pairs)."""
    ul_jsds = [_jsd(_punct_vec(r["gt"]), _punct_vec(r["ul"])) for _, r in group.iterrows() if r["gt"] and r["ul"]]
    sp_jsds = [_jsd(_punct_vec(r["gt"]), _punct_vec(r["sp"])) for _, r in group.iterrows() if r["gt"] and r["sp"]]
    return {"gt": 0.0, "userlm": _safe_mean(ul_jsds), "sysprompt": _safe_mean(sp_jsds)}


def metric_capitalization_jsd(group: pd.DataFrame) -> Dict[str, float]:
    """Squared JSD of uppercase/lowercase character distributions (skip empty pairs)."""
    ul_jsds = [_jsd(_cap_vec(r["gt"]), _cap_vec(r["ul"])) for _, r in group.iterrows() if r["gt"] and r["ul"]]
    sp_jsds = [_jsd(_cap_vec(r["gt"]), _cap_vec(r["sp"])) for _, r in group.iterrows() if r["gt"] and r["sp"]]
    return {"gt": 0.0, "userlm": _safe_mean(ul_jsds), "sysprompt": _safe_mean(sp_jsds)}


def metric_sentiment_jsd(group: pd.DataFrame) -> Dict[str, float]:
    """Squared JSD of positive/neutral/negative sentiment distributions (skip empty pairs)."""
    ul_jsds = [_jsd(_sent_vec(r["sent_gt"]), _sent_vec(r["sent_ul"])) for _, r in group.iterrows() if r["gt"] and r["ul"]]
    sp_jsds = [_jsd(_sent_vec(r["sent_gt"]), _sent_vec(r["sent_sp"])) for _, r in group.iterrows() if r["gt"] and r["sp"]]
    return {"gt": 0.0, "userlm": _safe_mean(ul_jsds), "sysprompt": _safe_mean(sp_jsds)}


def metric_liwc_jsd(group: pd.DataFrame) -> Optional[Dict[str, float]]:
    """Squared JSD of LIWC category distributions (skip empty pairs). None if LIWC unavailable."""
    if "liwc_gt" not in group.columns:
        return None
    ul_jsds = [_jsd(r["liwc_gt"], r["liwc_ul"]) for _, r in group.iterrows() if r["gt"] and r["ul"]]
    sp_jsds = [_jsd(r["liwc_gt"], r["liwc_sp"]) for _, r in group.iterrows() if r["gt"] and r["sp"]]
    return {"gt": 0.0, "userlm": _safe_mean(ul_jsds), "sysprompt": _safe_mean(sp_jsds)}


def metric_terminal_f1(df: pd.DataFrame) -> Dict[str, float]:
    """
    Dataset-level F1 for predicting whether a turn is the last in its conversation.
    Computed across all turns (not per-session) to ensure stable estimates.
    """
    gt_labels = df["gt_terminal"].astype(int).tolist()
    ul_labels = df["ul_terminal"].astype(int).tolist()
    sp_labels = df["sp_terminal"].astype(int).tolist()
    return {
        "gt": 1.0,
        "userlm": float(f1_score(gt_labels, ul_labels, zero_division=0)),
        "sysprompt": float(f1_score(gt_labels, sp_labels, zero_division=0)),
    }


# ---------------------------------------------------------------------------
# Aggregation: per-session metrics
# ---------------------------------------------------------------------------

METRIC_FUNCS = {
    "utterance_length": metric_utterance_length,
    "avg_word_length": metric_avg_word_length,
    "typo_rate": metric_typo_rate,
    "pos_jsd": metric_pos_jsd,
    "sbert": metric_sbert,
    "punctuation_jsd": metric_punctuation_jsd,
    "capitalization_jsd": metric_capitalization_jsd,
    "sentiment_jsd": metric_sentiment_jsd,
}


def compute_per_session_metrics(features_df: pd.DataFrame) -> pd.DataFrame:
    """
    Returns a DataFrame with one row per session_id and columns:
      <metric>_userlm, <metric>_sysprompt, <metric>_gt
    for each metric in METRIC_FUNCS (plus liwc_jsd if available).
    """
    rows = []
    has_liwc = "liwc_gt" in features_df.columns
    funcs = dict(METRIC_FUNCS)
    if has_liwc:
        funcs["liwc_jsd"] = metric_liwc_jsd

    for sid, group in features_df.groupby("session_id"):
        row = {"session_id": sid, "primary_tag": group["primary_tag"].iloc[0]}
        for metric_name, fn in funcs.items():
            result = fn(group)
            if result is None:
                continue
            row[f"{metric_name}_userlm"] = result["userlm"]
            row[f"{metric_name}_sysprompt"] = result["sysprompt"]
            row[f"{metric_name}_gt"] = result["gt"]
        # Signed differences for scalar metrics (sim - gt; positive = sim > gt)
        for m in ("utterance_length", "avg_word_length", "typo_rate"):
            if f"{m}_gt" in row:
                row[f"{m}_userlm_diff"] = row[f"{m}_userlm"] - row[f"{m}_gt"]
                row[f"{m}_sysprompt_diff"] = row[f"{m}_sysprompt"] - row[f"{m}_gt"]
        rows.append(row)

    return pd.DataFrame(rows)


def build_summary_tables(per_session_df: pd.DataFrame, features_df: pd.DataFrame):
    """
    Returns two DataFrames:

    summary_raw   — 3×N raw values (userlm | sysprompt | gt)
      Scalar metrics (utterance_length, avg_word_length, typo_rate): actual means
      Distribution/SBERT metrics: JSD/cosine-distance from GT (GT row = 0 by definition)
      terminal_f1: F1 score (GT = 1.0)

    summary_dist  — 2×N distances from GT (userlm | sysprompt)
      Scalar metrics: L1-norm
      Distribution/SBERT: JSD/cosine (same values as in summary_raw sim rows)
      terminal_f1: same F1 scores
    """
    # Metric names from raw columns (excludes _diff columns)
    metric_names = sorted(set(
        c.rsplit("_", 1)[0]
        for c in per_session_df.columns
        if c.endswith(("_userlm", "_sysprompt", "_gt"))
        and not c.endswith("_diff")
    ))
    scalar_metrics = {"utterance_length", "avg_word_length", "typo_rate"}

    f1 = metric_terminal_f1(features_df)

    # --- Raw summary (3 rows) ---
    raw_rows = {}
    for model in ("userlm", "sysprompt", "gt"):
        raw_rows[model] = {}
        for m in metric_names:
            col = f"{m}_{model}"
            if col in per_session_df.columns:
                raw_rows[model][m] = float(per_session_df[col].mean())
        raw_rows[model]["terminal_f1"] = f1[model]
    summary_raw = pd.DataFrame(raw_rows).T

    # --- Distances summary (2 rows) ---
    dist_rows = {}
    for model in ("userlm", "sysprompt"):
        dist_rows[model] = {}
        for m in metric_names:
            if m in scalar_metrics:
                col = f"{m}_{model}_diff"
            else:
                col = f"{m}_{model}"
            if col in per_session_df.columns:
                dist_rows[model][m] = float(per_session_df[col].mean())
        dist_rows[model]["terminal_f1"] = f1[model]
    summary_dist = pd.DataFrame(dist_rows).T

    return summary_raw, summary_dist


# ---------------------------------------------------------------------------
# Correlation analysis
# ---------------------------------------------------------------------------

def load_wb_details(path: str) -> pd.DataFrame:
    """
    Load wildbench_pairwise_eval.py details JSONL.
    Returns DataFrame with session_id, wb_score_model1, wb_score_model2, wb_reward.
    """
    rows = []
    with open(path) as f:
        for line in f:
            if not line.strip():
                continue
            rec = json.loads(line)
            rows.append({
                "session_id": rec["session_id"],
                "wb_score_model1": rec.get("wb_score_model1"),
                "wb_score_model2": rec.get("wb_score_model2"),
                "wb_reward": rec.get("final_score_model1"),
            })
    df = pd.DataFrame(rows).dropna(subset=["wb_score_model1", "wb_score_model2", "wb_reward"])
    logger.info("Loaded %d sessions from WB details.", len(df))
    return df


def compute_correlations(per_session_df: pd.DataFrame, wb_df: pd.DataFrame) -> pd.DataFrame:
    """
    For each metric, compute Spearman and Pearson ρ between:
      sim_gap = sysprompt_dist - userlm_dist   (positive = userlm closer to GT)
      wb_score_gap = wb_score_model1 - wb_score_model2
      wb_reward = WB-Reward (model1 perspective)

    Also includes raw correlations for userlm and sysprompt individually.
    """
    merged = per_session_df.merge(wb_df, on="session_id", how="inner")
    logger.info("Merged %d sessions for correlation analysis.", len(merged))
    merged["wb_score_gap"] = merged["wb_score_model1"] - merged["wb_score_model2"]

    metric_names = sorted(set(
        c.rsplit("_", 1)[0]
        for c in per_session_df.columns
        if c.endswith("_userlm")
    ))
    # terminal_f1 is dataset-level, skip per-session correlation
    if "terminal_f1" in metric_names:
        metric_names.remove("terminal_f1")

    scalar_metrics = {"utterance_length", "avg_word_length", "typo_rate"}
    rows = []
    for m in metric_names:
        # Use L1 columns for scalars; direct JSD/cosine for distributions
        ul_col = f"{m}_userlm_diff" if m in scalar_metrics else f"{m}_userlm"
        sp_col = f"{m}_sysprompt_diff" if m in scalar_metrics else f"{m}_sysprompt"
        if ul_col not in merged.columns or sp_col not in merged.columns:
            continue
        merged[f"{m}_sim_gap"] = merged[sp_col] - merged[ul_col]

        for y_col, y_label in [("wb_score_gap", "wb_score_gap"), ("wb_reward", "wb_reward")]:
            for x_col, x_label in [
                (f"{m}_sim_gap", "sim_gap (sp_dist - ul_dist)"),
                (ul_col, "userlm_dist"),
                (sp_col, "sysprompt_dist"),
            ]:
                valid = merged[[x_col, y_col]].dropna()
                if len(valid) < 10:
                    continue
                sp_rho, sp_p = spearmanr(valid[x_col], valid[y_col])
                pe_rho, pe_p = pearsonr(valid[x_col], valid[y_col])
                rows.append({
                    "metric": m,
                    "x_variable": x_label,
                    "y_variable": y_label,
                    "spearman_rho": round(float(sp_rho), 4),
                    "spearman_p": round(float(sp_p), 4),
                    "pearson_rho": round(float(pe_rho), 4),
                    "pearson_p": round(float(pe_p), 4),
                    "n": len(valid),
                })

    return pd.DataFrame(rows).sort_values(["y_variable", "spearman_rho"], ascending=[True, False])


# ---------------------------------------------------------------------------
# Plotting
# ---------------------------------------------------------------------------

def plot_summary_barplot(dist_df: pd.DataFrame, output_path: Path) -> None:
    """Bar chart of distances from GT for userlm and sysprompt."""
    import matplotlib.pyplot as plt

    metrics = list(dist_df.columns)
    x = np.arange(len(metrics))
    width = 0.35

    fig, ax = plt.subplots(figsize=(max(10, len(metrics) * 1.2), 5))
    ax.bar(x - width / 2, [dist_df.loc["userlm", m] for m in metrics], width,
           label="UserLM", color="#4C72B0", alpha=0.85)
    ax.bar(x + width / 2, [dist_df.loc["sysprompt", m] for m in metrics], width,
           label="sysPrompt", color="#DD8452", alpha=0.85)

    ax.set_xticks(x)
    ax.set_xticklabels(metrics, rotation=30, ha="right", fontsize=9)
    ax.set_ylabel("Distance from GT (lower = more faithful)")
    ax.set_title("User Simulator Distance from Ground Truth (per metric)")
    ax.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved summary barplot: %s", output_path)


def plot_correlation_barplot(corr_df: pd.DataFrame, output_path: Path) -> None:
    import matplotlib.pyplot as plt

    sim_gap_df = corr_df[
        (corr_df["x_variable"] == "sim_gap") & (corr_df["y_variable"] == "wb_score_gap")
    ].sort_values("spearman_rho", ascending=True)

    if sim_gap_df.empty:
        return

    fig, ax = plt.subplots(figsize=(7, max(4, len(sim_gap_df) * 0.5)))
    colors = ["#4C72B0" if r >= 0 else "#C44E52" for r in sim_gap_df["spearman_rho"]]
    ax.barh(sim_gap_df["metric"], sim_gap_df["spearman_rho"], color=colors)
    ax.axvline(0, color="black", linewidth=0.8)

    for i, (_, row) in enumerate(sim_gap_df.iterrows()):
        sig = ("***" if row["spearman_p"] < 0.001 else
               "**" if row["spearman_p"] < 0.01 else
               "*" if row["spearman_p"] < 0.05 else "")
        if sig:
            x = row["spearman_rho"]
            ax.text(x + (0.003 if x >= 0 else -0.003), i, sig,
                    va="center", ha="left" if x >= 0 else "right", fontsize=10)

    ax.set_xlabel("Spearman ρ  (sim_gap vs WB-score gap)", fontsize=11)
    ax.set_title(
        "Correlation: Simulator Fidelity Gap vs Agent Performance Gap\n"
        "(positive ρ: sessions where UserLM is more faithful → Agent-UserLM performs better)",
        fontsize=10,
    )
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches="tight")
    plt.close()
    logger.info("Saved correlation barplot: %s", output_path)


# ---------------------------------------------------------------------------
# Main
# ---------------------------------------------------------------------------

def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    parser.add_argument("--sim_turns", required=True,
                        help="Path to wildbench_simulation.jsonl (from generate_sim_turns.py)")
    parser.add_argument("--output_dir", required=True)
    parser.add_argument("--device", default="cpu", help="Device for SBERT and sentiment model (cpu or cuda[:N])")
    parser.add_argument("--wb_details", default=None,
                        help="Path to agent-rp2_vs_agent-userlm_details.jsonl (for correlation analysis)")
    parser.add_argument("--liwc_dict", default=None, help="Path to LIWC .dic file (optional)")
    parser.add_argument("--limit", type=int, default=None, help="Process only first N conversations (for testing)")
    args = parser.parse_args()

    output_dir = Path(args.output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)

    # Load simulation data
    logger.info("Loading sim turns from %s", args.sim_turns)
    sessions = []
    with open(args.sim_turns) as f:
        for line in f:
            if line.strip():
                sessions.append(json.loads(line))
    if args.limit:
        sessions = sessions[:args.limit]
    logger.info("Loaded %d sessions.", len(sessions))

    # Load LIWC if provided
    liwc_parse = None
    liwc_cats = []
    if args.liwc_dict:
        try:
            import liwc
            liwc_parse, liwc_cats_tuple = liwc.load_token_parser(args.liwc_dict)
            liwc_cats = sorted(liwc_cats_tuple)
            logger.info("Loaded LIWC dictionary with %d categories.", len(liwc_cats))
        except Exception as e:
            logger.warning("Failed to load LIWC dictionary: %s — skipping LIWC metric.", e)

    # Extract features
    features_df = extract_features(sessions, args.device, liwc_parse, liwc_cats)

    # Compute per-session metrics
    logger.info("Computing per-session metrics...")
    per_session_df = compute_per_session_metrics(features_df)
    per_session_path = output_dir / "per_session.csv"
    per_session_df.to_csv(per_session_path, index=False)
    logger.info("Saved per-session metrics: %s (%d rows)", per_session_path, len(per_session_df))

    # Build summary tables
    summary_raw, summary_dist = build_summary_tables(per_session_df, features_df)

    raw_path = output_dir / "summary_raw.csv"
    summary_raw.to_csv(raw_path)
    logger.info("Raw values (userlm/sysprompt/gt):\n%s", summary_raw.round(4).to_string())
    logger.info("Saved: %s", raw_path)

    dist_path = output_dir / "summary_distances.csv"
    summary_dist.to_csv(dist_path)
    logger.info("Distances from GT (userlm/sysprompt):\n%s", summary_dist.round(4).to_string())
    logger.info("Saved: %s", dist_path)

    plot_summary_barplot(summary_dist, output_dir / "summary_barplot.png")

    # Correlation analysis (optional)
    if args.wb_details:
        wb_df = load_wb_details(args.wb_details)
        corr_df = compute_correlations(per_session_df, wb_df)
        corr_path = output_dir / "correlation.csv"
        corr_df.to_csv(corr_path, index=False)
        logger.info("Correlation results (sim_gap vs wb_score_gap):\n%s",
                    corr_df[corr_df["x_variable"] == "sim_gap"][
                        ["metric", "y_variable", "spearman_rho", "spearman_p", "n"]
                    ].to_string(index=False))
        logger.info("Saved correlation results: %s", corr_path)
        plot_correlation_barplot(corr_df, output_dir / "correlation_barplot.png")


if __name__ == "__main__":
    main()
