"""
Comprehensive evaluation metrics for all tasks (T1–T5).
"""

from __future__ import annotations

from dataclasses import dataclass, field

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    confusion_matrix,
    f1_score,
    top_k_accuracy_score,
)


@dataclass
class ClassificationMetrics:
    """Container for classification metrics."""

    accuracy: float = 0.0
    balanced_accuracy: float = 0.0
    macro_f1: float = 0.0
    top_3_accuracy: float = 0.0
    top_5_accuracy: float = 0.0
    confusion: np.ndarray = field(default_factory=lambda: np.array([]))
    per_class_accuracy: dict[str, float] = field(default_factory=dict)


@dataclass
class RetrievalMetrics:
    """Container for retrieval metrics."""

    recall_at_1: float = 0.0
    recall_at_5: float = 0.0
    recall_at_10: float = 0.0
    mrr: float = 0.0
    median_rank: float = 0.0


@dataclass
class GenerationMetrics:
    """Container for generation metrics."""

    wer: float = 0.0
    cer: float = 0.0
    semantic_similarity: float = 0.0
    bert_score_f1: float = 0.0


def compute_classification_metrics(
    y_true: np.ndarray,
    y_pred: np.ndarray,
    y_prob: np.ndarray | None = None,
    n_classes: int | None = None,
    label_names: list[str] | None = None,
) -> ClassificationMetrics:
    """Compute all classification metrics.

    Parameters
    ----------
    y_true : np.ndarray
        True labels, shape (N,).
    y_pred : np.ndarray
        Predicted labels, shape (N,).
    y_prob : np.ndarray, optional
        Predicted probabilities, shape (N, C). For top-k accuracy.
    n_classes : int, optional
        Total number of classes.
    label_names : list[str], optional
        Class names for per-class reporting.

    Returns
    -------
    ClassificationMetrics
    """
    metrics = ClassificationMetrics()

    metrics.accuracy = accuracy_score(y_true, y_pred)
    metrics.balanced_accuracy = balanced_accuracy_score(y_true, y_pred)
    metrics.macro_f1 = f1_score(y_true, y_pred, average="macro", zero_division=0)

    if y_prob is not None:
        k_max = min(3, y_prob.shape[1])
        metrics.top_3_accuracy = top_k_accuracy_score(
            y_true, y_prob, k=k_max, labels=range(y_prob.shape[1])
        )
        k_max = min(5, y_prob.shape[1])
        metrics.top_5_accuracy = top_k_accuracy_score(
            y_true, y_prob, k=k_max, labels=range(y_prob.shape[1])
        )

    labels = list(range(n_classes)) if n_classes else None
    metrics.confusion = confusion_matrix(y_true, y_pred, labels=labels)

    # Per-class accuracy
    if label_names:
        for i, name in enumerate(label_names):
            mask = y_true == i
            if mask.sum() > 0:
                metrics.per_class_accuracy[name] = float((y_pred[mask] == i).mean())

    return metrics


def compute_retrieval_metrics(
    similarities: np.ndarray,
    ground_truth_indices: np.ndarray | None = None,
) -> RetrievalMetrics:
    """Compute retrieval metrics from similarity matrix.

    Parameters
    ----------
    similarities : np.ndarray
        Similarity matrix, shape (N_queries, N_candidates).
        Higher = more similar. Diagonal = correct matches (if ground_truth is None).
    ground_truth_indices : np.ndarray, optional
        Index of correct candidate for each query, shape (N_queries,).
        If None, assumes diagonal (query_i matches candidate_i).

    Returns
    -------
    RetrievalMetrics
    """
    n_queries, n_candidates = similarities.shape

    if ground_truth_indices is None:
        ground_truth_indices = np.arange(n_queries)

    # Rank candidates by similarity (descending)
    sorted_indices = np.argsort(-similarities, axis=1)

    metrics = RetrievalMetrics()
    ranks = []

    for i in range(n_queries):
        gt = ground_truth_indices[i]
        rank = np.where(sorted_indices[i] == gt)[0][0] + 1  # 1-indexed
        ranks.append(rank)

    ranks = np.array(ranks)

    metrics.recall_at_1 = float((ranks <= 1).mean())
    metrics.recall_at_5 = float((ranks <= 5).mean())
    metrics.recall_at_10 = float((ranks <= 10).mean())
    metrics.mrr = float((1.0 / ranks).mean())
    metrics.median_rank = float(np.median(ranks))

    return metrics


def compute_generation_metrics(
    predictions: list[str],
    references: list[str],
) -> GenerationMetrics:
    """Compute generation metrics.

    Parameters
    ----------
    predictions : list[str]
        Generated text sequences.
    references : list[str]
        Reference text sequences.

    Returns
    -------
    GenerationMetrics
    """
    metrics = GenerationMetrics()

    # WER and CER
    try:
        from jiwer import wer, cer

        metrics.wer = wer(references, predictions)
        metrics.cer = cer(references, predictions)
    except ImportError:
        pass

    # Semantic similarity
    try:
        from sentence_transformers import SentenceTransformer

        model = SentenceTransformer("paraphrase-multilingual-MiniLM-L12-v2")
        pred_embs = model.encode(predictions, convert_to_tensor=True)
        ref_embs = model.encode(references, convert_to_tensor=True)

        cos_sim = torch.nn.functional.cosine_similarity(pred_embs, ref_embs)
        metrics.semantic_similarity = float(cos_sim.mean().item())
    except ImportError:
        pass

    # BERTScore
    try:
        from bert_score import score as bert_score_fn

        P, R, F1 = bert_score_fn(predictions, references, lang="en", verbose=False)
        metrics.bert_score_f1 = float(F1.mean().item())
    except ImportError:
        pass

    return metrics
