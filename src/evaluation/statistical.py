"""
Statistical significance tests for comparing models/ablations.

Implements McNemar's test, paired permutation test, and corrected paired t-test
as required for IEEE-level rigor.
"""

from __future__ import annotations

import logging
from typing import Literal

import numpy as np
from scipy import stats

logger = logging.getLogger(__name__)


def mcnemar_test(
    y_true: np.ndarray,
    preds_a: np.ndarray,
    preds_b: np.ndarray,
    correction: bool = True,
) -> dict[str, float]:
    """McNemar's test for paired nominal data.

    Tests whether model A and model B make different types of errors.

    Parameters
    ----------
    y_true : np.ndarray
        Ground truth labels.
    preds_a : np.ndarray
        Predictions from model A.
    preds_b : np.ndarray
        Predictions from model B.
    correction : bool
        Whether to apply continuity correction.

    Returns
    -------
    dict with:
        'statistic': test statistic
        'p_value': p-value
        'significant': bool (at α=0.05)
        'b_c': count where A correct, B wrong
        'c_b': count where A wrong, B correct
    """
    correct_a = (preds_a == y_true)
    correct_b = (preds_b == y_true)

    # Contingency table
    b_c = np.sum(correct_a & ~correct_b)  # A correct, B wrong
    c_b = np.sum(~correct_a & correct_b)  # A wrong, B correct

    if b_c + c_b == 0:
        return {
            "statistic": 0.0,
            "p_value": 1.0,
            "significant": False,
            "b_c": int(b_c),
            "c_b": int(c_b),
        }

    if correction:
        statistic = (abs(b_c - c_b) - 1) ** 2 / (b_c + c_b)
    else:
        statistic = (b_c - c_b) ** 2 / (b_c + c_b)

    p_value = 1 - stats.chi2.cdf(statistic, df=1)

    return {
        "statistic": float(statistic),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "b_c": int(b_c),
        "c_b": int(c_b),
    }


def paired_permutation_test(
    scores_a: np.ndarray,
    scores_b: np.ndarray,
    n_permutations: int = 10000,
    seed: int = 42,
) -> dict[str, float]:
    """Paired permutation test for comparing two models.

    Non-parametric test that does not assume normal distribution.

    Parameters
    ----------
    scores_a : np.ndarray
        Per-sample scores (e.g., accuracy) from model A, shape (N,).
    scores_b : np.ndarray
        Per-sample scores from model B, shape (N,).
    n_permutations : int
        Number of random permutations.
    seed : int
        Random seed for reproducibility.

    Returns
    -------
    dict with:
        'observed_diff': mean(A) - mean(B)
        'p_value': two-tailed p-value
        'significant': bool (at α=0.05)
    """
    rng = np.random.RandomState(seed)
    diffs = scores_a - scores_b
    observed_diff = np.mean(diffs)

    count = 0
    for _ in range(n_permutations):
        signs = rng.choice([-1, 1], size=len(diffs))
        perm_diff = np.mean(signs * diffs)
        if abs(perm_diff) >= abs(observed_diff):
            count += 1

    p_value = count / n_permutations

    return {
        "observed_diff": float(observed_diff),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "n_permutations": n_permutations,
    }


def corrected_paired_ttest(
    cv_scores_a: np.ndarray,
    cv_scores_b: np.ndarray,
    n_train: int,
    n_test: int,
) -> dict[str, float]:
    """Corrected repeated k-fold cross-validation paired t-test.

    Nadeau & Bengio (2003) correction for dependent CV scores.

    Parameters
    ----------
    cv_scores_a : np.ndarray
        Per-fold scores from model A, shape (n_folds * n_repeats,).
    cv_scores_b : np.ndarray
        Per-fold scores from model B.
    n_train : int
        Training set size per fold.
    n_test : int
        Test set size per fold.

    Returns
    -------
    dict with t-statistic, corrected p-value, and significance.
    """
    diffs = cv_scores_a - cv_scores_b
    n = len(diffs)
    mean_diff = np.mean(diffs)
    var_diff = np.var(diffs, ddof=1)

    # Nadeau & Bengio correction
    correction = (1 / n) + (n_test / n_train)
    t_stat = mean_diff / np.sqrt(correction * var_diff + 1e-10)
    df = n - 1
    p_value = 2 * (1 - stats.t.cdf(abs(t_stat), df=df))

    return {
        "t_statistic": float(t_stat),
        "p_value": float(p_value),
        "significant": p_value < 0.05,
        "mean_diff": float(mean_diff),
        "n_folds": n,
        "correction_factor": float(correction),
    }


def bonferroni_correction(p_values: list[float], alpha: float = 0.05) -> list[dict]:
    """Apply Bonferroni correction for multiple comparisons.

    Parameters
    ----------
    p_values : list[float]
        Uncorrected p-values.
    alpha : float
        Family-wise error rate.

    Returns
    -------
    list[dict]
        Each entry has 'original_p', 'corrected_p', and 'significant'.
    """
    n = len(p_values)
    results = []
    for p in p_values:
        corrected = min(p * n, 1.0)
        results.append({
            "original_p": p,
            "corrected_p": corrected,
            "significant": corrected < alpha,
        })
    return results


def compute_confidence_interval(
    scores: np.ndarray,
    confidence: float = 0.95,
    method: Literal["normal", "bootstrap"] = "normal",
    n_bootstrap: int = 10000,
    seed: int = 42,
) -> tuple[float, float, float]:
    """Compute confidence interval for a metric.

    Parameters
    ----------
    scores : np.ndarray
        Per-sample or per-fold scores.
    confidence : float
        Confidence level (e.g., 0.95).
    method : str
        'normal' (parametric) or 'bootstrap' (non-parametric).
    n_bootstrap : int
        Number of bootstrap resamples (if method='bootstrap').
    seed : int
        Random seed.

    Returns
    -------
    tuple[float, float, float]
        (mean, lower_bound, upper_bound)
    """
    mean = float(np.mean(scores))

    if method == "normal":
        se = stats.sem(scores)
        h = se * stats.t.ppf((1 + confidence) / 2, len(scores) - 1)
        return mean, mean - h, mean + h

    elif method == "bootstrap":
        rng = np.random.RandomState(seed)
        bootstrap_means = []
        for _ in range(n_bootstrap):
            sample = rng.choice(scores, size=len(scores), replace=True)
            bootstrap_means.append(np.mean(sample))

        alpha = (1 - confidence) / 2
        lower = float(np.percentile(bootstrap_means, alpha * 100))
        upper = float(np.percentile(bootstrap_means, (1 - alpha) * 100))
        return mean, lower, upper

    else:
        raise ValueError(f"Unknown CI method: {method}")
