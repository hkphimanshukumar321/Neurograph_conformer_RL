"""
Functional tests — Evaluation metrics and statistical tests.

Verifies classification, retrieval, and generation metrics produce
correct results on known inputs. Tests statistical methods on
controlled distributions.
"""

import pytest
import numpy as np


class TestClassificationMetrics:
    """Verify classification metrics on known inputs."""

    def test_perfect_accuracy(self):
        from src.evaluation.metrics import compute_classification_metrics
        y_true = np.array([0, 1, 2, 3, 4, 0, 1, 2, 3, 4])
        y_pred = y_true.copy()
        m = compute_classification_metrics(y_true, y_pred, n_classes=5)
        assert m.accuracy == 1.0
        assert m.balanced_accuracy == 1.0
        assert m.macro_f1 == 1.0
        assert m.cohens_kappa == 1.0

    def test_zero_accuracy(self):
        from src.evaluation.metrics import compute_classification_metrics
        y_true = np.array([0, 0, 0, 0, 0])
        y_pred = np.array([1, 1, 1, 1, 1])
        m = compute_classification_metrics(y_true, y_pred, n_classes=2)
        assert m.accuracy == 0.0

    def test_confusion_matrix_shape(self):
        from src.evaluation.metrics import compute_classification_metrics
        y_true = np.array([0, 1, 2, 0, 1, 2])
        y_pred = np.array([0, 2, 1, 0, 0, 2])
        m = compute_classification_metrics(y_true, y_pred, n_classes=3)
        assert m.confusion.shape == (3, 3)
        assert m.confusion.sum() == len(y_true)

    def test_per_class_f1(self):
        from src.evaluation.metrics import compute_classification_metrics
        y_true = np.array([0, 0, 1, 1, 2, 2])
        y_pred = np.array([0, 0, 1, 1, 2, 2])
        m = compute_classification_metrics(y_true, y_pred, n_classes=3)
        assert len(m.per_class_f1) == 3
        assert all(f == 1.0 for f in m.per_class_f1)


class TestRetrievalMetrics:
    """Verify retrieval metrics on known similarity matrices."""

    def test_perfect_retrieval(self):
        from src.evaluation.metrics import compute_retrieval_metrics
        sim = np.eye(10) * 100 + np.random.randn(10, 10) * 0.01
        m = compute_retrieval_metrics(sim)
        assert m.recall_at_1 == 1.0
        assert m.mrr == 1.0
        assert m.median_rank == 1

    def test_worst_retrieval(self):
        from src.evaluation.metrics import compute_retrieval_metrics
        # Anti-diagonal — worst case
        n = 5
        sim = np.zeros((n, n))
        for i in range(n):
            sim[i, (n - 1 - i)] = 10.0
        m = compute_retrieval_metrics(sim)
        # Not perfect
        assert m.recall_at_1 < 1.0 or n == 1

    def test_recall_at_k_monotonic(self):
        from src.evaluation.metrics import compute_retrieval_metrics
        sim = np.random.randn(20, 20)
        m = compute_retrieval_metrics(sim)
        assert m.recall_at_1 <= m.recall_at_5
        assert m.recall_at_5 <= m.recall_at_10


class TestStatisticalMethods:
    """Verify statistical methods on controlled distributions."""

    def test_mcnemar_identical_models(self):
        from src.evaluation.statistical import mcnemar_test
        y = np.random.randint(0, 3, 100)
        p = y.copy()
        result = mcnemar_test(y, p, p)
        assert result["p_value"] == 1.0
        assert not result["significant"]

    def test_mcnemar_different_models(self):
        from src.evaluation.statistical import mcnemar_test
        np.random.seed(42)
        y = np.random.randint(0, 2, 200)
        p_a = y.copy()  # Perfect
        p_b = np.random.randint(0, 2, 200)  # Random
        result = mcnemar_test(y, p_a, p_b)
        assert result["significant"]  # Should be significant

    def test_permutation_test_same_distribution(self):
        from src.evaluation.statistical import paired_permutation_test
        np.random.seed(42)
        scores = np.random.randn(50)
        result = paired_permutation_test(scores, scores, n_permutations=1000)
        assert result["observed_diff"] == 0.0
        assert result["p_value"] == 1.0

    def test_permutation_test_different_distributions(self):
        from src.evaluation.statistical import paired_permutation_test
        np.random.seed(42)
        a = np.random.randn(100) + 2.0  # Higher mean
        b = np.random.randn(100)
        result = paired_permutation_test(a, b, n_permutations=1000)
        assert result["observed_diff"] > 0

    def test_confidence_interval_contains_mean(self):
        from src.evaluation.statistical import compute_confidence_interval
        scores = np.random.randn(100) + 0.5
        mean, lower, upper = compute_confidence_interval(scores, confidence=0.95)
        assert lower < mean < upper

    def test_bootstrap_ci(self):
        from src.evaluation.statistical import compute_confidence_interval
        scores = np.random.randn(100)
        mean, lower, upper = compute_confidence_interval(
            scores, confidence=0.95, method="bootstrap"
        )
        assert lower < mean < upper

    def test_bonferroni_correction(self):
        from src.evaluation.statistical import bonferroni_correction
        p_vals = [0.01, 0.02, 0.05, 0.10]
        corrected = bonferroni_correction(p_vals, alpha=0.05)
        assert corrected[0]["corrected_p"] == 0.04  # 0.01 * 4
        assert corrected[0]["significant"]  # 0.04 < 0.05
        assert not corrected[3]["significant"]  # 0.40 > 0.05


class TestCallbackFunctional:
    """Verify training callbacks work correctly."""

    def test_early_stopping_triggers(self):
        from src.training.callbacks import EarlyStopping
        es = EarlyStopping(patience=3, monitor="val_loss", mode="min")
        # Simulated improving then plateauing
        values = [1.0, 0.9, 0.8, 0.8, 0.8, 0.8]
        for v in values:
            stopped = es(v)
        assert stopped  # Should trigger after 3 non-improving epochs

    def test_early_stopping_improves(self):
        from src.training.callbacks import EarlyStopping
        es = EarlyStopping(patience=5, mode="max")
        values = [0.1, 0.2, 0.3, 0.4, 0.5]
        for v in values:
            stopped = es(v)
        assert not stopped  # Always improving

    def test_scheduler_warmup_increases(self):
        import torch
        from src.training.schedulers import CosineWarmupScheduler
        opt = torch.optim.Adam([torch.randn(5, requires_grad=True)], lr=1e-3)
        sched = CosineWarmupScheduler(opt, warmup_epochs=5, max_epochs=50)
        lrs = []
        for _ in range(10):
            lrs.append(sched.get_last_lr()[0])
            sched.step()
        # LR should increase during warmup
        assert lrs[4] > lrs[0]
