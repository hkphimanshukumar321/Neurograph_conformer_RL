"""
Streaming inference — real-time EEG processing with ring buffer.
"""

from __future__ import annotations

import logging
import time
from collections import deque

import numpy as np
import torch
import torch.nn as nn

logger = logging.getLogger(__name__)


class StreamingInference:
    """Real-time streaming inference for EEG classification.

    Maintains a ring buffer of incoming EEG data and periodically
    classifies the current window.

    Parameters
    ----------
    model : nn.Module
        Trained model (should be exported/quantized for efficiency).
    window_size_s : float
        Classification window size in seconds.
    stride_s : float
        Stride between classifications in seconds.
    sfreq : float
        Sampling frequency in Hz.
    n_channels : int
        Number of EEG channels.
    confidence_threshold : float
        Minimum confidence to emit a prediction.
    device : str
        Inference device.
    """

    def __init__(
        self,
        model: nn.Module,
        window_size_s: float = 2.0,
        stride_s: float = 0.5,
        sfreq: float = 250.0,
        n_channels: int = 9,
        confidence_threshold: float = 0.7,
        device: str = "cpu",
    ):
        self.model = model.to(device).eval()
        self.device = device
        self.sfreq = sfreq
        self.n_channels = n_channels
        self.confidence_threshold = confidence_threshold

        self.window_samples = int(window_size_s * sfreq)
        self.stride_samples = int(stride_s * sfreq)

        # Ring buffer
        self.buffer = np.zeros((n_channels, 0), dtype=np.float32)
        self.samples_since_last = 0

        # History
        self.predictions: list[dict] = []

        logger.info(
            f"StreamingInference: window={window_size_s}s ({self.window_samples} samples), "
            f"stride={stride_s}s ({self.stride_samples} samples), "
            f"confidence={confidence_threshold}"
        )

    def feed(self, new_data: np.ndarray) -> list[dict]:
        """Feed new EEG samples into the buffer.

        Parameters
        ----------
        new_data : np.ndarray
            New EEG data, shape (n_channels, n_new_samples).

        Returns
        -------
        list[dict]
            List of predictions emitted (may be empty).
        """
        assert new_data.shape[0] == self.n_channels, \
            f"Expected {self.n_channels} channels, got {new_data.shape[0]}"

        # Append to buffer
        self.buffer = np.concatenate([self.buffer, new_data], axis=1)
        self.samples_since_last += new_data.shape[1]

        # Emit predictions at stride intervals
        results = []
        while (self.samples_since_last >= self.stride_samples and
               self.buffer.shape[1] >= self.window_samples):

            window = self.buffer[:, -self.window_samples:]
            prediction = self._classify(window)
            results.append(prediction)
            self.samples_since_last -= self.stride_samples

        # Limit buffer size (keep 2x window to avoid allocation)
        max_buffer = self.window_samples * 2
        if self.buffer.shape[1] > max_buffer:
            self.buffer = self.buffer[:, -max_buffer:]

        return results

    @torch.no_grad()
    def _classify(self, window: np.ndarray) -> dict:
        """Classify a single window.

        Parameters
        ----------
        window : np.ndarray
            EEG window, shape (n_channels, window_samples).

        Returns
        -------
        dict with prediction, confidence, and latency.
        """
        t_start = time.perf_counter()

        # Convert to tensor
        x = torch.from_numpy(window).unsqueeze(0).to(self.device)

        # Forward pass
        outputs = self.model(x, task="classification")
        logits = outputs["cls_logits"]

        # Softmax
        probs = torch.softmax(logits, dim=-1)
        confidence, predicted = probs.max(dim=-1)
        confidence = confidence.item()
        predicted = predicted.item()

        t_end = time.perf_counter()
        latency_ms = (t_end - t_start) * 1000

        result = {
            "prediction": predicted,
            "confidence": confidence,
            "latency_ms": latency_ms,
            "timestamp": time.time(),
            "accepted": confidence >= self.confidence_threshold,
        }

        if not result["accepted"]:
            result["prediction"] = -1  # uncertain

        self.predictions.append(result)
        return result

    def reset(self) -> None:
        """Reset buffer and prediction history."""
        self.buffer = np.zeros((self.n_channels, 0), dtype=np.float32)
        self.samples_since_last = 0
        self.predictions.clear()

    def get_stats(self) -> dict:
        """Get streaming inference statistics."""
        if not self.predictions:
            return {}

        latencies = [p["latency_ms"] for p in self.predictions]
        accepted = [p for p in self.predictions if p["accepted"]]

        return {
            "total_predictions": len(self.predictions),
            "accepted_predictions": len(accepted),
            "acceptance_rate": len(accepted) / len(self.predictions),
            "mean_latency_ms": float(np.mean(latencies)),
            "p95_latency_ms": float(np.percentile(latencies, 95)),
            "mean_confidence": float(np.mean([p["confidence"] for p in self.predictions])),
        }
