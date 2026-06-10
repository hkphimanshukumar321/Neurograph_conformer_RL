"""
Device management — automatic GPU detection, multi-GPU support, memory reporting.
"""

from __future__ import annotations

from dataclasses import dataclass

import torch


@dataclass
class DeviceInfo:
    """Container for device information."""

    device: torch.device
    name: str
    type: str  # "cuda", "mps", "cpu"
    memory_total_gb: float | None = None
    memory_free_gb: float | None = None


class DeviceManager:
    """Manages device selection and memory monitoring.

    Parameters
    ----------
    device : str
        Device specification: "auto", "cuda", "cuda:0", "cpu", "mps".
    """

    def __init__(self, device: str = "auto"):
        self.device = self._resolve_device(device)
        self.info = self._get_info()

    @staticmethod
    def _resolve_device(device: str) -> torch.device:
        """Resolve device string to torch.device."""
        if device == "auto":
            if torch.cuda.is_available():
                return torch.device("cuda")
            elif hasattr(torch.backends, "mps") and torch.backends.mps.is_available():
                return torch.device("mps")
            else:
                return torch.device("cpu")
        return torch.device(device)

    def _get_info(self) -> DeviceInfo:
        """Gather device information."""
        dtype = self.device.type
        name = dtype

        mem_total = None
        mem_free = None

        if dtype == "cuda":
            idx = self.device.index or 0
            name = torch.cuda.get_device_name(idx)
            mem_total = torch.cuda.get_device_properties(idx).total_mem / (1024**3)
            mem_free = (
                torch.cuda.get_device_properties(idx).total_mem
                - torch.cuda.memory_allocated(idx)
            ) / (1024**3)

        return DeviceInfo(
            device=self.device,
            name=name,
            type=dtype,
            memory_total_gb=mem_total,
            memory_free_gb=mem_free,
        )

    def summary(self) -> str:
        """Return a human-readable device summary."""
        lines = [
            f"Device: {self.info.name} ({self.info.type})",
        ]
        if self.info.memory_total_gb is not None:
            lines.append(f"  Memory: {self.info.memory_total_gb:.1f} GB total")
        if self.info.memory_free_gb is not None:
            lines.append(f"  Free:   {self.info.memory_free_gb:.1f} GB")
        return "\n".join(lines)

    @staticmethod
    def memory_stats() -> dict[str, float] | None:
        """Return current CUDA memory statistics in GB, or None if CPU."""
        if not torch.cuda.is_available():
            return None
        return {
            "allocated_gb": torch.cuda.memory_allocated() / (1024**3),
            "reserved_gb": torch.cuda.memory_reserved() / (1024**3),
            "max_allocated_gb": torch.cuda.max_memory_allocated() / (1024**3),
        }

    @staticmethod
    def reset_peak_memory() -> None:
        """Reset peak memory tracking (useful for profiling)."""
        if torch.cuda.is_available():
            torch.cuda.reset_peak_memory_stats()


def get_device(device: str = "auto") -> torch.device:
    """Convenience function to get a torch.device.

    Parameters
    ----------
    device : str
        Device specification: "auto", "cuda", "cuda:0", "cpu", "mps".

    Returns
    -------
    torch.device
    """
    return DeviceManager._resolve_device(device)
