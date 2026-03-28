"""Lightweight wall-clock profiler for training loops."""

from __future__ import annotations

import logging
import time
from collections import defaultdict

from torch.utils.tensorboard import SummaryWriter

logger = logging.getLogger(__name__)


class Profiler:
    """Accumulates wall-clock time per named phase and logs periodic summaries."""

    def __init__(self):
        self._totals: dict[str, float] = defaultdict(float)
        self._counts: dict[str, int] = defaultdict(int)
        self._start: float = 0.0

    def begin(self):
        """Mark the start of a timed section."""
        self._start = time.perf_counter()

    def record(self, phase: str):
        """Record elapsed time since the last ``begin()`` (or ``record()``) call."""
        now = time.perf_counter()
        self._totals[phase] += now - self._start
        self._counts[phase] += 1
        self._start = now

    def log_and_reset(
        self,
        episode: int,
        writer: SummaryWriter | None = None,
    ):
        """Log a summary line and optional TensorBoard scalars, then reset."""
        total = sum(self._totals.values())
        if total == 0:
            return

        parts = []
        for phase in sorted(self._totals, key=self._totals.__getitem__, reverse=True):
            secs = self._totals[phase]
            pct = 100 * secs / total
            parts.append(f"{phase}: {secs:.2f}s ({pct:.0f}%)")
            if writer:
                writer.add_scalar(f"profile/{phase}_pct", pct, episode)
                writer.add_scalar(f"profile/{phase}_sec", secs, episode)

        logger.info(
            "Profile (last window) | %s | total: %.2fs",
            " | ".join(parts),
            total,
        )

        self._totals.clear()
        self._counts.clear()
