"""N-tuple network hyperparameter configuration."""

from dataclasses import dataclass, field

DEFAULT_PATTERNS: list[tuple[int, ...]] = [
    (0, 1, 2, 3, 4, 5),
    (4, 5, 6, 7, 8, 9),
    (4, 5, 6, 8, 9, 10),
    (8, 9, 10, 12, 13, 14),
]


@dataclass
class NTupleConfig:
    """
    Hyperparameters for n-tuple network training.

    Parameters
    ----------
    patterns : list[tuple[int, ...]]
        Board position patterns for LUT indexing. Each tuple lists positions
        (0-15) in the 4x4 grid. Default: four 6-tuples from Szubert &
        Jaskowski 2014.
    lr : float
        Learning rate for TD updates.
    max_episodes : int
        Maximum number of training episodes.
    eval_interval : int
        Episodes between evaluations.
    eval_episodes : int
        Number of episodes per evaluation.
    v_init : float
        Optimistic initialization value. The total initial board evaluation is
        distributed evenly across all LUT weights. 0.0 (default) disables
        optimistic initialization.
    """

    patterns: list[tuple[int, ...]] = field(
        default_factory=lambda: list(DEFAULT_PATTERNS)
    )
    lr: float = 0.0025
    max_episodes: int = 100_000
    eval_interval: int = 1_000
    eval_episodes: int = 25
    v_init: float = 0.0
