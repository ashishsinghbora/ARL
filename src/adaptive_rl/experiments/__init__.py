"""Experiment runners and lifecycle management for AdaptiveRL."""

from __future__ import annotations

from adaptive_rl.experiments.manager import ExperimentManager, ExperimentManifest, ExperimentResult
from adaptive_rl.experiments.metadata import (
    EpisodeRecord,
    ExperimentMetadata,
    load_episodes_csv,
    save_episodes_csv,
)
from adaptive_rl.experiments.runner import BaseExperimentRunner

__all__ = [
    "BaseExperimentRunner",
    "EpisodeRecord",
    "ExperimentMetadata",
    "ExperimentManager",
    "ExperimentManifest",
    "ExperimentResult",
    "GeneralizationExperimentRunner",
    "load_episodes_csv",
    "save_episodes_csv",
]


def __getattr__(name: str) -> object:
    """Lazy-load heavy modules to avoid circular imports at package init time."""
    if name == "GeneralizationExperimentRunner":
        from adaptive_rl.experiments.generalization_runner import (
            GeneralizationExperimentRunner,
        )

        return GeneralizationExperimentRunner
    raise AttributeError(f"module 'adaptive_rl.experiments' has no attribute {name!r}")
