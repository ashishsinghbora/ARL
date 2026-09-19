"""Reproducibility metadata and experiment provenance for AdaptiveRL.

Every training run emits a metadata.json and episodes.csv alongside the saved
model weights.  Together they form the provenance record needed to reproduce
the exact experiment from scratch:

    experiments/results/
        models/
            <name>_final.zip          <- model weights
        metadata/
            <name>_metadata.json      <- full run provenance
            <name>_episodes.csv       <- per-episode reward/length/outcome
"""

from __future__ import annotations

import csv
import json
import platform
import sys
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional


def _utc_now() -> str:
    """Return current UTC time in ISO 8601 format."""
    return datetime.now(timezone.utc).isoformat()


@dataclass
class ExperimentMetadata:
    """Complete reproducibility record for one training run.

    Attributes:
        experiment_name: Unique identifier from config.name.
        algorithm: Algorithm name (e.g. 'ppo', 'sac').
        environment: Environment name (e.g. 'gridworld').
        seed: Random seed used for the experiment.
        total_timesteps: Configured training budget.
        actual_timesteps: Timesteps actually executed.
        episodes_completed: Number of episodes that terminated during training.
        mean_reward: Rolling mean episodic reward at end of training.
        success_rate: Fraction of defined episodes that ended in success (None if undefined).
        collision_rate: Fraction of defined episodes that ended in collision (None if undefined).
        final_model_path: Absolute path to saved final model weights.
        checkpoint_paths: List of intermediate checkpoint paths.
        config_snapshot: Full serialized ExperimentConfig.
        python_version: Python interpreter version string.
        platform_info: OS / platform identifier.
        adaptive_rl_version: Package version.
        started_at: ISO 8601 UTC timestamp when training started.
        finished_at: ISO 8601 UTC timestamp when training finished.
        duration_seconds: Wall-clock training duration in seconds.
        notes: Optional free-form annotation.
    """

    experiment_name: str
    algorithm: str
    environment: str
    seed: int
    total_timesteps: int
    actual_timesteps: int
    episodes_completed: int
    mean_reward: float
    success_rate: Optional[float] = None
    collision_rate: Optional[float] = None
    final_model_path: str = ""
    checkpoint_paths: List[str] = field(default_factory=list)
    config_snapshot: Dict[str, Any] = field(default_factory=dict)
    python_version: str = field(default_factory=lambda: sys.version)
    platform_info: str = field(default_factory=lambda: platform.platform())
    adaptive_rl_version: str = "0.1.0"
    started_at: str = field(default_factory=_utc_now)
    finished_at: str = field(default_factory=_utc_now)
    duration_seconds: float = 0.0
    notes: str = ""

    def save(self, output_dir: str | Path, name: Optional[str] = None) -> Path:
        """Serialize experiment metadata to JSON file.

        Args:
            output_dir: Directory to write the metadata file into.
            name: Optional filename stem (defaults to experiment_name).

        Returns:
            Path: Written metadata file path.
        """
        target_dir = Path(output_dir)
        target_dir.mkdir(parents=True, exist_ok=True)
        stem = name or self.experiment_name
        target_path = target_dir / f"{stem}_metadata.json"
        data = asdict(self)
        with open(target_path, "w", encoding="utf-8") as f:
            json.dump(data, f, indent=2, default=str)
        return target_path

    @classmethod
    def load(cls, path: str | Path) -> ExperimentMetadata:
        """Deserialize an ExperimentMetadata from a saved JSON file.

        Args:
            path: Path to the metadata JSON file.

        Returns:
            ExperimentMetadata: Loaded instance.
        """
        with open(path, "r", encoding="utf-8") as f:
            data = json.load(f)
        return cls(**data)


@dataclass
class EpisodeRecord:
    """Per-episode training outcome record."""

    episode: int
    reward: float
    length: int
    success: Optional[bool] = None
    collision: Optional[bool] = None
    timestep: int = 0


def save_episodes_csv(
    records: List[EpisodeRecord],
    output_dir: str | Path,
    name: str,
) -> Path:
    """Write per-episode training records to a CSV file.

    Args:
        records: List of EpisodeRecord instances.
        output_dir: Directory to write the CSV file into.
        name: Experiment name used as filename stem.

    Returns:
        Path: Written CSV file path.
    """
    target_dir = Path(output_dir)
    target_dir.mkdir(parents=True, exist_ok=True)
    csv_path = target_dir / f"{name}_episodes.csv"

    fieldnames = ["episode", "reward", "length", "success", "collision", "timestep"]
    with open(csv_path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        for rec in records:
            d = asdict(rec)
            writer.writerow({k: ("" if v is None else v) for k, v in d.items()})

    return csv_path


def load_episodes_csv(path: str | Path) -> List[EpisodeRecord]:
    """Read per-episode training records from a CSV file, preserving nullability.

    Empty strings are parsed as None; 'True'/'False' (and numeric 1/0) are parsed
    as booleans.

    Args:
        path: Path to the episodes CSV file.

    Returns:
        List of EpisodeRecord instances with preserved None/bool outcomes.
    """
    records: List[EpisodeRecord] = []
    csv_path = Path(path)
    if not csv_path.exists():
        return records

    def _parse_optional_bool(val: Any) -> Optional[bool]:
        if val is None:
            return None
        s = str(val).strip()
        if s == "" or s.lower() == "none":
            return None
        if s.lower() in ("true", "1"):
            return True
        if s.lower() in ("false", "0"):
            return False
        return None

    with open(csv_path, "r", newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            records.append(
                EpisodeRecord(
                    episode=int(row["episode"]),
                    reward=float(row["reward"]),
                    length=int(row["length"]),
                    success=_parse_optional_bool(row.get("success")),
                    collision=_parse_optional_bool(row.get("collision")),
                    timestep=int(row.get("timestep", 0) or 0),
                )
            )
    return records
