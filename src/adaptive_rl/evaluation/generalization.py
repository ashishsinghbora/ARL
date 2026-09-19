"""Generalization evaluation framework for measuring agent transfer to unseen environments."""

from __future__ import annotations

import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional, Sequence, Tuple

import gymnasium as gym
import numpy as np
from pydantic import BaseModel, ConfigDict, Field

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.environments.registry import make_env
from adaptive_rl.evaluation.metrics import EvaluationMetrics
from adaptive_rl.metrics import (
    DefaultOutcomePolicy,
    EpisodeMetrics,
    EpisodeMetricsAccumulator,
    OutcomePolicy,
    TrafficOutcomePolicy,
    compute_rate,
)


class GeneralizationDistribution(BaseModel):
    """Declarative specification of strictly partitioned training and test seed distributions."""

    model_config = ConfigDict(extra="ignore")

    train_seeds: List[int] = Field(..., description="Seeds comprising the training distribution")
    test_seeds: List[int] = Field(..., description="Seeds comprising unseen test distribution")
    description: str = Field(
        default="Disjoint train/test seed partition",
        description="Human-readable description of the generalization benchmark",
    )

    def model_post_init(self, __context: Any) -> None:
        """Validate that training and test seed distributions have strictly zero overlap."""
        train_set = set(self.train_seeds)
        test_set = set(self.test_seeds)
        overlap = train_set.intersection(test_set)
        if overlap:
            raise ValueError(
                f"Data integrity violation: Found {len(overlap)} overlapping seeds between "
                f"training and test distributions (e.g. {sorted(list(overlap))[:5]}). "
                "Train and test distributions must be strictly disjoint."
            )
        if not self.train_seeds:
            raise ValueError("Training seed distribution cannot be empty.")
        if not self.test_seeds:
            raise ValueError("Test seed distribution cannot be empty.")

    @classmethod
    def from_ranges(
        cls,
        train_range: Tuple[int, int] = (1000, 1050),
        test_range: Tuple[int, int] = (2000, 2050),
        description: str = "Disjoint seed ranges",
    ) -> GeneralizationDistribution:
        """Construct distribution from half-open [start, end) seed ranges."""
        train_seeds = list(range(train_range[0], train_range[1]))
        test_seeds = list(range(test_range[0], test_range[1]))
        return cls(train_seeds=train_seeds, test_seeds=test_seeds, description=description)


class GeneralizationReport(BaseModel):
    """Standardized report quantifying an agent's generalization capability on unseen layouts."""

    model_config = ConfigDict(extra="ignore")

    experiment_name: str = Field(..., description="Unique experiment identifier")
    environment_name: str = Field(..., description="Target environment name")
    algorithm_name: str = Field(..., description="RL algorithm used")
    train_seeds: List[int] = Field(..., description="Seeds used for training evaluation")
    test_seeds: List[int] = Field(..., description="Unseen seeds used for evaluation")
    train_metrics: EvaluationMetrics = Field(
        ..., description="Performance on training distribution"
    )
    test_metrics: EvaluationMetrics = Field(
        ..., description="Performance on unseen test distribution"
    )
    generalization_gap_success: Optional[float] = Field(
        default=None,
        description="Drop in success rate: train_success_rate - test_success_rate (or None if undefined)",
    )
    generalization_gap_reward: float = Field(
        ...,
        description="Drop in mean reward: train_mean_reward - test_mean_reward",
    )
    relative_success_retention: Optional[float] = Field(
        default=None,
        description="Ratio of test success rate to train success rate (1.0 = perfect transfer, or None if undefined)",
    )
    environment_parameters: Dict[str, Any] = Field(
        default_factory=dict,
        description="Parameters of the environment instance",
    )
    timestamp: str = Field(
        default_factory=lambda: datetime.now(timezone.utc).isoformat(),
        description="ISO 8601 UTC execution timestamp",
    )
    metadata: Dict[str, Any] = Field(
        default_factory=dict, description="Additional context metadata"
    )

    def save_json(self, path: str | Path) -> Path:
        """Serialize the generalization report to formatted JSON file."""
        target = Path(path)
        target.parent.mkdir(parents=True, exist_ok=True)
        with open(target, "w", encoding="utf-8") as f:
            json.dump(self.model_dump(), f, indent=2)
        return target


class GeneralizationEvaluator:
    """Evaluates trained policies across separate train and unseen test seed distributions."""

    def __init__(
        self,
        algorithm: BaseAlgorithm,
        env: Optional[gym.Env] = None,
        env_name: Optional[str] = None,
        env_kwargs: Optional[Dict[str, Any]] = None,
        outcome_policy: Optional[OutcomePolicy] = None,
    ) -> None:
        """Initialize generalization evaluator.

        Args:
            algorithm: Policy to benchmark.
            env: Instantiated Gymnasium environment (optional).
            env_name: Registered environment name (optional).
            env_kwargs: Parameters passed to make_env.
            outcome_policy: Optional explicit domain OutcomePolicy instance.
        """
        self.algorithm = algorithm
        self.env_kwargs = dict(env_kwargs or {})

        if env_name is not None:
            self.env_name = env_name
            self.env = env if env is not None else make_env(env_name, **self.env_kwargs)
        elif env is not None:
            self.env = env
            spec = getattr(env, "spec", None)
            if spec is not None and getattr(spec, "id", None):
                self.env_name = str(spec.id)
            else:
                self.env_name = type(env).__name__
        else:
            raise ValueError("GeneralizationEvaluator requires either 'env' or 'env_name'.")

        if outcome_policy is not None:
            self.outcome_policy: OutcomePolicy = outcome_policy
            self._explicit_policy = True
        elif "traffic" in self.env_name.lower():
            self.outcome_policy = TrafficOutcomePolicy()
            self._explicit_policy = False
        else:
            self.outcome_policy = DefaultOutcomePolicy()
            self._explicit_policy = False

        self.last_episode_metrics: List[EpisodeMetrics] = []
        self.last_train_metrics: List[EpisodeMetrics] = []
        self.last_test_metrics: List[EpisodeMetrics] = []

    def _make_episode_accumulator(self) -> EpisodeMetricsAccumulator:
        """Bind a policy for one episode without treating inferred defaults as explicit."""
        if self._explicit_policy:
            return EpisodeMetricsAccumulator(outcome_policy=self.outcome_policy)
        if isinstance(self.outcome_policy, TrafficOutcomePolicy):
            return EpisodeMetricsAccumulator(is_traffic=True)
        return EpisodeMetricsAccumulator()

    def _evaluate_seed_list(
        self,
        seeds: Sequence[int],
        deterministic: bool = True,
    ) -> EvaluationMetrics:
        """Evaluate agent over exact list of seeds, one episode per seed."""
        episode_metrics: List[EpisodeMetrics] = []

        for seed in seeds:
            obs, info = self.env.reset(seed=int(seed))
            acc = self._make_episode_accumulator()
            done = False

            while not done:
                action, _ = self.algorithm.predict(obs, deterministic=deterministic)
                obs, reward, terminated, truncated, step_info = self.env.step(action)
                acc.record_step(
                    reward=float(reward),
                    terminated=terminated,
                    truncated=truncated,
                    info=step_info,
                )
                done = terminated or truncated

            m = acc.finish()
            episode_metrics.append(m)

        self.last_episode_metrics = episode_metrics

        total = len(seeds)
        rewards = [m.reward for m in episode_metrics]
        lengths = [m.length for m in episode_metrics]

        # Rate aggregation denominator semantics:
        # Rates (success_rate, collision_rate) are calculated among defined episodes (non-None).
        # If unavailable across all episodes (all None), the rate evaluates to None.
        success_rate = compute_rate([m.success for m in episode_metrics])
        collision_rate = compute_rate([m.collision for m in episode_metrics])
        truncation_rate: Optional[float] = (
            float(sum(1 for m in episode_metrics if m.truncated) / total) if total > 0 else None
        )

        return EvaluationMetrics(
            episodes=total,
            mean_reward=float(np.mean(rewards)) if rewards else 0.0,
            std_reward=float(np.std(rewards)) if rewards else 0.0,
            min_reward=float(np.min(rewards)) if rewards else 0.0,
            max_reward=float(np.max(rewards)) if rewards else 0.0,
            success_rate=success_rate,
            collision_rate=collision_rate,
            truncation_rate=truncation_rate,
            mean_episode_length=float(np.mean(lengths)) if lengths else 0.0,
            std_episode_length=float(np.std(lengths)) if lengths else 0.0,
            additional_metrics={
                "evaluated_seeds": [int(s) for s in seeds],
                "all_rewards": rewards,
                "all_lengths": lengths,
            },
        )

    def evaluate_generalization(
        self,
        distribution: GeneralizationDistribution,
        experiment_name: str = "generalization_benchmark",
        deterministic: bool = True,
        metadata: Optional[Dict[str, Any]] = None,
    ) -> GeneralizationReport:
        """Execute twin evaluation over training seeds and unseen test seeds.

        Args:
            distribution: Validated disjoint train and test seeds.
            experiment_name: Identifier for the experiment run.
            deterministic: Whether policy actions are chosen deterministically.
            metadata: Additional run context.

        Returns:
            GeneralizationReport: Complete comparison report and metrics.
        """
        train_metrics = self._evaluate_seed_list(
            distribution.train_seeds, deterministic=deterministic
        )
        self.last_train_metrics = list(self.last_episode_metrics)

        test_metrics = self._evaluate_seed_list(
            distribution.test_seeds, deterministic=deterministic
        )
        self.last_test_metrics = list(self.last_episode_metrics)

        if train_metrics.success_rate is not None and test_metrics.success_rate is not None:
            gap_success: Optional[float] = float(
                train_metrics.success_rate - test_metrics.success_rate
            )
            retention: Optional[float] = (
                float(test_metrics.success_rate / train_metrics.success_rate)
                if train_metrics.success_rate > 0.0
                else (1.0 if test_metrics.success_rate == 0.0 else 0.0)
            )
        else:
            gap_success = None
            retention = None

        gap_reward = float(train_metrics.mean_reward - test_metrics.mean_reward)

        algo_name = getattr(self.algorithm, "name", type(self.algorithm).__name__)

        return GeneralizationReport(
            experiment_name=experiment_name,
            environment_name=self.env_name,
            algorithm_name=algo_name,
            train_seeds=distribution.train_seeds,
            test_seeds=distribution.test_seeds,
            train_metrics=train_metrics,
            test_metrics=test_metrics,
            generalization_gap_success=gap_success,
            generalization_gap_reward=gap_reward,
            relative_success_retention=retention,
            environment_parameters=dict(self.env_kwargs),
            metadata=dict(metadata or {}),
        )

    def close(self) -> None:
        """Close the benchmark environment."""
        if hasattr(self, "env") and self.env is not None:
            self.env.close()
