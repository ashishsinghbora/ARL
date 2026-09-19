"""Evaluation metrics data structures and standardization schemas."""

from __future__ import annotations

from typing import Any, Dict, Optional

from pydantic import BaseModel, ConfigDict, Field

# Backward-compatible re-exports from canonical adaptive_rl.metrics module
from adaptive_rl.metrics import (
    EpisodeMetrics as EpisodeMetrics,
)
from adaptive_rl.metrics import (
    EpisodeMetricsAccumulator as EpisodeMetricsAccumulator,
)
from adaptive_rl.metrics import (
    compute_rate as compute_rate,
)
from adaptive_rl.metrics import (
    extract_episode_metrics as extract_episode_metrics,
)


class EvaluationMetrics(BaseModel):
    """Container for reinforcement learning evaluation results.

    Tracks core episodic return statistics, episode-level outcome rates,
    and optional domain-specific telemetry. Metrics unavailable for a given
    environment (e.g. collision_rate in non-spatial environments) are explicitly
    set to None, never silently collapsed into 0.0.

    Rate Denominator Semantics:
        Outcome rates (success_rate, collision_rate, overflow_rate) are calculated
        among episodes where the metric is defined (non-None). Episodes where
        the metric was not tracked or unavailable (None) are excluded from both
        numerator and denominator. If a metric is unavailable across all episodes
        (all None), the rate evaluates to None.
    """

    model_config = ConfigDict(extra="ignore")

    episodes: int = Field(..., gt=0, description="Total evaluation episodes executed")
    mean_reward: float = Field(..., description="Mean cumulative episodic reward")
    std_reward: float = Field(0.0, description="Standard deviation of episodic reward")
    min_reward: float = Field(0.0, description="Minimum episodic reward observed")
    max_reward: float = Field(0.0, description="Maximum episodic reward observed")
    success_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of episodes reaching target among episodes where success is defined (None if unavailable)",
    )
    collision_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of episodes ending in collision among episodes where collision is defined (None if unavailable)",
    )
    overflow_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of episodes experiencing queue overflow (traffic only; None if unavailable)",
    )
    truncation_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes reaching max step limit"
    )
    mean_episode_length: float = Field(..., ge=0.0, description="Mean step count per episode")
    std_episode_length: float = Field(
        0.0, ge=0.0, description="Standard deviation of episode length"
    )
    additional_metrics: Dict[str, Any] = Field(
        default_factory=dict,
        description="Environment-specific metrics (e.g. energy consumption, path length, traffic telemetry)",
    )


class StandardizedExperimentMetrics(BaseModel):
    """Standardized cross-paradigm evaluation metrics schema for AdaptiveRL.

    Provides a stable, uniform schema across reinforcement learning policies
    and classical deterministic/sampling planners. Metrics unavailable for a
    particular algorithm or environment are explicitly set to None (JSON null),
    never silently defaulted to 0.

    Aggregation Rules for Traffic Telemetry:
        - mean_queue_length: Mean of per-timestep total queue lengths averaged across all evaluation timesteps.
        - mean_max_wait_time: Mean of per-episode maximum vehicle waiting times across all evaluation episodes.
        - mean_wait_time: Mean of per-timestep mean vehicle waiting times averaged across all evaluation timesteps.
        - mean_total_departures: Mean cumulative vehicle departures per episode across all evaluation episodes.
        - total_departures: Total vehicle departures summed across all evaluation episodes.
        - cumulative_delay: Mean cumulative vehicle delay per episode across all evaluation episodes.
        - overflow_rate: Fraction of evaluation episodes with at least one queue overflow event.
    """

    model_config = ConfigDict(extra="ignore")

    episodes: int = Field(..., gt=0, description="Total evaluation episodes executed")
    episode_return: Optional[float] = Field(
        None, description="Mean cumulative episodic return (RL only; null for classical planners)"
    )
    success_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes reaching goal or objective"
    )
    collision_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes ending in collision"
    )
    overflow_rate: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Fraction of episodes experiencing queue overflow (traffic only)",
    )
    truncation_rate: Optional[float] = Field(
        None, ge=0.0, le=1.0, description="Fraction of episodes truncated by max step limit"
    )
    episode_length: Optional[float] = Field(None, ge=0.0, description="Mean step count per episode")
    path_length: Optional[float] = Field(
        None,
        ge=0.0,
        description="Mean geometric path length (grid steps or Euclidean distance)",
    )
    path_efficiency: Optional[float] = Field(
        None,
        ge=0.0,
        le=1.0,
        description="Ratio of optimal/straight-line distance to actual path length",
    )
    planning_time: Optional[float] = Field(
        None, ge=0.0, description="Mean planning/search or inference wall-clock time in seconds"
    )
    generalization_gap: Optional[float] = Field(
        None, description="Performance gap between training and unseen test distributions"
    )
    battery_remaining: Optional[float] = Field(
        None, ge=0.0, description="Mean remaining battery level (drone environments)"
    )
    battery_used: Optional[float] = Field(
        None, ge=0.0, description="Mean battery energy consumed (drone environments)"
    )
    dynamic_collision_count: Optional[int] = Field(
        None, ge=0, description="Total dynamic obstacle collision events"
    )
    mean_queue_length: Optional[float] = Field(
        None,
        ge=0.0,
        description="Mean of per-timestep total queue lengths across all evaluation timesteps (traffic only)",
    )
    mean_max_wait_time: Optional[float] = Field(
        None,
        ge=0.0,
        description="Mean of per-episode maximum vehicle waiting times across all evaluation episodes (traffic only)",
    )
    mean_wait_time: Optional[float] = Field(
        None,
        ge=0.0,
        description="Mean of per-timestep mean vehicle waiting times across all evaluation timesteps (traffic only)",
    )
    mean_total_departures: Optional[float] = Field(
        None,
        ge=0.0,
        description="Mean cumulative vehicle departures per episode across all evaluation episodes (traffic only)",
    )
    total_departures: Optional[int] = Field(
        None,
        ge=0,
        description="Total vehicle departures summed across all evaluation episodes (traffic only)",
    )
    cumulative_delay: Optional[float] = Field(
        None,
        ge=0.0,
        description="Mean cumulative vehicle delay per episode across all evaluation episodes (traffic only)",
    )
    additional_metrics: Dict[str, Any] = Field(
        default_factory=dict, description="Arbitrary domain-specific metric dictionary"
    )

    @classmethod
    def from_rl_metrics(
        cls,
        eval_metrics: EvaluationMetrics,
        generalization_gap: Optional[float] = None,
    ) -> StandardizedExperimentMetrics:
        """Construct standardized metrics from RL EvaluationMetrics."""
        extra = eval_metrics.additional_metrics

        def _first_not_none(*values: Any) -> Any:
            for val in values:
                if val is not None:
                    return val
            return None

        # Extract traffic-specific aggregates
        mean_q = _first_not_none(extra.get("mean_queue_length"), extra.get("mean_queue"))
        mean_mw = _first_not_none(extra.get("mean_max_wait_time"), extra.get("mean_max_wait"))
        mean_w = _first_not_none(extra.get("mean_wait_time"), extra.get("mean_wait"))
        mean_dep = _first_not_none(extra.get("mean_total_departures"), extra.get("mean_departures"))
        tot_dep = extra.get("total_departures")
        cum_del = _first_not_none(extra.get("cumulative_delay"), extra.get("mean_cumulative_delay"))
        overflow_r = _first_not_none(eval_metrics.overflow_rate, extra.get("overflow_rate"))
        truncation_r = _first_not_none(eval_metrics.truncation_rate, extra.get("truncation_rate"))

        handled_keys = {
            "mean_path_length",
            "path_length",
            "path_efficiency",
            "mean_planning_time",
            "planning_time",
            "generalization_gap",
            "mean_battery_remaining",
            "battery_remaining",
            "mean_battery_used",
            "battery_used",
            "dynamic_collision_count",
            "mean_queue_length",
            "mean_queue",
            "mean_max_wait_time",
            "mean_max_wait",
            "mean_wait_time",
            "mean_wait",
            "mean_total_departures",
            "mean_departures",
            "total_departures",
            "cumulative_delay",
            "mean_cumulative_delay",
            "overflow_rate",
            "truncation_rate",
        }

        return cls(
            episodes=eval_metrics.episodes,
            episode_return=eval_metrics.mean_reward,
            success_rate=eval_metrics.success_rate,
            collision_rate=eval_metrics.collision_rate,
            overflow_rate=overflow_r,
            truncation_rate=truncation_r,
            episode_length=eval_metrics.mean_episode_length,
            path_length=_first_not_none(extra.get("mean_path_length"), extra.get("path_length")),
            path_efficiency=extra.get("path_efficiency"),
            planning_time=_first_not_none(
                extra.get("mean_planning_time"), extra.get("planning_time")
            ),
            generalization_gap=_first_not_none(generalization_gap, extra.get("generalization_gap")),
            battery_remaining=_first_not_none(
                extra.get("mean_battery_remaining"), extra.get("battery_remaining")
            ),
            battery_used=_first_not_none(extra.get("mean_battery_used"), extra.get("battery_used")),
            dynamic_collision_count=extra.get("dynamic_collision_count"),
            mean_queue_length=mean_q,
            mean_max_wait_time=mean_mw,
            mean_wait_time=mean_w,
            mean_total_departures=mean_dep,
            total_departures=tot_dep,
            cumulative_delay=cum_del,
            additional_metrics={k: v for k, v in extra.items() if k not in handled_keys},
        )

    @classmethod
    def from_planner_metrics(
        cls,
        planner_metrics: Any,
    ) -> StandardizedExperimentMetrics:
        """Construct standardized metrics from PlannerEvaluationMetrics."""
        extra = getattr(planner_metrics, "additional_metrics", {})
        mean_path = getattr(planner_metrics, "mean_path_length", None)
        mean_time = getattr(planner_metrics, "mean_planning_time", None)

        return cls(
            episodes=getattr(planner_metrics, "episodes", 1),
            episode_return=None,  # Classical planners do not accumulate RL reward returns
            success_rate=getattr(planner_metrics, "success_rate", None),
            collision_rate=getattr(planner_metrics, "collision_rate", None),
            overflow_rate=None,
            truncation_rate=None,
            episode_length=None,  # Independent: RL step count is not applicable to geometric paths
            path_length=mean_path,
            path_efficiency=extra.get("path_efficiency"),
            planning_time=mean_time,
            generalization_gap=None,
            battery_remaining=None,
            battery_used=None,
            dynamic_collision_count=None,
            additional_metrics=extra,
        )

    def to_csv_dict(self) -> Dict[str, Any]:
        """Convert flat scalar fields to a dictionary suitable for CSV serialization."""
        data = self.model_dump(exclude={"additional_metrics"})
        flat: Dict[str, Any] = {}
        for k, v in data.items():
            if v is not None:
                flat[k] = v
            else:
                flat[k] = ""
        return flat


__all__ = [
    "EvaluationMetrics",
    "StandardizedExperimentMetrics",
]
