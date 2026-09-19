"""Canonical episode metrics data structure, accumulator, and extraction contract.

Defines the single source of truth for episodic performance outcomes across
reinforcement learning algorithms, classical planners, and evaluation routines.
"""

from __future__ import annotations

from collections.abc import Mapping, Sequence
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, SupportsFloat

# Canonical precedence order for episode outcome fields.
SUCCESS_KEYS: Sequence[str] = ("episode_success", "success", "is_success")
COLLISION_KEYS: Sequence[str] = ("collision", "is_collision", "had_collision")
OVERFLOW_KEYS: Sequence[str] = ("overflow", "step_overflow", "had_overflow")
TRAFFIC_KEYS: Sequence[str] = (
    "queue_lengths",
    "total_queue",
    "step_overflow",
    "had_overflow",
    "step_controlled",
    "phase_switched",
    "step_departures",
    "step_arrivals",
    "premature_switch",
)

_EXCLUDED_OUTCOME_KEYS = frozenset(SUCCESS_KEYS).union(COLLISION_KEYS)


def _extract_flag(
    info: Optional[Mapping[str, Any]],
    keys: Sequence[str],
) -> Optional[bool]:
    """Extract a boolean outcome flag following strict precedence.

    Rules:
    - Candidate keys are checked in strict sequence order.
    - If a key is present and its value is not None, bool(value) is returned.
    - Explicit False remains False.
    - If no candidate key is present, or all candidates are None, None is returned.
    """
    if not info:
        return None

    for key in keys:
        if key in info and info[key] is not None:
            return bool(info[key])

    return None


@dataclass(frozen=True)
class EpisodeMetrics:
    """Immutable, typed container representing one canonical episode outcome.

    Attributes:
        reward: Total cumulative episodic reward.
        length: Total timesteps elapsed in the episode.
        success: True if successful, False if explicitly unsuccessful,
            or None if the metric is undefined/unavailable.
        collision: True if a collision occurred, False if explicitly
            collision-free, or None if the metric is undefined/unavailable.
        terminated: True if the episode ended via natural termination.
        truncated: True if the episode ended via truncation/timeout.
        additional_metrics: Environment-specific metrics.
    """

    reward: float
    length: int
    success: Optional[bool]
    collision: Optional[bool]
    terminated: bool
    truncated: bool
    additional_metrics: Mapping[str, Any] = field(default_factory=dict)

    def __post_init__(self) -> None:
        """Validate and normalize values while preserving nullable semantics."""
        if not isinstance(self.reward, float):
            object.__setattr__(self, "reward", float(self.reward))

        if not isinstance(self.length, int):
            object.__setattr__(self, "length", int(self.length))

        if self.success is not None and not isinstance(self.success, bool):
            object.__setattr__(self, "success", bool(self.success))

        if self.collision is not None and not isinstance(self.collision, bool):
            object.__setattr__(self, "collision", bool(self.collision))

        if not isinstance(self.terminated, bool):
            object.__setattr__(self, "terminated", bool(self.terminated))

        if not isinstance(self.truncated, bool):
            object.__setattr__(self, "truncated", bool(self.truncated))

        if not isinstance(self.additional_metrics, Mapping):
            raise TypeError(
                "additional_metrics must be a Mapping, "
                f"got {type(self.additional_metrics).__name__}"
            )

        object.__setattr__(
            self,
            "additional_metrics",
            dict(self.additional_metrics),
        )

    def to_dict(self) -> Dict[str, Any]:
        """Serialize episode metrics to a dictionary."""
        return {
            "reward": self.reward,
            "length": self.length,
            "success": self.success,
            "collision": self.collision,
            "terminated": self.terminated,
            "truncated": self.truncated,
            "additional_metrics": dict(self.additional_metrics),
        }


class OutcomePolicy:
    """Stateless strategy for interpreting episode-local facts.

    EpisodeMetricsAccumulator owns all episode-local state.

    OutcomePolicy instances MUST NOT store per-episode facts. They may be
    safely shared across sequential episodes, vectorized environments, and
    concurrent accumulators. Constructor attributes, if any, should contain
    immutable configuration only.

    Policy selection belongs to the accumulator and is immutable once the
    first transition has been recorded.
    """

    __slots__ = ()

    def resolve_collision(
        self,
        accumulator: EpisodeMetricsAccumulator,
    ) -> Optional[bool]:
        """Resolve the final collision outcome."""
        if not accumulator.has_collision_info:
            return None

        return accumulator.had_collision

    def resolve_success(
        self,
        accumulator: EpisodeMetricsAccumulator,
    ) -> Optional[bool]:
        """Resolve the final success outcome."""
        if accumulator.explicit_success is not None:
            return accumulator.explicit_success

        if not accumulator.has_success_info:
            return None

        terminal_success = _extract_flag(
            accumulator.last_info,
            SUCCESS_KEYS,
        )

        if terminal_success is not None:
            return terminal_success

        return accumulator.had_success

    def get_additional_metrics(
        self,
        accumulator: EpisodeMetricsAccumulator,
    ) -> Dict[str, Any]:
        """Return policy-specific additional metrics."""
        return {}


class DefaultOutcomePolicy(OutcomePolicy):
    """Standard goal-directed outcome policy.

    Intended for navigation, continuous-control, and gridworld-style
    environments.

    Rules:
    - No success telemetry anywhere -> success=None.
    - Explicit terminal success takes precedence.
    - Otherwise an observed intermediate success is preserved.
    - Collision is handled by the accumulator's universal invariant and
      overrides positive success at finish time.
    """

    __slots__ = ()


class TrafficOutcomePolicy(OutcomePolicy):
    """Traffic-domain outcome policy.

    Traffic success is established from episode-local facts according to:

    1. Any queue overflow -> success=False.
    2. Natural termination -> success=False.
    3. The terminal success flag is authoritative.
    4. If terminal success is unavailable and no failure occurred -> None.
    5. Intermediate success never latches into the final traffic outcome.

    The policy is stateless and safe to share between accumulators.
    """

    __slots__ = ()

    def resolve_success(
        self,
        accumulator: EpisodeMetricsAccumulator,
    ) -> Optional[bool]:
        """Resolve traffic success strictly from episode-local facts."""
        # 1. Overflow anywhere in the episode is a hard failure.
        if accumulator.had_overflow:
            return False

        # 2. Natural termination represents premature traffic failure.
        # Horizon completion should normally be represented by truncation.
        if accumulator.terminated and not accumulator.truncated:
            return False

        # 3. Terminal success is authoritative for traffic environments.
        terminal_success = _extract_flag(
            accumulator.last_info,
            SUCCESS_KEYS,
        )
        if terminal_success is not None:
            return terminal_success

        # 4. Explicit episode-level success override remains supported.
        if accumulator.explicit_success is not None:
            return accumulator.explicit_success

        # 5. No terminal success information means undefined.
        return None

    def get_additional_metrics(
        self,
        accumulator: EpisodeMetricsAccumulator,
    ) -> Dict[str, Any]:
        """Expose traffic-specific overflow information."""
        if not accumulator.has_overflow_info:
            return {}

        return {
            "had_overflow": accumulator.had_overflow,
        }


class EpisodeMetricsAccumulator:
    """Collect episode facts and produce the canonical EpisodeMetrics object.

    Architecture:

        raw environment steps
            ↓
        EpisodeMetricsAccumulator
            ↓
        OutcomePolicy
            ↓
        EpisodeMetrics

    The accumulator owns episode-local facts.
    The policy interprets those facts.
    EpisodeMetrics is the immutable canonical result.

    A policy cannot be replaced after the first record_step() call.
    """

    def __init__(
        self,
        outcome_policy: Optional[OutcomePolicy] = None,
        *,
        is_traffic: Optional[bool] = None,
    ) -> None:
        """Initialize an episode accumulator.

        Args:
            outcome_policy:
                Explicit domain outcome policy.
            is_traffic:
                Legacy compatibility selector. When supplied together with
                outcome_policy, it must agree with the policy type.
        """
        if outcome_policy is not None and is_traffic is not None:
            policy_is_traffic = isinstance(
                outcome_policy,
                TrafficOutcomePolicy,
            )

            if bool(is_traffic) != policy_is_traffic:
                raise ValueError(
                    "Conflicting configuration: "
                    f"outcome_policy={type(outcome_policy).__name__} "
                    f"is incompatible with is_traffic={is_traffic!r}."
                )

        if outcome_policy is not None:
            self._outcome_policy: OutcomePolicy = outcome_policy
            self._explicit_policy = True
        elif is_traffic is not None:
            self._outcome_policy = TrafficOutcomePolicy() if is_traffic else DefaultOutcomePolicy()
            self._explicit_policy = True
        else:
            self._outcome_policy = DefaultOutcomePolicy()
            self._explicit_policy = False

        self._policy_locked = False

        self.reward: float = 0.0
        self.length: int = 0
        self.terminated: bool = False
        self.truncated: bool = False

        self.has_collision_info: bool = False
        self.had_collision: bool = False

        self.has_success_info: bool = False
        self.had_success: bool = False
        self.explicit_success: Optional[bool] = None

        self.has_overflow_info: bool = False
        self.had_overflow: bool = False

        self.last_info: Dict[str, Any] = {}
        self.step_infos: List[Mapping[str, Any]] = []

    def _ensure_policy_mutable(self) -> None:
        """Reject policy replacement after episode processing begins."""
        if self._policy_locked:
            raise RuntimeError(
                "Cannot change OutcomePolicy after the first record_step() "
                "of an episode. Policy selection is immutable for the "
                "lifetime of an accumulator."
            )

    @property
    def outcome_policy(self) -> OutcomePolicy:
        """Return the policy bound to this episode."""
        return self._outcome_policy

    @outcome_policy.setter
    def outcome_policy(self, policy: OutcomePolicy) -> None:
        """Set the policy before episode processing begins."""
        if not isinstance(policy, OutcomePolicy):
            raise TypeError(
                f"outcome_policy must be an OutcomePolicy instance, got {type(policy).__name__}"
            )

        if policy is self._outcome_policy:
            return

        self._ensure_policy_mutable()
        self._outcome_policy = policy

    @property
    def is_traffic(self) -> bool:
        """Compatibility property indicating traffic policy selection."""
        return isinstance(
            self._outcome_policy,
            TrafficOutcomePolicy,
        )

    @is_traffic.setter
    def is_traffic(self, value: Optional[bool]) -> None:
        """Set legacy traffic policy selection before episode processing."""
        if value is None:
            return

        requested_is_traffic = bool(value)
        current_is_traffic = self.is_traffic

        # No semantic change requested.
        if requested_is_traffic == current_is_traffic:
            return

        self._ensure_policy_mutable()

        # Explicit policy selection cannot be overridden by legacy config.
        if self._explicit_policy:
            raise ValueError(
                f"Cannot set is_traffic={requested_is_traffic} because an explicit "
                f"outcome_policy={type(self._outcome_policy).__name__} "
                "was provided."
            )

        self._outcome_policy = (
            TrafficOutcomePolicy() if requested_is_traffic else DefaultOutcomePolicy()
        )

    def record_step(
        self,
        reward: float | SupportsFloat = 0.0,
        terminated: bool = False,
        truncated: bool = False,
        info: Optional[Mapping[str, Any]] = None,
    ) -> None:
        """Record one environment transition.

        Policy selection may be automatically upgraded to TrafficOutcomePolicy
        only on the first step when no explicit policy was configured.
        """
        info_dict = dict(info) if info is not None else {}

        # Legacy domain detection is allowed exactly once, before the
        # accumulator becomes policy-locked.
        if (
            not self._policy_locked
            and not self._explicit_policy
            and isinstance(self._outcome_policy, DefaultOutcomePolicy)
            and any(key in info_dict for key in TRAFFIC_KEYS)
        ):
            self._outcome_policy = TrafficOutcomePolicy()

        # Once the first transition is processed, policy selection is frozen.
        self._policy_locked = True

        self.reward += float(reward)
        self.length += 1
        self.terminated = bool(terminated)
        self.truncated = bool(truncated)

        self.last_info = info_dict
        self.step_infos.append(info_dict)

        # Collision facts are episode-local and monotonic.
        collision_flag = _extract_flag(
            info_dict,
            COLLISION_KEYS,
        )
        if collision_flag is not None:
            self.has_collision_info = True
            if collision_flag:
                self.had_collision = True

        # Success facts are episode-local and monotonic.
        success_flag = _extract_flag(
            info_dict,
            SUCCESS_KEYS,
        )
        if success_flag is not None:
            self.has_success_info = True
            if success_flag:
                self.had_success = True

        # Overflow facts are episode-local and monotonic.
        overflow_flag = _extract_flag(
            info_dict,
            OVERFLOW_KEYS,
        )
        if overflow_flag is not None:
            self.has_overflow_info = True
            if overflow_flag:
                self.had_overflow = True

    def finish(
        self,
        *,
        additional_metrics: Optional[Mapping[str, Any]] = None,
    ) -> EpisodeMetrics:
        """Finalize the episode into immutable canonical metrics."""
        collision = self.outcome_policy.resolve_collision(self)
        success = self.outcome_policy.resolve_success(self)

        # Universal invariant:
        # A collision always overrides a positive success result.
        if collision is True and success is True:
            success = False

        if additional_metrics is not None:
            extra: Dict[str, Any] = dict(additional_metrics)
        else:
            extra = {
                key: value
                for key, value in self.last_info.items()
                if key not in _EXCLUDED_OUTCOME_KEYS
            }

        policy_extra = self.outcome_policy.get_additional_metrics(self)

        for key, value in policy_extra.items():
            extra.setdefault(key, value)

        return EpisodeMetrics(
            reward=self.reward,
            length=self.length,
            success=success,
            collision=collision,
            terminated=self.terminated,
            truncated=self.truncated,
            additional_metrics=extra,
        )


def compute_rate(
    values: Sequence[Optional[bool]],
) -> Optional[float]:
    """Compute True rate among defined boolean outcomes.

    Canonical denominator semantics:
    - True contributes 1.
    - False contributes 0.
    - None is excluded.
    - If no defined values exist, return None.

    Examples:
        >>> compute_rate([True, False, None])
        0.5
        >>> compute_rate([None, None])
        >>> compute_rate([True, True])
        1.0
        >>> compute_rate([False, False])
        0.0
    """
    defined = [value for value in values if value is not None]

    if not defined:
        return None

    return float(sum(value is True for value in defined) / len(defined))


def _has_traffic_telemetry(info: Mapping[str, Any]) -> bool:
    """Return whether an info mapping contains traffic telemetry."""
    return any(key in info for key in TRAFFIC_KEYS)


def _resolve_episode_step_infos(
    info: Optional[Mapping[str, Any]],
    step_infos: Optional[Sequence[Mapping[str, Any]]],
) -> List[Mapping[str, Any]]:
    """Resolve a single, ordered episode info sequence.

    Contract:
    - step_infos is an ordered sequence of per-step Gymnasium info dicts.
    - If only info is provided, info is treated as the terminal step.
    - If only step_infos is provided, it is treated as the complete episode trace.
    - If both are provided, info is appended only when it is not already
      represented by object identity at the end of step_infos (i.e.
      `step_infos[-1] is info`). Object identity avoids unsafe equality
      comparisons over arbitrary telemetry payloads (e.g. NumPy arrays
      or non-boolean equality predicates).
    """
    resolved: List[Mapping[str, Any]] = list(step_infos or [])

    if info is None:
        return resolved

    if resolved and resolved[-1] is info:
        return resolved

    resolved.append(info)
    return resolved


def extract_episode_metrics(
    reward: float | SupportsFloat,
    length: int,
    terminated: bool,
    truncated: bool,
    info: Optional[Mapping[str, Any]] = None,
    *,
    additional_metrics: Optional[Mapping[str, Any]] = None,
    step_infos: Optional[Sequence[Mapping[str, Any]]] = None,
    had_collision: Optional[bool] = None,
    had_success: Optional[bool] = None,
    had_overflow: Optional[bool] = None,
    is_traffic: Optional[bool] = None,
    outcome_policy: Optional[OutcomePolicy] = None,
) -> EpisodeMetrics:
    """Create canonical EpisodeMetrics through the accumulator.

    This function is intentionally a thin compatibility/convenience layer.
    All outcome derivation is delegated to EpisodeMetricsAccumulator and its
    bound OutcomePolicy.

    If both `outcome_policy` and `is_traffic` are provided, they must agree.
    If neither is provided, traffic telemetry is detected from the resolved
    episode trace before the accumulator is created, so policy selection is
    fixed before replay begins.

    Explicit overrides:
    - had_collision=True marks collision as observed.
    - had_collision=False only provides an explicit collision-free observation
      when no actual collision has already been observed.
    - had_success sets explicit episode success.
    - had_overflow=True marks overflow as observed.
    """
    resolved_infos = _resolve_episode_step_infos(
        info=info,
        step_infos=step_infos,
    )

    resolved_is_traffic = is_traffic

    if outcome_policy is None and resolved_is_traffic is None:
        if any(_has_traffic_telemetry(step_info) for step_info in resolved_infos):
            resolved_is_traffic = True

    accumulator = EpisodeMetricsAccumulator(
        outcome_policy=outcome_policy,
        is_traffic=resolved_is_traffic,
    )

    for step_info in resolved_infos:
        accumulator.record_step(info=step_info)

    # Explicit episode-level fields are applied after replaying raw facts.
    accumulator.reward = float(reward)
    accumulator.length = int(length)
    accumulator.terminated = bool(terminated)
    accumulator.truncated = bool(truncated)

    if had_collision is not None:
        accumulator.has_collision_info = True

        # Never erase a collision observed on an actual step.
        if had_collision:
            accumulator.had_collision = True

    if had_success is not None:
        accumulator.has_success_info = True
        accumulator.had_success = bool(had_success)
        accumulator.explicit_success = bool(had_success)

    if had_overflow is not None:
        accumulator.has_overflow_info = True

        # Overflow is monotonic; an explicit False cannot erase an observed
        # overflow from the episode trace.
        if had_overflow:
            accumulator.had_overflow = True

    return accumulator.finish(
        additional_metrics=additional_metrics,
    )


__all__ = [
    "COLLISION_KEYS",
    "DefaultOutcomePolicy",
    "EpisodeMetrics",
    "EpisodeMetricsAccumulator",
    "OVERFLOW_KEYS",
    "OutcomePolicy",
    "SUCCESS_KEYS",
    "TRAFFIC_KEYS",
    "TrafficOutcomePolicy",
    "compute_rate",
    "extract_episode_metrics",
]
