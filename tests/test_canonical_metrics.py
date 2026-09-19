"""Regression and contract tests for the canonical EpisodeMetrics extraction and consumers."""

from __future__ import annotations

import csv
import json
from dataclasses import FrozenInstanceError, is_dataclass
from pathlib import Path
from typing import Any, Dict, Optional

import gymnasium as gym
import numpy as np
import pytest
from gymnasium import spaces

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.evaluation.evaluator import Evaluator
from adaptive_rl.evaluation.generalization import (
    GeneralizationDistribution,
    GeneralizationEvaluator,
)
from adaptive_rl.metrics import (
    EpisodeMetrics,
    EpisodeMetricsAccumulator,
    extract_episode_metrics,
)
from adaptive_rl.training.callbacks import (
    MetricLoggerCallback,
    SB3CallbackAdapter,
)


class MockPolicyAlgo(BaseAlgorithm):
    """Minimal policy algorithm returning deterministic discrete or continuous zero actions."""

    def __init__(self, action_space: Optional[gym.Space] = None) -> None:
        self.action_space = action_space
        self.name = "mock_policy"

    def predict(self, observation: Any, deterministic: bool = True) -> tuple[Any, Any]:
        if self.action_space is not None and hasattr(self.action_space, "shape"):
            shape = getattr(self.action_space, "shape", ())
            return np.zeros(shape, dtype=np.float32), {}
        return 0, {}

    def train(self, total_timesteps: int, callback: Any = None) -> None:
        pass

    def save(self, path: Any) -> None:
        pass

    def load(self, path: Any, env: Optional[gym.Env] = None) -> None:
        pass


class MockStepEnv(gym.Env):
    """Scriptable environment returning programmed step outcomes."""

    def __init__(self, step_outcomes: list[tuple[float, bool, bool, Dict[str, Any]]]) -> None:
        super().__init__()
        self.observation_space = spaces.Box(low=-1.0, high=1.0, shape=(2,), dtype=np.float32)
        self.action_space = spaces.Discrete(2)
        self.step_outcomes = list(step_outcomes)
        self.step_idx = 0

    def reset(
        self,
        *,
        seed: Optional[int] = None,
        options: Optional[dict[str, Any]] = None,
    ) -> tuple[np.ndarray, dict[str, Any]]:
        super().reset(seed=seed)
        return np.zeros(2, dtype=np.float32), {}

    def step(self, action: Any) -> tuple[np.ndarray, float, bool, bool, dict[str, Any]]:
        if self.step_idx < len(self.step_outcomes):
            rew, term, trunc, info = self.step_outcomes[self.step_idx]
            self.step_idx += 1
        else:
            rew, term, trunc, info = 0.0, True, False, {}
        return np.zeros(2, dtype=np.float32), rew, term, trunc, dict(info)


# ---------------------------------------------------------------------------
# Core Dataclass & Precedence Unit Tests
# ---------------------------------------------------------------------------


def test_episode_metrics_is_immutable_dataclass() -> None:
    """EpisodeMetrics must be an immutable frozen dataclass."""
    m = EpisodeMetrics(
        reward=10.0,
        length=5,
        success=True,
        collision=False,
        terminated=True,
        truncated=False,
        additional_metrics={"step": 5},
    )
    assert is_dataclass(m)
    with pytest.raises(FrozenInstanceError):
        m.reward = 20.0  # type: ignore[misc,union-attr]

    with pytest.raises(FrozenInstanceError):
        m.success = False  # type: ignore[misc,union-attr]


def test_episode_metrics_mutation_isolation() -> None:
    """Mutating caller's dictionary after extraction must not alter EpisodeMetrics."""
    raw_info = {"battery": 80.0, "success": True}
    m = extract_episode_metrics(
        reward=5.0,
        length=10,
        terminated=True,
        truncated=False,
        info=raw_info,
    )
    assert m.additional_metrics["battery"] == 80.0
    raw_info["battery"] = 10.0
    assert m.additional_metrics["battery"] == 80.0


def test_episode_success_true() -> None:
    """episode_success=True extracts as success=True."""
    m = extract_episode_metrics(
        reward=1.0,
        length=1,
        terminated=True,
        truncated=False,
        info={"episode_success": True},
    )
    assert m.success is True


def test_success_false() -> None:
    """Explicit success=False must remain False."""
    m = extract_episode_metrics(
        reward=0.0,
        length=10,
        terminated=True,
        truncated=False,
        info={"success": False},
    )
    assert m.success is False


def test_missing_success_returns_none() -> None:
    """Missing success information must remain None, never coerced to False."""
    m1 = extract_episode_metrics(1.0, 5, True, False, info={})
    m2 = extract_episode_metrics(1.0, 5, True, False, info=None)
    m3 = extract_episode_metrics(1.0, 5, True, False, info={"battery": 50})
    assert m1.success is None
    assert m2.success is None
    assert m3.success is None


def test_episode_success_takes_precedence_over_success() -> None:
    """Precedence MUST be episode_success -> success -> is_success."""
    # episode_success=True overrides success=False
    m1 = extract_episode_metrics(
        1.0,
        5,
        True,
        False,
        info={"episode_success": True, "success": False, "is_success": False},
    )
    assert m1.success is True

    # episode_success=False overrides success=True (explicit False remains False)
    m2 = extract_episode_metrics(
        1.0,
        5,
        True,
        False,
        info={"episode_success": False, "success": True, "is_success": True},
    )
    assert m2.success is False

    # success overrides is_success when episode_success is absent
    m3 = extract_episode_metrics(
        1.0,
        5,
        True,
        False,
        info={"success": True, "is_success": False},
    )
    assert m3.success is True

    m4 = extract_episode_metrics(
        1.0,
        5,
        True,
        False,
        info={"success": False, "is_success": True},
    )
    assert m4.success is False

    # is_success used when others are absent
    m5 = extract_episode_metrics(1.0, 5, True, False, info={"is_success": True})
    assert m5.success is True

    m6 = extract_episode_metrics(1.0, 5, True, False, info={"is_success": False})
    assert m6.success is False


def test_collision_true() -> None:
    """collision=True extracts as collision=True."""
    m = extract_episode_metrics(
        -10.0,
        3,
        True,
        False,
        info={"collision": True},
    )
    assert m.collision is True


def test_missing_collision_returns_none() -> None:
    """Missing collision information must remain None, never coerced to False."""
    m1 = extract_episode_metrics(0.0, 5, True, False, info={})
    m2 = extract_episode_metrics(0.0, 5, True, False, info=None)
    m3 = extract_episode_metrics(0.0, 5, True, False, info={"other": 1})
    assert m1.collision is None
    assert m2.collision is None
    assert m3.collision is None


def test_collision_alias_precedence() -> None:
    """Collision precedence MUST be collision -> is_collision -> had_collision."""
    # collision=True overrides others
    m1 = extract_episode_metrics(
        0.0,
        5,
        True,
        False,
        info={"collision": True, "is_collision": False, "had_collision": False},
    )
    assert m1.collision is True

    # collision=False overrides others (explicit False remains False)
    m2 = extract_episode_metrics(
        0.0,
        5,
        True,
        False,
        info={"collision": False, "is_collision": True, "had_collision": True},
    )
    assert m2.collision is False

    # is_collision overrides had_collision
    m3 = extract_episode_metrics(
        0.0,
        5,
        True,
        False,
        info={"is_collision": True, "had_collision": False},
    )
    assert m3.collision is True

    m4 = extract_episode_metrics(
        0.0,
        5,
        True,
        False,
        info={"is_collision": False, "had_collision": True},
    )
    assert m4.collision is False

    # had_collision used when others are absent
    m5 = extract_episode_metrics(0.0, 5, True, False, info={"had_collision": True})
    assert m5.collision is True

    m6 = extract_episode_metrics(0.0, 5, True, False, info={"had_collision": False})
    assert m6.collision is False


def test_terminated_truncated_preserved_exactly() -> None:
    """terminated and truncated flags must be preserved exactly from arguments."""
    for term in (True, False):
        for trunc in (True, False):
            m = extract_episode_metrics(
                reward=10.0,
                length=5,
                terminated=term,
                truncated=trunc,
                info={"TimeLimit.truncated": not trunc},  # Ignored: info must not override
            )
            assert m.terminated is term
            assert m.truncated is trunc


def test_environment_specific_metrics_preserved_in_additional_metrics() -> None:
    """Environment-specific telemetry must be retained in additional_metrics."""
    info = {
        "battery_remaining": 45.2,
        "traffic_departures": 18,
        "wind_vector": (1.0, -0.5, 0.0),
        "success": True,
        "collision": False,
        "episode_success": True,
    }
    m = extract_episode_metrics(100.0, 20, True, False, info=info)
    assert m.additional_metrics["battery_remaining"] == 45.2
    assert m.additional_metrics["traffic_departures"] == 18
    assert m.additional_metrics["wind_vector"] == (1.0, -0.5, 0.0)
    # Core outcome keys should not clutter additional_metrics
    assert "success" not in m.additional_metrics
    assert "collision" not in m.additional_metrics
    assert "episode_success" not in m.additional_metrics


def test_explicit_false_distinguishable_from_missing_none() -> None:
    """Verify explicit False is distinct from missing None across success and collision."""
    m_false = extract_episode_metrics(
        1.0,
        5,
        True,
        False,
        info={"success": False, "collision": False},
    )
    m_none = extract_episode_metrics(1.0, 5, True, False, info={})

    assert m_false.success is False
    assert m_false.collision is False
    assert m_none.success is None
    assert m_none.collision is None

    assert m_false.success is not m_none.success
    assert m_false.collision is not m_none.collision


def test_no_heuristic_inference() -> None:
    """Do not infer success/collision from rewards, termination status, or episode length."""
    # Huge positive reward and natural termination without success info remains None
    m1 = extract_episode_metrics(1000.0, 5, terminated=True, truncated=False, info={})
    assert m1.success is None
    assert m1.collision is None

    # Huge negative penalty without collision info remains None
    m2 = extract_episode_metrics(-500.0, 1, terminated=True, truncated=False, info={})
    assert m2.success is None
    assert m2.collision is None

    # Max step truncation without success info remains None
    m3 = extract_episode_metrics(0.0, 1000, terminated=False, truncated=True, info={})
    assert m3.success is None
    assert m3.collision is None


# ---------------------------------------------------------------------------
# Training Integration Unit Tests (PPO & SAC)
# ---------------------------------------------------------------------------


def test_ppo_consumes_canonical_episode_metrics() -> None:
    """Verify SB3CallbackAdapter extracts EpisodeMetrics and MetricLoggerCallback consumes them."""
    logger = MetricLoggerCallback()
    adapter = SB3CallbackAdapter(callbacks=[logger])

    # Simulate an episode completion step from Stable-Baselines3
    adapter.locals = {
        "dones": [True],
        "rewards": [15.0],
        "infos": [{"success": True, "collision": False, "custom_metric": 42}],
    }
    adapter.num_timesteps = 10

    # Execute step hook
    adapter._on_step()

    assert logger.total_episodes == 1
    assert len(logger.episode_metrics) == 1
    m = logger.episode_metrics[0]
    assert isinstance(m, EpisodeMetrics)
    assert m.reward == 15.0
    assert m.length == 1
    assert m.success is True
    assert m.collision is False
    assert m.additional_metrics["custom_metric"] == 42
    assert logger.successes == 1
    assert logger.collisions == 0


def test_sac_consumes_canonical_episode_metrics() -> None:
    """Verify SACTrainer and SB3CallbackAdapter extract and consume canonical EpisodeMetrics."""
    logger = MetricLoggerCallback()
    adapter = SB3CallbackAdapter(callbacks=[logger])

    # Simulate multi-step SAC episode ending in collision
    adapter.locals = {
        "dones": [False],
        "rewards": [1.0],
        "infos": [{}],
    }
    adapter.num_timesteps = 1
    adapter._on_step()

    adapter.locals = {
        "dones": [True],
        "rewards": [-10.0],
        "infos": [{"collision": True, "success": False}],
    }
    adapter.num_timesteps = 2
    adapter._on_step()

    assert logger.total_episodes == 1
    assert len(logger.episode_metrics) == 1
    m = logger.episode_metrics[0]
    assert isinstance(m, EpisodeMetrics)
    assert m.reward == -9.0
    assert m.length == 2
    assert m.success is False
    assert m.collision is True
    assert logger.successes == 0
    assert logger.collisions == 1


# ---------------------------------------------------------------------------
# Evaluator & GeneralizationEvaluator Integration Tests
# ---------------------------------------------------------------------------


def test_evaluator_consumes_canonical_episode_metrics() -> None:
    """Verify Evaluator creates canonical EpisodeMetrics and aggregates them accurately."""
    # Episode 1: Success (reward 10, length 2)
    # Episode 2: Collision (reward -5, length 1)
    step_trace = [
        # Episode 1: step 0, then step 1 (terminal success)
        (5.0, False, False, {}),
        (5.0, True, False, {"success": True, "collision": False}),
        # Episode 2: step 0 (terminal collision)
        (-5.0, True, False, {"collision": True, "success": False}),
    ]
    env = MockStepEnv(step_trace)
    algo = MockPolicyAlgo()
    evaluator = Evaluator(algorithm=algo, env=env)

    metrics = evaluator.evaluate(num_episodes=2, deterministic=True)

    # Evaluator records EpisodeMetrics objects
    assert len(evaluator.last_episode_metrics) == 2
    assert all(isinstance(m, EpisodeMetrics) for m in evaluator.last_episode_metrics)

    m1, m2 = evaluator.last_episode_metrics
    assert m1.reward == 10.0
    assert m1.length == 2
    assert m1.success is True
    assert m1.collision is False

    assert m2.reward == -5.0
    assert m2.length == 1
    assert m2.success is False
    assert m2.collision is True

    # Aggregated EvaluationMetrics must reflect the canonical metrics
    assert metrics.episodes == 2
    assert metrics.mean_reward == 2.5  # (10 + -5) / 2
    assert metrics.success_rate == 0.5
    assert metrics.collision_rate == 0.5
    env.close()


def test_evaluator_missing_metrics_semantics() -> None:
    """Verify Evaluator preserves None when metrics are unavailable."""
    # Environment with no success or collision keys (e.g. classical continuous control)
    step_trace = [
        (1.0, True, False, {"arbitrary": "value"}),
    ]
    env = MockStepEnv(step_trace)
    algo = MockPolicyAlgo()
    evaluator = Evaluator(algorithm=algo, env=env)

    metrics = evaluator.evaluate(num_episodes=1, deterministic=True)
    m = evaluator.last_episode_metrics[0]
    assert m.success is None
    assert m.collision is None

    # Rates must be None, NOT collapsed to 0.0
    assert metrics.success_rate is None
    assert metrics.collision_rate is None
    env.close()


def test_generalization_evaluator_consumes_canonical_episode_metrics() -> None:
    """Verify GeneralizationEvaluator creates and aggregates canonical EpisodeMetrics."""
    # Distribution of 2 train seeds and 2 test seeds
    distribution = GeneralizationDistribution(
        train_seeds=[10, 11],
        test_seeds=[20, 21],
    )

    # Step trace: each episode is 1 step
    # Seed 10: success
    # Seed 11: collision
    # Seed 20: success
    # Seed 21: failure (explicit False, no collision)
    step_trace = [
        (10.0, True, False, {"success": True, "collision": False}),
        (-10.0, True, False, {"collision": True, "success": False}),
        (10.0, True, False, {"success": True, "collision": False}),
        (0.0, True, False, {"success": False, "collision": False}),
    ]
    env = MockStepEnv(step_trace)
    algo = MockPolicyAlgo()
    gen_eval = GeneralizationEvaluator(algorithm=algo, env=env)

    report = gen_eval.evaluate_generalization(
        distribution=distribution,
        experiment_name="gen_test",
        deterministic=True,
    )

    assert len(gen_eval.last_train_metrics) == 2
    assert len(gen_eval.last_test_metrics) == 2
    assert all(isinstance(m, EpisodeMetrics) for m in gen_eval.last_train_metrics)
    assert all(isinstance(m, EpisodeMetrics) for m in gen_eval.last_test_metrics)

    # Train: 1 success, 1 collision -> success_rate = 0.5, collision_rate = 0.5
    assert report.train_metrics.success_rate == 0.5
    assert report.train_metrics.collision_rate == 0.5

    # Test: 1 success, 1 failure -> success_rate = 0.5, collision_rate = 0.0
    assert report.test_metrics.success_rate == 0.5
    assert report.test_metrics.collision_rate == 0.0

    gen_eval.close()


# ---------------------------------------------------------------------------
# Regression Tests: Conflicting Info Resolution & Canonical Consumer Alignment
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "info,expected_success,expected_collision",
    [
        # Direct collision=True overriding success=True
        ({"collision": True, "success": True}, False, True),
        # Alias is_collision=True overriding episode_success=True
        ({"is_collision": True, "episode_success": True}, False, True),
        # Alias had_collision=True overriding is_success=True
        ({"had_collision": True, "is_success": True}, False, True),
        # Multiple success and collision keys present
        (
            {
                "collision": True,
                "is_collision": True,
                "had_collision": True,
                "episode_success": True,
                "success": True,
                "is_success": True,
            },
            False,
            True,
        ),
        # Collision=False preserves success=True
        ({"collision": False, "success": True}, True, False),
        # Collision=False preserves success=False
        ({"collision": False, "success": False}, False, False),
        # Missing collision preserves success=True
        ({"success": True}, True, None),
        # Missing collision preserves success=False
        ({"success": False}, False, None),
        # Collision=True with missing success keeps success=None (no heuristic inference)
        ({"collision": True}, None, True),
    ],
)
def test_conflicting_info_collision_overrides_success(
    info: Dict[str, Any],
    expected_success: Optional[bool],
    expected_collision: Optional[bool],
) -> None:
    """Verify that collision always overrides positive success flags canonically in extract_episode_metrics."""
    m = extract_episode_metrics(
        reward=0.0,
        length=10,
        terminated=True,
        truncated=False,
        info=info,
    )
    assert m.success is expected_success
    assert m.collision is expected_collision


def test_evaluator_aggregation_cannot_disagree_with_episode_metrics() -> None:
    """Verify that Evaluator aggregation strictly mirrors EpisodeMetrics with zero divergence.

    Even when raw info contains conflicting flags or varied outcomes across multiple episodes,
    the evaluator's aggregated rates must mathematically equal the canonical EpisodeMetrics flags.
    """
    step_trace = [
        # Episode 1: Conflicting info (collision=True, success=True) -> canonical success=False, collision=True
        (1.0, True, False, {"collision": True, "success": True}),
        # Episode 2: Clean success
        (10.0, True, False, {"success": True, "collision": False}),
        # Episode 3: Clean failure without collision
        (-2.0, True, False, {"success": False, "collision": False}),
        # Episode 4: Truncated timeout without collision or success info
        (0.0, False, True, {}),
        # Episode 5: Conflicting info via aliases (had_collision=True, episode_success=True)
        (-5.0, True, False, {"had_collision": True, "episode_success": True}),
    ]
    env = MockStepEnv(step_trace)
    algo = MockPolicyAlgo()
    evaluator = Evaluator(algorithm=algo, env=env)

    num_episodes = 5
    metrics = evaluator.evaluate(num_episodes=num_episodes, deterministic=True)

    # Validate recorded EpisodeMetrics list
    recorded_metrics = evaluator.last_episode_metrics
    assert len(recorded_metrics) == num_episodes
    assert all(isinstance(m, EpisodeMetrics) for m in recorded_metrics)

    # Episode 1: Conflicting info resolved canonically
    assert recorded_metrics[0].success is False
    assert recorded_metrics[0].collision is True

    # Episode 2: Success
    assert recorded_metrics[1].success is True
    assert recorded_metrics[1].collision is False

    # Episode 3: Failure
    assert recorded_metrics[2].success is False
    assert recorded_metrics[2].collision is False

    # Episode 4: Truncated with missing outcomes
    assert recorded_metrics[3].success is None
    assert recorded_metrics[3].collision is None
    assert recorded_metrics[3].truncated is True

    # Episode 5: Conflicting aliases resolved canonically
    assert recorded_metrics[4].success is False
    assert recorded_metrics[4].collision is True

    # Evaluator aggregation MUST match EpisodeMetrics counts exactly
    expected_successes = sum(1 for m in recorded_metrics if m.success is True)
    expected_collisions = sum(1 for m in recorded_metrics if m.collision is True)
    expected_truncations = sum(1 for m in recorded_metrics if m.truncated)

    assert expected_successes == 1
    assert expected_collisions == 2
    assert expected_truncations == 1

    valid_success_count = sum(1 for m in recorded_metrics if m.success is not None)
    valid_collision_count = sum(1 for m in recorded_metrics if m.collision is not None)
    assert valid_success_count == 4
    assert valid_collision_count == 4

    assert metrics.success_rate == pytest.approx(expected_successes / valid_success_count)
    assert metrics.collision_rate == pytest.approx(expected_collisions / valid_collision_count)
    assert metrics.truncation_rate == pytest.approx(expected_truncations / num_episodes)
    env.close()


def test_all_consumers_consistent_on_conflicting_info() -> None:
    """Verify that all consumers interpret conflicting info identically via EpisodeMetrics."""
    from adaptive_rl.curriculum.callbacks import CurriculumCallback
    from adaptive_rl.curriculum.curriculum import Curriculum
    from adaptive_rl.curriculum.stage import CurriculumStage

    conflicting_info = {"collision": True, "success": True}

    # 1. Evaluator
    env_eval = MockStepEnv([(0.0, True, False, dict(conflicting_info))])
    evaluator = Evaluator(algorithm=MockPolicyAlgo(), env=env_eval)
    eval_metrics = evaluator.evaluate(num_episodes=1, deterministic=True)
    assert evaluator.last_episode_metrics[0].success is False
    assert evaluator.last_episode_metrics[0].collision is True
    assert eval_metrics.success_rate == 0.0
    assert eval_metrics.collision_rate == 1.0
    env_eval.close()

    # 2. GeneralizationEvaluator
    env_gen = MockStepEnv([(0.0, True, False, dict(conflicting_info))])
    gen_eval = GeneralizationEvaluator(algorithm=MockPolicyAlgo(), env=env_gen)
    gen_dist = GeneralizationDistribution(train_seeds=[1], test_seeds=[2])
    # Need 2 step outcomes for train and test
    env_gen.step_outcomes = [
        (0.0, True, False, dict(conflicting_info)),
        (0.0, True, False, dict(conflicting_info)),
    ]
    gen_report = gen_eval.evaluate_generalization(distribution=gen_dist, experiment_name="test")
    assert gen_report.train_metrics.success_rate == 0.0
    assert gen_report.train_metrics.collision_rate == 1.0
    assert gen_report.test_metrics.success_rate == 0.0
    assert gen_report.test_metrics.collision_rate == 1.0
    gen_eval.close()

    # 3. MetricLoggerCallback
    logger = MetricLoggerCallback()
    ep_metrics = extract_episode_metrics(
        reward=0.0,
        length=1,
        terminated=True,
        truncated=False,
        info=conflicting_info,
    )
    logger.on_episode_end(
        episode=1,
        episode_reward=0.0,
        episode_length=1,
        info=conflicting_info,
        metrics=ep_metrics,
    )
    assert logger.successes == 0
    assert logger.collisions == 1
    assert logger.success_rate == 0.0
    assert logger.collision_rate == 1.0

    # 4. CurriculumCallback
    stage = CurriculumStage(stage_id=0, name="stage0", environment_parameters={})
    curriculum = Curriculum(name="test_curr", stages=[stage], eval_window=5)
    curr_cb = CurriculumCallback(curriculum=curriculum, verbose=0)
    curr_cb.on_episode_end(
        episode=1,
        episode_reward=0.0,
        episode_length=1,
        info=conflicting_info,
        metrics=ep_metrics,
    )
    assert curr_cb.recent_successes[-1] == 0.0


def test_intermediate_collision_not_repeated_on_terminal_step() -> None:
    """Verify that collision on intermediate step is retained even if terminal step reports collision=False.

    Step 1: collision=False
    Step 2: collision=True
    Step 3: collision=False
    Step 4: terminated=True, collision=False, success=True
    Canonical outcome: collision=True, success=False (collision overrides success).
    """
    step_infos = [
        {"collision": False},
        {"collision": True},
        {"collision": False},
        {"collision": False, "success": True},
    ]

    # 1. EpisodeMetricsAccumulator
    acc = EpisodeMetricsAccumulator()
    acc.record_step(reward=1.0, terminated=False, truncated=False, info=step_infos[0])
    acc.record_step(reward=-5.0, terminated=False, truncated=False, info=step_infos[1])
    acc.record_step(reward=0.0, terminated=False, truncated=False, info=step_infos[2])
    acc.record_step(reward=10.0, terminated=True, truncated=False, info=step_infos[3])
    m_acc = acc.finish()

    assert m_acc.collision is True
    assert m_acc.success is False
    assert m_acc.reward == 6.0
    assert m_acc.length == 4
    assert m_acc.terminated is True
    assert m_acc.truncated is False

    # 2. extract_episode_metrics with step_infos
    m_extract = extract_episode_metrics(
        reward=6.0,
        length=4,
        terminated=True,
        truncated=False,
        info=step_infos[-1],
        step_infos=step_infos[:-1],
    )
    assert m_extract.collision is True
    assert m_extract.success is False

    # 3. Evaluator
    step_trace = [
        (1.0, False, False, step_infos[0]),
        (-5.0, False, False, step_infos[1]),
        (0.0, False, False, step_infos[2]),
        (10.0, True, False, step_infos[3]),
    ]
    env = MockStepEnv(step_trace)
    evaluator = Evaluator(algorithm=MockPolicyAlgo(), env=env)
    eval_metrics = evaluator.evaluate(num_episodes=1, deterministic=True)

    assert evaluator.last_episode_metrics[0].collision is True
    assert evaluator.last_episode_metrics[0].success is False
    assert eval_metrics.collision_rate == 1.0
    assert eval_metrics.success_rate == 0.0
    env.close()

    # 4. GeneralizationEvaluator
    env_gen = MockStepEnv(
        [
            (1.0, False, False, step_infos[0]),
            (-5.0, False, False, step_infos[1]),
            (0.0, False, False, step_infos[2]),
            (10.0, True, False, step_infos[3]),
            (1.0, False, False, step_infos[0]),
            (-5.0, False, False, step_infos[1]),
            (0.0, False, False, step_infos[2]),
            (10.0, True, False, step_infos[3]),
        ]
    )
    gen_eval = GeneralizationEvaluator(algorithm=MockPolicyAlgo(), env=env_gen)
    gen_dist = GeneralizationDistribution(train_seeds=[42], test_seeds=[43])
    gen_report = gen_eval.evaluate_generalization(distribution=gen_dist, experiment_name="test_col")
    assert gen_report.train_metrics.collision_rate == 1.0
    assert gen_report.train_metrics.success_rate == 0.0
    assert gen_report.test_metrics.collision_rate == 1.0
    assert gen_report.test_metrics.success_rate == 0.0
    env_gen.close()


def test_intermediate_success_occurring_before_terminal_step() -> None:
    """Verify that success reached on an intermediate step is recorded as success if collision-free.

    Step 1: success=False, collision=False
    Step 2: success=True, collision=False (reached target)
    Step 3: success=False, collision=False
    Step 4: terminated=True, success=False, collision=False
    Canonical outcome: success=True, collision=False.
    """
    step_infos = [
        {"collision": False},
        {"success": True, "collision": False},
        {"collision": False},
        {"collision": False},
    ]

    # 1. EpisodeMetricsAccumulator
    acc = EpisodeMetricsAccumulator()
    acc.record_step(reward=0.0, terminated=False, truncated=False, info=step_infos[0])
    acc.record_step(reward=10.0, terminated=False, truncated=False, info=step_infos[1])
    acc.record_step(reward=0.0, terminated=False, truncated=False, info=step_infos[2])
    acc.record_step(reward=0.0, terminated=True, truncated=False, info=step_infos[3])
    m = acc.finish()

    assert m.success is True
    assert m.collision is False
    assert m.reward == 10.0
    assert m.length == 4

    # 2. extract_episode_metrics with step_infos
    m_ext = extract_episode_metrics(
        reward=10.0,
        length=4,
        terminated=True,
        truncated=False,
        info=step_infos[-1],
        step_infos=step_infos[:-1],
    )
    assert m_ext.success is True
    assert m_ext.collision is False

    # 3. Evaluator
    step_trace = [
        (0.0, False, False, step_infos[0]),
        (10.0, False, False, step_infos[1]),
        (0.0, False, False, step_infos[2]),
        (0.0, True, False, step_infos[3]),
    ]
    env = MockStepEnv(step_trace)
    evaluator = Evaluator(algorithm=MockPolicyAlgo(), env=env)
    eval_metrics = evaluator.evaluate(num_episodes=1, deterministic=True)
    assert evaluator.last_episode_metrics[0].success is True
    assert evaluator.last_episode_metrics[0].collision is False
    assert eval_metrics.success_rate == 1.0
    assert eval_metrics.collision_rate == 0.0
    env.close()

    # Contrast: intermediate success followed by intermediate collision must fail
    step_trace_col = [
        (0.0, False, False, {"collision": False}),
        (10.0, False, False, {"success": True, "collision": False}),
        (-5.0, False, False, {"collision": True}),
        (0.0, True, False, {"collision": False}),
    ]
    env_col = MockStepEnv(step_trace_col)
    evaluator_col = Evaluator(algorithm=MockPolicyAlgo(), env=env_col)
    eval_col_metrics = evaluator_col.evaluate(num_episodes=1, deterministic=True)
    assert evaluator_col.last_episode_metrics[0].success is False
    assert evaluator_col.last_episode_metrics[0].collision is True
    assert eval_col_metrics.success_rate == 0.0
    assert eval_col_metrics.collision_rate == 1.0
    env_col.close()


def test_traffic_overflow_followed_by_success_flag() -> None:
    """Verify traffic success contract: queue overflow or premature termination strictly forbids success.

    Case 1: Traffic overflow on step 2, but terminal step reports success=True -> success=False.
    Case 2: Early termination (terminated=True) in traffic -> success=False.
    Case 3: Clean completion to horizon (truncated=True, terminated=False, no overflow, controlled queue) -> success=True.
    """
    # Case 1: Overflow followed by success=True
    traffic_steps_overflow = [
        (1.0, False, False, {"total_queue": 5, "step_overflow": False}),
        (-10.0, False, False, {"total_queue": 30, "step_overflow": True}),
        (0.0, False, False, {"total_queue": 15, "step_overflow": False}),
        (5.0, False, True, {"total_queue": 2, "step_overflow": False, "success": True}),
    ]
    acc1 = EpisodeMetricsAccumulator(is_traffic=True)
    for r, term, trunc, inf in traffic_steps_overflow:
        acc1.record_step(reward=r, terminated=term, truncated=trunc, info=inf)
    m1 = acc1.finish()
    assert m1.success is False

    env1 = MockStepEnv(traffic_steps_overflow)
    evaluator1 = Evaluator(algorithm=MockPolicyAlgo(), env=env1)
    eval_metrics1 = evaluator1.evaluate(num_episodes=1, deterministic=True)
    assert evaluator1.last_episode_metrics[0].success is False
    assert eval_metrics1.success_rate == 0.0
    assert eval_metrics1.overflow_rate == 1.0
    env1.close()

    # Case 2: Early termination (terminated=True) in traffic
    traffic_steps_early_term = [
        (1.0, False, False, {"total_queue": 5, "step_overflow": False}),
        (-10.0, True, False, {"total_queue": 30, "step_overflow": False, "success": True}),
    ]
    acc2 = EpisodeMetricsAccumulator(is_traffic=True)
    for r, term, trunc, inf in traffic_steps_early_term:
        acc2.record_step(reward=r, terminated=term, truncated=trunc, info=inf)
    m2 = acc2.finish()
    assert m2.success is False

    # Case 3: Clean completion to horizon without overflow -> success=True
    traffic_steps_clean = [
        (1.0, False, False, {"total_queue": 5, "step_overflow": False}),
        (2.0, False, False, {"total_queue": 4, "step_overflow": False}),
        (5.0, False, True, {"total_queue": 2, "step_overflow": False, "success": True}),
    ]
    acc3 = EpisodeMetricsAccumulator(is_traffic=True)
    for r, term, trunc, inf in traffic_steps_clean:
        acc3.record_step(reward=r, terminated=term, truncated=trunc, info=inf)
    m3 = acc3.finish()
    assert m3.success is True

    env3 = MockStepEnv(traffic_steps_clean)
    evaluator3 = Evaluator(algorithm=MockPolicyAlgo(), env=env3)
    eval_metrics3 = evaluator3.evaluate(num_episodes=1, deterministic=True)
    assert evaluator3.last_episode_metrics[0].success is True
    assert eval_metrics3.success_rate == 1.0
    assert eval_metrics3.overflow_rate == 0.0
    env3.close()


def test_mixed_outcomes_across_episodes_and_documented_denominator_semantics() -> None:
    """Verify aggregated rate semantics across mixed True, False, and None episode outcomes.

    Denominator semantics:
    - Rate calculations use total evaluation episodes (num_episodes) as canonical denominator.
    - If an outcome metric is unavailable across all episodes (all None), rate is None.
    - Episodes where metric is False or None do not contribute to the numerator.
    """
    step_trace: list[tuple[float, bool, bool, dict[str, Any]]] = [
        # Episode 1: Success=True, Collision=False
        (10.0, True, False, {"success": True, "collision": False}),
        # Episode 2: Success=False, Collision=True
        (-5.0, True, False, {"success": False, "collision": True}),
        # Episode 3: Outcome keys unavailable (e.g. non-spatial continuous control)
        (3.0, True, False, {"sensor_data": 42}),
    ]
    env = MockStepEnv(step_trace)
    evaluator = Evaluator(algorithm=MockPolicyAlgo(), env=env)
    metrics = evaluator.evaluate(num_episodes=3, deterministic=True)

    recorded = evaluator.last_episode_metrics
    assert len(recorded) == 3

    assert recorded[0].success is True
    assert recorded[0].collision is False

    assert recorded[1].success is False
    assert recorded[1].collision is True

    assert recorded[2].success is None
    assert recorded[2].collision is None

    # Total episodes is 3, but defined episodes is 2:
    # 1 success / 2 defined episodes = 0.5
    # 1 collision / 2 defined episodes = 0.5
    assert metrics.episodes == 3
    assert metrics.success_rate == 0.5
    assert metrics.collision_rate == 0.5
    env.close()

    # If all episodes have metric unavailable, rate must be None (not 0.0)
    all_none_trace: list[tuple[float, bool, bool, dict[str, Any]]] = [
        (1.0, True, False, {}),
        (2.0, True, False, {}),
    ]
    env_none = MockStepEnv(all_none_trace)
    evaluator_none = Evaluator(algorithm=MockPolicyAlgo(), env=env_none)
    metrics_none = evaluator_none.evaluate(num_episodes=2, deterministic=True)
    assert metrics_none.success_rate is None
    assert metrics_none.collision_rate is None
    env_none.close()


def test_callback_fallback_extraction_preserves_truncation_and_termination() -> None:
    """Verify callback fallback extraction properly inspects info flags instead of hardcoding terminated=True."""
    # 1. MetricLoggerCallback
    logger = MetricLoggerCallback()

    # Episode with TimeLimit.truncated: True
    logger.on_episode_end(
        episode=1,
        episode_reward=5.0,
        episode_length=100,
        info={"TimeLimit.truncated": True},
        metrics=None,
    )
    m1 = logger.episode_metrics[0]
    assert m1.truncated is True
    assert m1.terminated is False

    # Episode with gymnasium truncated: True
    logger.on_episode_end(
        episode=2,
        episode_reward=3.0,
        episode_length=50,
        info={"truncated": True},
        metrics=None,
    )
    m2 = logger.episode_metrics[1]
    assert m2.truncated is True
    assert m2.terminated is False

    # Episode with natural termination (terminated: True)
    logger.on_episode_end(
        episode=3,
        episode_reward=10.0,
        episode_length=20,
        info={"terminated": True},
        metrics=None,
    )
    m3 = logger.episode_metrics[2]
    assert m3.terminated is True
    assert m3.truncated is False

    # Episode with empty info defaults to natural termination
    logger.on_episode_end(
        episode=4,
        episode_reward=0.0,
        episode_length=10,
        info={},
        metrics=None,
    )
    m4 = logger.episode_metrics[3]
    assert m4.terminated is True
    assert m4.truncated is False

    # 2. CurriculumCallback
    from adaptive_rl.curriculum.callbacks import CurriculumCallback
    from adaptive_rl.curriculum.curriculum import Curriculum
    from adaptive_rl.curriculum.stage import CurriculumStage

    stage = CurriculumStage(stage_id=0, name="stage0", environment_parameters={})
    curriculum = Curriculum(name="test_curr", stages=[stage], eval_window=5)
    curr_cb = CurriculumCallback(curriculum=curriculum, verbose=0)

    # Call on_episode_end with truncated info
    curr_cb.on_episode_end(
        episode=1,
        episode_reward=5.0,
        episode_length=100,
        info={"TimeLimit.truncated": True, "success": False},
        metrics=None,
    )
    assert len(curr_cb.recent_rewards) == 1
    assert curr_cb.recent_rewards[0] == 5.0


def test_api_cleanup_single_public_owner() -> None:
    """Verify EpisodeMetrics, EpisodeMetricsAccumulator, and extract_episode_metrics have one clear public owner.

    The canonical owner is adaptive_rl.metrics (and top-level adaptive_rl).
    adaptive_rl.evaluation and adaptive_rl.evaluation.metrics must not re-export them.
    """
    import adaptive_rl
    import adaptive_rl.evaluation
    import adaptive_rl.evaluation.metrics as eval_metrics_mod
    import adaptive_rl.metrics as canonical_metrics_mod

    # Authoritative exports from adaptive_rl.metrics
    assert hasattr(canonical_metrics_mod, "EpisodeMetrics")
    assert hasattr(canonical_metrics_mod, "EpisodeMetricsAccumulator")
    assert hasattr(canonical_metrics_mod, "extract_episode_metrics")
    assert "EpisodeMetrics" in canonical_metrics_mod.__all__
    assert "EpisodeMetricsAccumulator" in canonical_metrics_mod.__all__
    assert "extract_episode_metrics" in canonical_metrics_mod.__all__

    # Convenience exports from top-level adaptive_rl
    assert hasattr(adaptive_rl, "EpisodeMetrics")
    assert hasattr(adaptive_rl, "EpisodeMetricsAccumulator")
    assert hasattr(adaptive_rl, "extract_episode_metrics")
    assert "EpisodeMetrics" in adaptive_rl.__all__
    assert "EpisodeMetricsAccumulator" in adaptive_rl.__all__
    assert "extract_episode_metrics" in adaptive_rl.__all__

    # adaptive_rl.evaluation does NOT re-export them
    assert "EpisodeMetrics" not in adaptive_rl.evaluation.__all__
    assert "EpisodeMetricsAccumulator" not in adaptive_rl.evaluation.__all__
    assert "extract_episode_metrics" not in adaptive_rl.evaluation.__all__

    # adaptive_rl.evaluation.metrics does NOT re-export them
    assert "EpisodeMetrics" not in eval_metrics_mod.__all__
    assert "EpisodeMetricsAccumulator" not in eval_metrics_mod.__all__
    assert "extract_episode_metrics" not in eval_metrics_mod.__all__


def test_compute_rate_canonical_semantics() -> None:
    """Verify compute_rate enforces exact canonical denominator semantics."""
    from adaptive_rl.metrics import compute_rate

    # [True, False, None] -> 1/2 = 0.5
    assert compute_rate([True, False, None]) == 0.5

    # [None, None] -> None
    assert compute_rate([None, None]) is None

    # [True, True] -> 1.0
    assert compute_rate([True, True]) == 1.0

    # [False, False] -> 0.0
    assert compute_rate([False, False]) == 0.0

    # Empty list -> None
    assert compute_rate([]) is None

    # Mixed single defined
    assert compute_rate([None, True, None]) == 1.0
    assert compute_rate([None, False, None]) == 0.0


def test_traffic_outcome_policy_composition() -> None:
    """Verify domain separation: TrafficOutcomePolicy is composed into EpisodeMetricsAccumulator."""
    from adaptive_rl.metrics import (
        DefaultOutcomePolicy,
        EpisodeMetricsAccumulator,
        TrafficOutcomePolicy,
    )

    # 1. Default policy does not care about overflow
    acc_default = EpisodeMetricsAccumulator(outcome_policy=DefaultOutcomePolicy())
    acc_default.record_step(info={"overflow": True, "success": True}, terminated=True)
    m_default = acc_default.finish()
    assert m_default.success is True

    # 2. Traffic policy forces success=False when overflow occurred
    acc_traffic = EpisodeMetricsAccumulator(outcome_policy=TrafficOutcomePolicy())
    acc_traffic.record_step(info={"overflow": True})
    acc_traffic.record_step(info={"success": True}, truncated=True)
    m_traffic = acc_traffic.finish()
    assert m_traffic.success is False
    assert m_traffic.additional_metrics.get("had_overflow") is True

    # 3. Dynamic composition via property setter
    acc_dynamic = EpisodeMetricsAccumulator()
    assert isinstance(acc_dynamic.outcome_policy, DefaultOutcomePolicy)
    acc_dynamic.is_traffic = True
    assert isinstance(acc_dynamic.outcome_policy, TrafficOutcomePolicy)


def test_full_lifecycle_end_to_end_undefined_metrics(tmp_path: Path) -> None:
    """Verify full pipeline (env -> callback adapter -> metrics -> logger -> result -> serialized output) preserves None."""
    from adaptive_rl.experiments.metadata import (
        EpisodeRecord,
        ExperimentMetadata,
        save_episodes_csv,
    )
    from adaptive_rl.training.callbacks import MetricLoggerCallback, SB3CallbackAdapter
    from adaptive_rl.training.trainer import TrainingResult

    logger = MetricLoggerCallback()
    adapter = SB3CallbackAdapter(callbacks=[logger])

    # Simulate environment where success and collision are undefined
    adapter.locals = {
        "dones": [False],
        "infos": [{"other_data": 42}],
        "rewards": [1.0],
    }
    adapter.num_timesteps = 1
    adapter._on_step()

    # Terminal transition
    adapter.locals = {
        "dones": [True],
        "infos": [{"other_data": 43}],
        "rewards": [2.0],
    }
    adapter.num_timesteps = 2
    adapter._on_step()

    # Verify EpisodeMetrics
    assert len(logger.episode_metrics) == 1
    m = logger.episode_metrics[0]
    assert m.success is None
    assert m.collision is None
    assert m.reward == 3.0
    assert m.length == 2

    # Verify MetricLoggerCallback rates are None (never collapsed to 0.0)
    assert logger.success_rate is None
    assert logger.collision_rate is None

    # Construct EpisodeRecord
    rec = EpisodeRecord(
        episode=1,
        reward=m.reward,
        length=m.length,
        success=m.success,
        collision=m.collision,
        timestep=2,
    )
    assert rec.success is None
    assert rec.collision is None

    # Construct TrainingResult
    res = TrainingResult(
        experiment_name="test_undef",
        total_timesteps=2,
        episodes_completed=1,
        mean_reward=3.0,
        final_model_path=tmp_path / "model.zip",
        success_rate=logger.success_rate,
        collision_rate=logger.collision_rate,
    )
    assert res.success_rate is None
    assert res.collision_rate is None

    # Construct ExperimentMetadata and save JSON
    meta = ExperimentMetadata(
        experiment_name="test_undef",
        algorithm="ppo",
        environment="dummy",
        seed=42,
        total_timesteps=2,
        actual_timesteps=2,
        episodes_completed=1,
        mean_reward=3.0,
        success_rate=logger.success_rate,
        collision_rate=logger.collision_rate,
        final_model_path=str(tmp_path / "model.zip"),
    )
    json_path = meta.save(tmp_path)
    with open(json_path) as f:
        meta_dict = json.load(f)
    assert meta_dict["success_rate"] is None
    assert meta_dict["collision_rate"] is None

    # Save CSV and verify None serializes as empty string ("")
    csv_path = save_episodes_csv([rec], tmp_path, "test_undef")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert len(rows) == 1
    assert rows[0]["success"] == ""
    assert rows[0]["collision"] == ""


def test_full_lifecycle_end_to_end_explicit_false(tmp_path: Path) -> None:
    """Verify explicit False remains False throughout the entire lifecycle."""
    from adaptive_rl.experiments.metadata import (
        EpisodeRecord,
        ExperimentMetadata,
        save_episodes_csv,
    )
    from adaptive_rl.training.callbacks import MetricLoggerCallback, SB3CallbackAdapter
    from adaptive_rl.training.trainer import TrainingResult

    logger = MetricLoggerCallback()
    adapter = SB3CallbackAdapter(callbacks=[logger])

    adapter.locals = {
        "dones": [True],
        "infos": [{"success": False, "collision": False}],
        "rewards": [0.0],
    }
    adapter.num_timesteps = 1
    adapter._on_step()

    m = logger.episode_metrics[0]
    assert m.success is False
    assert m.collision is False

    assert logger.success_rate == 0.0
    assert logger.collision_rate == 0.0

    rec = EpisodeRecord(
        episode=1,
        reward=m.reward,
        length=m.length,
        success=m.success,
        collision=m.collision,
        timestep=1,
    )
    assert rec.success is False
    assert rec.collision is False

    res = TrainingResult(
        experiment_name="test_false",
        total_timesteps=1,
        episodes_completed=1,
        mean_reward=0.0,
        final_model_path=tmp_path / "model.zip",
        success_rate=logger.success_rate,
        collision_rate=logger.collision_rate,
    )
    assert res.success_rate == 0.0
    assert res.collision_rate == 0.0

    meta = ExperimentMetadata(
        experiment_name="test_false",
        algorithm="ppo",
        environment="dummy",
        seed=42,
        total_timesteps=1,
        actual_timesteps=1,
        episodes_completed=1,
        mean_reward=0.0,
        success_rate=logger.success_rate,
        collision_rate=logger.collision_rate,
        final_model_path=str(tmp_path / "model.zip"),
    )
    json_path = meta.save(tmp_path)
    with open(json_path) as f:
        meta_dict = json.load(f)
    assert meta_dict["success_rate"] == 0.0
    assert meta_dict["collision_rate"] == 0.0

    csv_path = save_episodes_csv([rec], tmp_path, "test_false")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))
    assert rows[0]["success"] == "False"
    assert rows[0]["collision"] == "False"


def test_full_lifecycle_multi_step_precedence_and_truncation() -> None:
    """Verify multi-step precedence: collision overrides success, and truncation is preserved."""
    from adaptive_rl.training.callbacks import MetricLoggerCallback, SB3CallbackAdapter

    logger = MetricLoggerCallback()
    adapter = SB3CallbackAdapter(callbacks=[logger])

    # Step 1: No result
    adapter.locals = {"dones": [False], "infos": [{}], "rewards": [1.0]}
    adapter.num_timesteps = 1
    adapter._on_step()

    # Step 2: Collision occurred
    adapter.locals = {"dones": [False], "infos": [{"collision": True}], "rewards": [-5.0]}
    adapter.num_timesteps = 2
    adapter._on_step()

    # Step 3: Success flag reported
    adapter.locals = {"dones": [False], "infos": [{"success": True}], "rewards": [10.0]}
    adapter.num_timesteps = 3
    adapter._on_step()

    # Step 4: Truncated timeout
    adapter.locals = {
        "dones": [True],
        "infos": [{"TimeLimit.truncated": True}],
        "rewards": [0.0],
    }
    adapter.num_timesteps = 4
    adapter._on_step()

    assert len(logger.episode_metrics) == 1
    m = logger.episode_metrics[0]
    # Universal invariant: collision overrides success
    assert m.collision is True
    assert m.success is False
    # Truncation preserved
    assert m.truncated is True
    assert m.terminated is False
    assert m.length == 4
    assert m.reward == 6.0


def test_full_lifecycle_mixed_batch_denominators(tmp_path: Path) -> None:
    """Verify mixed batch [True, False, None] aggregates using defined-episodes denominator and serializes cleanly."""
    from adaptive_rl.experiments.metadata import EpisodeRecord, save_episodes_csv
    from adaptive_rl.training.callbacks import MetricLoggerCallback, SB3CallbackAdapter

    logger = MetricLoggerCallback()
    adapter = SB3CallbackAdapter(callbacks=[logger])

    # Ep 1: Success=True, Collision=False
    adapter.locals = {
        "dones": [True],
        "infos": [{"success": True, "collision": False}],
        "rewards": [10.0],
    }
    adapter.num_timesteps = 1
    adapter._on_step()

    # Ep 2: Success=False, Collision=False
    adapter.locals = {
        "dones": [True],
        "infos": [{"success": False, "collision": False}],
        "rewards": [0.0],
    }
    adapter.num_timesteps = 2
    adapter._on_step()

    # Ep 3: Undefined outcomes
    adapter.locals = {
        "dones": [True],
        "infos": [{}],
        "rewards": [5.0],
    }
    adapter.num_timesteps = 3
    adapter._on_step()

    assert logger.total_episodes == 3
    # Defined episodes: Ep 1 (True), Ep 2 (False) -> 1 / 2 = 0.5
    assert logger.success_rate == 0.5
    # Defined episodes for collision: Ep 1 (False), Ep 2 (False) -> 0 / 2 = 0.0
    assert logger.collision_rate == 0.0

    records = [
        EpisodeRecord(
            episode=i + 1,
            reward=m.reward,
            length=m.length,
            success=m.success,
            collision=m.collision,
            timestep=i + 1,
        )
        for i, m in enumerate(logger.episode_metrics)
    ]

    csv_path = save_episodes_csv(records, tmp_path, "mixed_batch")
    with open(csv_path) as f:
        rows = list(csv.DictReader(f))

    assert len(rows) == 3
    assert rows[0]["success"] == "True"
    assert rows[0]["collision"] == "False"
    assert rows[1]["success"] == "False"
    assert rows[1]["collision"] == "False"
    assert rows[2]["success"] == ""
    assert rows[2]["collision"] == ""


def test_traffic_terminal_success_semantics() -> None:
    """Explicitly verify the deterministic traffic terminal success rules required by Blocker #4:
    - intermediate success=True + terminal missing -> None (intermediate does NOT latch)
    - intermediate success=False + terminal missing -> None
    - terminal success=True -> True
    - terminal success=False -> False
    - terminal success=None -> None
    - overflow + terminal success -> False (overflow overrides positive terminal success)
    - truncation + terminal success -> True (controlled horizon completion)
    """
    # 1. intermediate success=True + terminal missing -> None
    acc1 = EpisodeMetricsAccumulator(is_traffic=True)
    acc1.record_step(reward=1.0, info={"success": True, "step_controlled": True})
    acc1.record_step(reward=1.0, info={"step_controlled": True}, truncated=True)
    m1 = acc1.finish()
    assert m1.success is None, (
        "Intermediate success=True must NOT latch when terminal success is missing"
    )

    # 2. intermediate success=False + terminal missing -> None
    acc2 = EpisodeMetricsAccumulator(is_traffic=True)
    acc2.record_step(reward=-1.0, info={"success": False})
    acc2.record_step(reward=0.0, info={}, truncated=True)
    m2 = acc2.finish()
    assert m2.success is None, (
        "Intermediate success=False must NOT latch when terminal success is missing"
    )

    # 3. terminal success=True -> True
    acc3 = EpisodeMetricsAccumulator(is_traffic=True)
    acc3.record_step(reward=1.0, info={})
    acc3.record_step(reward=1.0, info={"success": True}, truncated=True)
    m3 = acc3.finish()
    assert m3.success is True

    # 4. terminal success=False -> False
    acc4 = EpisodeMetricsAccumulator(is_traffic=True)
    acc4.record_step(reward=1.0, info={})
    acc4.record_step(reward=1.0, info={"success": False}, truncated=True)
    m4 = acc4.finish()
    assert m4.success is False

    # 5. terminal success=None -> None
    acc5 = EpisodeMetricsAccumulator(is_traffic=True)
    acc5.record_step(reward=1.0, info={})
    acc5.record_step(reward=1.0, info={}, truncated=True)
    m5 = acc5.finish()
    assert m5.success is None

    # 6. overflow + terminal success -> False
    acc6 = EpisodeMetricsAccumulator(is_traffic=True)
    acc6.record_step(reward=-10.0, info={"had_overflow": True})
    acc6.record_step(reward=1.0, info={"success": True}, truncated=True)
    m6 = acc6.finish()
    assert m6.success is False, "Overflow on any step must override terminal success to False"
    assert m6.additional_metrics.get("had_overflow") is True

    # 7. truncation + terminal success -> True
    acc7 = EpisodeMetricsAccumulator(is_traffic=True)
    acc7.record_step(reward=1.0, info={"had_overflow": False})
    acc7.record_step(reward=1.0, info={"had_overflow": False, "success": True}, truncated=True)
    m7 = acc7.finish()
    assert m7.success is True
    assert m7.truncated is True
    assert m7.terminated is False


def test_episodes_csv_roundtrip_preserves_nullability(tmp_path: Path) -> None:
    """Verify that serialization to CSV and deserialization preserves the exact tri-state semantics:
    None -> serialize -> deserialize -> None
    False -> serialize -> deserialize -> False
    True -> serialize -> deserialize -> True
    """
    from adaptive_rl.experiments.metadata import (
        EpisodeRecord,
        load_episodes_csv,
        save_episodes_csv,
    )

    records = [
        EpisodeRecord(episode=1, reward=10.0, length=5, success=None, collision=None, timestep=5),
        EpisodeRecord(
            episode=2, reward=0.0, length=10, success=False, collision=False, timestep=15
        ),
        EpisodeRecord(episode=3, reward=25.0, length=8, success=True, collision=False, timestep=23),
        EpisodeRecord(
            episode=4, reward=-50.0, length=4, success=False, collision=True, timestep=27
        ),
    ]

    csv_path = save_episodes_csv(records, tmp_path, "roundtrip_test")
    loaded = load_episodes_csv(csv_path)

    assert len(loaded) == 4

    # Record 1: None / None
    assert loaded[0].episode == 1
    assert loaded[0].reward == 10.0
    assert loaded[0].length == 5
    assert loaded[0].success is None
    assert loaded[0].collision is None
    assert loaded[0].timestep == 5

    # Record 2: False / False
    assert loaded[1].episode == 2
    assert loaded[1].reward == 0.0
    assert loaded[1].length == 10
    assert loaded[1].success is False
    assert loaded[1].collision is False
    assert loaded[1].timestep == 15

    # Record 3: True / False
    assert loaded[2].episode == 3
    assert loaded[2].reward == 25.0
    assert loaded[2].length == 8
    assert loaded[2].success is True
    assert loaded[2].collision is False
    assert loaded[2].timestep == 23

    # Record 4: False / True
    assert loaded[3].episode == 4
    assert loaded[3].reward == -50.0
    assert loaded[3].length == 4
    assert loaded[3].success is False
    assert loaded[3].collision is True
    assert loaded[3].timestep == 27


def test_accumulator_configuration_matrix() -> None:
    """Verify outcome policy selection, compatibility mode, matching flags, and rejection of conflicting configs."""
    import pytest

    from adaptive_rl.metrics import (
        DefaultOutcomePolicy,
        EpisodeMetricsAccumulator,
        TrafficOutcomePolicy,
        extract_episode_metrics,
    )

    # 1. no policy + no is_traffic -> DefaultOutcomePolicy
    acc_none = EpisodeMetricsAccumulator()
    assert isinstance(acc_none.outcome_policy, DefaultOutcomePolicy)
    assert acc_none.is_traffic is False

    # 2. traffic compatibility mode (is_traffic=True)
    acc_compat_traffic = EpisodeMetricsAccumulator(is_traffic=True)
    assert isinstance(acc_compat_traffic.outcome_policy, TrafficOutcomePolicy)
    assert acc_compat_traffic.is_traffic is True

    # 3. non-traffic compatibility mode (is_traffic=False)
    acc_compat_default = EpisodeMetricsAccumulator(is_traffic=False)
    assert isinstance(acc_compat_default.outcome_policy, DefaultOutcomePolicy)
    assert acc_compat_default.is_traffic is False

    # 4. explicit default policy
    acc_explicit_default = EpisodeMetricsAccumulator(outcome_policy=DefaultOutcomePolicy())
    assert isinstance(acc_explicit_default.outcome_policy, DefaultOutcomePolicy)
    assert acc_explicit_default.is_traffic is False

    # 5. explicit traffic policy
    acc_explicit_traffic = EpisodeMetricsAccumulator(outcome_policy=TrafficOutcomePolicy())
    assert isinstance(acc_explicit_traffic.outcome_policy, TrafficOutcomePolicy)
    assert acc_explicit_traffic.is_traffic is True

    # 6. explicit policy + matching is_traffic
    acc_match_def = EpisodeMetricsAccumulator(
        outcome_policy=DefaultOutcomePolicy(), is_traffic=False
    )
    assert isinstance(acc_match_def.outcome_policy, DefaultOutcomePolicy)

    acc_match_traf = EpisodeMetricsAccumulator(
        outcome_policy=TrafficOutcomePolicy(), is_traffic=True
    )
    assert isinstance(acc_match_traf.outcome_policy, TrafficOutcomePolicy)

    # 7. explicit policy + conflicting is_traffic -> clear ValueError
    with pytest.raises(ValueError, match="Conflicting configuration"):
        EpisodeMetricsAccumulator(outcome_policy=DefaultOutcomePolicy(), is_traffic=True)

    with pytest.raises(ValueError, match="Conflicting configuration"):
        EpisodeMetricsAccumulator(outcome_policy=TrafficOutcomePolicy(), is_traffic=False)

    # 8. extract_episode_metrics with conflicting configuration -> clear ValueError
    with pytest.raises(ValueError, match="Conflicting configuration"):
        extract_episode_metrics(
            reward=1.0,
            length=5,
            terminated=True,
            truncated=False,
            outcome_policy=DefaultOutcomePolicy(),
            is_traffic=True,
        )

    with pytest.raises(ValueError, match="Conflicting configuration"):
        extract_episode_metrics(
            reward=1.0,
            length=5,
            terminated=True,
            truncated=False,
            outcome_policy=TrafficOutcomePolicy(),
            is_traffic=False,
        )

    # 9. Dynamic property setter rejects conflicting toggle when policy was explicitly specified
    acc_locked_def = EpisodeMetricsAccumulator(outcome_policy=DefaultOutcomePolicy())
    with pytest.raises(ValueError, match="Cannot set is_traffic=True"):
        acc_locked_def.is_traffic = True

    acc_locked_traf = EpisodeMetricsAccumulator(outcome_policy=TrafficOutcomePolicy())
    with pytest.raises(ValueError, match="Cannot set is_traffic=False"):
        acc_locked_traf.is_traffic = False


def test_isolated_fallback_domain_detection() -> None:
    """Verify isolated fallback domain detection behavior and that explicit policy always wins."""
    from adaptive_rl.metrics import (
        DefaultOutcomePolicy,
        EpisodeMetricsAccumulator,
        TrafficOutcomePolicy,
    )

    # 1. Unspecified configuration falls back to TrafficOutcomePolicy when traffic telemetry detected
    acc_fallback = EpisodeMetricsAccumulator()
    assert isinstance(acc_fallback.outcome_policy, DefaultOutcomePolicy)
    acc_fallback.record_step(info={"queue_lengths": [3, 2], "step_overflow": False})
    assert isinstance(acc_fallback.outcome_policy, TrafficOutcomePolicy)

    # 2. Explicit DefaultOutcomePolicy is NEVER overridden by traffic telemetry (explicit policy wins)
    acc_explicit = EpisodeMetricsAccumulator(outcome_policy=DefaultOutcomePolicy())
    assert isinstance(acc_explicit.outcome_policy, DefaultOutcomePolicy)
    acc_explicit.record_step(
        info={"queue_lengths": [3, 2], "overflow": True, "success": True},
        terminated=True,
    )
    # Still DefaultOutcomePolicy!
    assert isinstance(acc_explicit.outcome_policy, DefaultOutcomePolicy)
    m = acc_explicit.finish()
    # In DefaultOutcomePolicy, overflow does not invalidate success
    assert m.success is True


def test_traffic_outcome_policy_explicit_semantics() -> None:
    """Verify TrafficOutcomePolicy semantics using explicit policy instance."""
    from adaptive_rl.metrics import EpisodeMetricsAccumulator, TrafficOutcomePolicy

    # 1. Intermediate success=True with terminal success missing -> None (no latching)
    acc1 = EpisodeMetricsAccumulator(outcome_policy=TrafficOutcomePolicy())
    acc1.record_step(reward=1.0, info={"success": True})
    acc1.record_step(reward=1.0, info={}, truncated=True)
    m1 = acc1.finish()
    assert m1.success is None

    # 2. Intermediate success=True with terminal success=False -> False
    acc2 = EpisodeMetricsAccumulator(outcome_policy=TrafficOutcomePolicy())
    acc2.record_step(reward=1.0, info={"success": True})
    acc2.record_step(reward=0.0, info={"success": False}, truncated=True)
    m2 = acc2.finish()
    assert m2.success is False

    # 3. Terminal success=True -> True
    acc3 = EpisodeMetricsAccumulator(outcome_policy=TrafficOutcomePolicy())
    acc3.record_step(reward=1.0, info={"success": True}, truncated=True)
    m3 = acc3.finish()
    assert m3.success is True

    # 4. Terminal success=False -> False
    acc4 = EpisodeMetricsAccumulator(outcome_policy=TrafficOutcomePolicy())
    acc4.record_step(reward=1.0, info={"success": False}, truncated=True)
    m4 = acc4.finish()
    assert m4.success is False

    # 5. Terminal success missing -> None
    acc5 = EpisodeMetricsAccumulator(outcome_policy=TrafficOutcomePolicy())
    acc5.record_step(reward=1.0, info={}, truncated=True)
    m5 = acc5.finish()
    assert m5.success is None

    # 6. Overflow anywhere forces success=False
    acc6 = EpisodeMetricsAccumulator(outcome_policy=TrafficOutcomePolicy())
    acc6.record_step(reward=-5.0, info={"overflow": True})
    acc6.record_step(reward=1.0, info={"success": True}, truncated=True)
    m6 = acc6.finish()
    assert m6.success is False
    assert m6.additional_metrics.get("had_overflow") is True

    # 7. Premature termination forces success=False
    acc7 = EpisodeMetricsAccumulator(outcome_policy=TrafficOutcomePolicy())
    acc7.record_step(reward=0.0, info={"success": True}, terminated=True, truncated=False)
    m7 = acc7.finish()
    assert m7.success is False
    assert m7.terminated is True

    # 8. Controlled horizon completion (truncation) with terminal success -> True
    acc8 = EpisodeMetricsAccumulator(outcome_policy=TrafficOutcomePolicy())
    acc8.record_step(reward=1.0, info={"success": True}, terminated=False, truncated=True)
    m8 = acc8.finish()
    assert m8.success is True
    assert m8.truncated is True
    assert m8.terminated is False

    # 9. Collision anywhere overrides positive terminal success to False
    acc9 = EpisodeMetricsAccumulator(outcome_policy=TrafficOutcomePolicy())
    acc9.record_step(reward=-10.0, info={"collision": True})
    acc9.record_step(reward=1.0, info={"success": True}, truncated=True)
    m9 = acc9.finish()
    assert m9.collision is True
    assert m9.success is False


def test_evaluator_and_generalization_explicit_policy() -> None:
    """Verify Evaluator and GeneralizationEvaluator with explicit OutcomePolicy."""
    from adaptive_rl.evaluation.evaluator import Evaluator
    from adaptive_rl.evaluation.generalization import (
        GeneralizationDistribution,
        GeneralizationEvaluator,
    )
    from adaptive_rl.metrics import TrafficOutcomePolicy

    env_eval = MockStepEnv(
        [
            (1.0, False, True, {"success": True, "collision": False}),
            (0.0, True, False, {"success": False, "collision": True}),
        ]
    )

    evaluator = Evaluator(
        algorithm=MockPolicyAlgo(),
        env=env_eval,
        outcome_policy=TrafficOutcomePolicy(),
    )
    assert isinstance(evaluator.outcome_policy, TrafficOutcomePolicy)
    metrics = evaluator.evaluate(num_episodes=2)
    assert metrics.episodes == 2
    assert len(evaluator.last_episode_metrics) == 2

    # In traffic, ep 2 with terminated=True has success=False and collision=True
    assert evaluator.last_episode_metrics[0].success is True
    assert evaluator.last_episode_metrics[1].success is False
    assert evaluator.last_episode_metrics[1].collision is True

    # GeneralizationEvaluator
    env_gen = MockStepEnv(
        [
            (1.0, False, True, {"success": True, "collision": False}),
            (0.0, True, False, {"success": False, "collision": True}),
            (1.0, False, True, {"success": True, "collision": False}),
            (0.0, True, False, {"success": False, "collision": True}),
        ]
    )
    gen_eval = GeneralizationEvaluator(
        algorithm=MockPolicyAlgo(),
        env=env_gen,
        outcome_policy=TrafficOutcomePolicy(),
    )
    assert isinstance(gen_eval.outcome_policy, TrafficOutcomePolicy)
    dist = GeneralizationDistribution(train_seeds=[1, 2], test_seeds=[3, 4])
    report = gen_eval.evaluate_generalization(dist)
    assert report.train_metrics.episodes == 2
    assert report.test_metrics.episodes == 2


def test_shared_traffic_outcome_policy_cross_episode_isolation() -> None:
    """Verify that a single shared TrafficOutcomePolicy instance does NOT leak state across episodes or environments.

    Regression test for: TrafficOutcomePolicy reused across multiple accumulators/episodes
    must not allow overflow information from one episode/environment to contaminate another.
    """
    from adaptive_rl.metrics import EpisodeMetricsAccumulator, TrafficOutcomePolicy

    shared_policy = TrafficOutcomePolicy()

    # Episode 1: Suffers catastrophic queue overflow
    acc1 = EpisodeMetricsAccumulator(outcome_policy=shared_policy)
    acc1.record_step(reward=-10.0, info={"overflow": True})
    acc1.record_step(reward=1.0, info={"success": True}, truncated=True)
    m1 = acc1.finish()
    assert m1.success is False
    assert m1.additional_metrics.get("had_overflow") is True

    # Episode 2: Clean episode reusing the exact same shared_policy instance!
    acc2 = EpisodeMetricsAccumulator(outcome_policy=shared_policy)
    acc2.record_step(reward=1.0, info={"queue_lengths": [1, 0], "overflow": False})
    acc2.record_step(reward=1.0, info={"success": True}, truncated=True)
    m2 = acc2.finish()
    assert m2.success is True, (
        "Episode 2 must NOT be marked failed due to overflow in previous Episode 1 sharing the same policy"
    )
    assert m2.additional_metrics.get("had_overflow") is False

    # Episode 3: Premature termination on shared policy
    acc3 = EpisodeMetricsAccumulator(outcome_policy=shared_policy)
    acc3.record_step(reward=0.0, info={"success": True}, terminated=True, truncated=False)
    m3 = acc3.finish()
    assert m3.success is False
    assert m3.terminated is True

    # Episode 4: Clean truncated episode on shared policy
    acc4 = EpisodeMetricsAccumulator(outcome_policy=shared_policy)
    acc4.record_step(reward=1.0, info={"queue_lengths": [0, 0], "overflow": False})
    acc4.record_step(reward=1.0, info={"success": True}, terminated=False, truncated=True)
    m4 = acc4.finish()
    assert m4.success is True, (
        "Episode 4 must succeed; previous premature termination on shared policy must not leak"
    )
    assert m4.terminated is False
    assert m4.truncated is True

    # Vectorized / concurrent environments sharing the same policy instance
    acc_env_a = EpisodeMetricsAccumulator(outcome_policy=shared_policy)
    acc_env_b = EpisodeMetricsAccumulator(outcome_policy=shared_policy)

    # Step 1: Env A overflows, Env B is clean
    acc_env_a.record_step(reward=-5.0, info={"had_overflow": True})
    acc_env_b.record_step(reward=1.0, info={"had_overflow": False})

    # Step 2: Both finish with terminal success=True
    acc_env_a.record_step(reward=1.0, info={"success": True}, truncated=True)
    acc_env_b.record_step(reward=1.0, info={"success": True}, truncated=True)

    m_env_a = acc_env_a.finish()
    m_env_b = acc_env_b.finish()

    assert m_env_a.success is False, "Env A had overflow and must fail"
    assert m_env_a.additional_metrics.get("had_overflow") is True

    assert m_env_b.success is True, (
        "Env B had no overflow and must succeed; must not be contaminated by concurrent Env A sharing same policy"
    )
    assert m_env_b.additional_metrics.get("had_overflow") is False


def test_callback_adapter_requires_metrics_parameter() -> None:
    """SB3CallbackAdapter always passes metrics; legacy signatures fail loudly."""
    import pytest

    from adaptive_rl.training.callbacks import BaseCallback, SB3CallbackAdapter

    class LegacyCallback(BaseCallback):
        def on_episode_end(
            self,
            episode: int,
            episode_reward: float,
            episode_length: int,
            info: Optional[Dict[str, Any]] = None,
        ) -> None:
            return None

    adapter = SB3CallbackAdapter(callbacks=[LegacyCallback()])
    adapter.locals = {
        "dones": [True],
        "rewards": [1.0],
        "infos": [{"success": True}],
    }
    adapter.num_timesteps = 1
    with pytest.raises(TypeError, match="metrics"):
        adapter._on_step()


def test_outcome_policy_selection_is_immutable_after_first_step() -> None:
    """Policy cannot be swapped after the first record_step of an episode."""
    import pytest

    from adaptive_rl.metrics import (
        DefaultOutcomePolicy,
        EpisodeMetricsAccumulator,
        TrafficOutcomePolicy,
    )

    acc = EpisodeMetricsAccumulator()
    acc.record_step(reward=1.0, info={"success": True})
    assert isinstance(acc.outcome_policy, DefaultOutcomePolicy)

    with pytest.raises(RuntimeError, match="immutable"):
        acc.outcome_policy = TrafficOutcomePolicy()

    with pytest.raises(RuntimeError, match="immutable"):
        acc.is_traffic = True

    # Late traffic telemetry must not switch a locked default policy
    acc.record_step(
        reward=1.0,
        info={"queue_lengths": [3, 1], "overflow": True, "success": True},
        truncated=True,
    )
    assert isinstance(acc.outcome_policy, DefaultOutcomePolicy)
    m = acc.finish()
    assert m.success is True


def test_extract_episode_metrics_info_and_step_infos_not_double_counted() -> None:
    """Terminal info is recorded once when it is already the last step_infos entry."""
    from adaptive_rl.metrics import extract_episode_metrics

    terminal = {"success": True, "collision": False}
    prefix = [{"collision": True}, {"collision": False}]
    full = prefix + [terminal]

    # Historical call style: prefix + terminal info
    m_split = extract_episode_metrics(
        reward=3.0,
        length=3,
        terminated=True,
        truncated=False,
        info=terminal,
        step_infos=prefix,
    )
    assert m_split.collision is True
    assert m_split.success is False
    assert m_split.length == 3
    assert m_split.additional_metrics.get("collision") is None

    # Complete trace plus terminal info must not wipe last_info with a duplicate
    m_full = extract_episode_metrics(
        reward=3.0,
        length=3,
        terminated=True,
        truncated=False,
        info=terminal,
        step_infos=full,
    )
    assert m_full.collision is True
    assert m_full.success is False
    assert m_full.length == 3

    # Complete trace only (info omitted)
    m_trace = extract_episode_metrics(
        reward=3.0,
        length=3,
        terminated=True,
        truncated=False,
        step_infos=full,
    )
    assert m_trace.collision is True
    assert m_trace.success is False

    # Full-trace extract sees later traffic keys and selects TrafficOutcomePolicy up front
    traffic_steps = [
        {"success": True},
        {"queue_lengths": [2, 0], "overflow": True, "success": True},
    ]
    m_traffic = extract_episode_metrics(
        reward=1.0,
        length=2,
        terminated=False,
        truncated=True,
        info=traffic_steps[-1],
        step_infos=traffic_steps,
    )
    assert m_traffic.success is False
    assert m_traffic.additional_metrics.get("had_overflow") is True


def test_builtin_outcome_policies_reject_episode_state() -> None:
    """Built-in policies have empty slots so they cannot accumulate episode facts."""
    import pytest

    from adaptive_rl.metrics import DefaultOutcomePolicy, TrafficOutcomePolicy

    for policy in (DefaultOutcomePolicy(), TrafficOutcomePolicy()):
        with pytest.raises(AttributeError):
            policy.had_overflow = True  # type: ignore[attr-defined]


def test_extract_episode_metrics_arbitrary_telemetry_payloads() -> None:
    """Arbitrary telemetry payloads (e.g. NumPy arrays) do not raise equality errors."""
    from adaptive_rl.metrics import extract_episode_metrics

    # 1. Terminal mapping containing a multi-element NumPy array
    terminal_with_array = {
        "success": True,
        "queue_lengths": np.array([2, 1, 0]),
    }
    m1 = extract_episode_metrics(
        reward=5.0,
        length=1,
        terminated=False,
        truncated=True,
        info=terminal_with_array,
    )
    assert m1.success is True
    assert np.array_equal(m1.additional_metrics["queue_lengths"], np.array([2, 1, 0]))

    # 2. Nested non-scalar telemetry mapping containing NumPy arrays
    terminal_nested = {
        "success": True,
        "nested": {
            "values": np.array([1, 2, 3]),
        },
    }
    m2 = extract_episode_metrics(
        reward=2.5,
        length=1,
        terminated=True,
        truncated=False,
        info=terminal_nested,
    )
    assert m2.success is True
    assert "nested" in m2.additional_metrics

    # 3. Multi-step episode where step_infos and info contain distinct NumPy array objects
    step1 = {"step": 1, "state": np.array([0.1, 0.2])}
    step2 = {"step": 2, "success": True, "state": np.array([0.3, 0.4])}
    m3 = extract_episode_metrics(
        reward=10.0,
        length=2,
        terminated=True,
        truncated=False,
        info=step2,
        step_infos=[step1, step2],
    )
    assert m3.success is True
    assert m3.length == 2

    # 4. Caller provides separate, non-identical dict with numpy array as info
    step_prefix = [{"step": 1, "obs": np.array([1.0, 2.0])}]
    step_terminal = {"step": 2, "success": True, "obs": np.array([3.0, 4.0])}
    m4 = extract_episode_metrics(
        reward=1.0,
        length=2,
        terminated=True,
        truncated=False,
        info=step_terminal,
        step_infos=step_prefix,
    )
    assert m4.success is True
    assert m4.length == 2


def test_extract_episode_metrics_terminal_identity_deduplication() -> None:
    """Terminal info identity deduplication preserves canonical semantics without mutation."""
    from adaptive_rl.metrics import _resolve_episode_step_infos, extract_episode_metrics

    terminal = {"success": True, "collision": False}
    prefix = [{"collision": True}, {"collision": False}]

    # Case A: step_infos=[terminal], info=terminal using SAME object -> recorded once
    step_infos_single = [terminal]
    resolved_single = _resolve_episode_step_infos(info=terminal, step_infos=step_infos_single)
    assert len(resolved_single) == 1
    assert resolved_single[0] is terminal

    m_single = extract_episode_metrics(
        reward=1.0,
        length=1,
        terminated=True,
        truncated=False,
        info=terminal,
        step_infos=step_infos_single,
    )
    assert m_single.success is True
    assert m_single.collision is False

    # Case B: step_infos=[prefix], info=terminal -> terminal must be appended once
    step_infos_prefix = list(prefix)
    resolved_prefix = _resolve_episode_step_infos(info=terminal, step_infos=step_infos_prefix)
    assert len(resolved_prefix) == 3
    assert resolved_prefix[-1] is terminal

    m_prefix = extract_episode_metrics(
        reward=3.0,
        length=3,
        terminated=True,
        truncated=False,
        info=terminal,
        step_infos=step_infos_prefix,
    )
    # Collision in prefix overrides success in terminal
    assert m_prefix.collision is True
    assert m_prefix.success is False
    assert m_prefix.length == 3

    # Case C: caller sequence is not mutated
    original_trace = [prefix[0], prefix[1]]
    _resolve_episode_step_infos(info=terminal, step_infos=original_trace)
    assert len(original_trace) == 2


def test_resolve_step_infos_numpy_equality_would_crash() -> None:
    """Prove that dict-equality deduplication crashes on NumPy telemetry.

    This test explicitly demonstrates the bug that existed when
    _resolve_episode_step_infos used ``dict(resolved[-1]) == dict(info)``
    instead of identity-based deduplication.  The ValueError is the
    exact failure mode reported in PR #116.
    """
    from adaptive_rl.metrics import _resolve_episode_step_infos

    # Two structurally identical but distinct dict objects containing
    # multi-element NumPy arrays.  dict-equality comparison raises
    # ValueError because NumPy element-wise == returns an array whose
    # truth value is ambiguous.
    step = {"success": True, "queue_lengths": np.array([2, 1, 0])}
    terminal = {"success": True, "queue_lengths": np.array([2, 1, 0])}

    # Prove the old comparison would raise.
    with pytest.raises(ValueError, match="ambiguous"):
        dict(step) == dict(terminal)  # noqa: B015

    # The fixed implementation must NOT raise.
    resolved = _resolve_episode_step_infos(info=terminal, step_infos=[step])
    # step and terminal are distinct objects, so terminal is appended.
    assert len(resolved) == 2
    assert resolved[0] is step
    assert resolved[1] is terminal

    # Same object path: no duplication, no crash.
    resolved_same = _resolve_episode_step_infos(info=step, step_infos=[step])
    assert len(resolved_same) == 1
    assert resolved_same[0] is step


def test_extract_episode_metrics_numpy_terminal_separate_objects() -> None:
    """End-to-end extraction with NumPy telemetry and separate info/step_infos objects.

    This exercises the full canonical extraction path with the exact
    scenario that triggered the ValueError before the identity fix.
    """
    # Use non-traffic keys to avoid automatic TrafficOutcomePolicy detection,
    # which would override success semantics.
    step_info = {
        "success": False,
        "sensor_readings": np.array([5, 3, 1]),
    }
    terminal_info = {
        "success": True,
        "sensor_readings": np.array([2, 1, 0]),
    }

    m = extract_episode_metrics(
        reward=10.0,
        length=2,
        terminated=True,
        truncated=False,
        info=terminal_info,
        step_infos=[step_info],
    )
    assert m.success is True
    assert m.length == 2
    assert m.reward == 10.0
    assert np.array_equal(m.additional_metrics["sensor_readings"], np.array([2, 1, 0]))


def test_extract_episode_metrics_complex_arbitrary_telemetry() -> None:
    """Terminal-info resolution handles complex heterogeneous telemetry.

    Covers: lists, tuples, nested dicts, NumPy arrays, NumPy scalar values.
    None of these types should cause the resolution or extraction to crash.
    """
    complex_terminal = {
        "success": True,
        "list_data": [1, 2, 3],
        "tuple_data": (4.0, 5.0),
        "nested_dict": {
            "inner_array": np.array([10, 20, 30]),
            "inner_scalar": np.float64(3.14),
        },
        "array_2d": np.array([[1, 2], [3, 4]]),
        "np_int": np.int32(42),
        "np_bool": np.bool_(True),
    }

    # info-only extraction
    m1 = extract_episode_metrics(
        reward=7.0,
        length=1,
        terminated=True,
        truncated=False,
        info=complex_terminal,
    )
    assert m1.success is True
    assert m1.additional_metrics["list_data"] == [1, 2, 3]
    assert m1.additional_metrics["tuple_data"] == (4.0, 5.0)
    assert np.array_equal(
        m1.additional_metrics["nested_dict"]["inner_array"],
        np.array([10, 20, 30]),
    )
    assert m1.additional_metrics["nested_dict"]["inner_scalar"] == pytest.approx(3.14)
    assert np.array_equal(
        m1.additional_metrics["array_2d"],
        np.array([[1, 2], [3, 4]]),
    )
    assert m1.additional_metrics["np_int"] == 42
    assert m1.additional_metrics["np_bool"] is True or m1.additional_metrics["np_bool"] == True  # noqa: E712

    # step_infos + info with complex telemetry (distinct objects)
    prefix = [{"step": 0, "readings": np.array([0.1, 0.2, 0.3])}]
    m2 = extract_episode_metrics(
        reward=5.0,
        length=2,
        terminated=True,
        truncated=False,
        info=complex_terminal,
        step_infos=prefix,
    )
    assert m2.success is True
    assert m2.length == 2

    # step_infos + info with same complex object
    full_trace = [prefix[0], complex_terminal]
    m3 = extract_episode_metrics(
        reward=5.0,
        length=2,
        terminated=True,
        truncated=False,
        info=complex_terminal,
        step_infos=full_trace,
    )
    assert m3.success is True
    assert m3.length == 2


def test_resolve_step_infos_info_only() -> None:
    """When only info is provided, it becomes the single episode step."""
    from adaptive_rl.metrics import _resolve_episode_step_infos

    terminal = {"success": True, "value": 42}
    resolved = _resolve_episode_step_infos(info=terminal, step_infos=None)
    assert len(resolved) == 1
    assert resolved[0] is terminal


def test_resolve_step_infos_step_infos_only() -> None:
    """When only step_infos is provided, it is returned as the complete trace."""
    from adaptive_rl.metrics import _resolve_episode_step_infos

    steps = [{"a": 1}, {"b": 2}, {"success": True}]
    resolved = _resolve_episode_step_infos(info=None, step_infos=steps)
    assert len(resolved) == 3
    # Ordering preserved.
    assert resolved[0]["a"] == 1
    assert resolved[2]["success"] is True
