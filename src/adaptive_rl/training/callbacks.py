"""Training lifecycle callbacks for AdaptiveRL."""

from __future__ import annotations

from abc import ABC
from typing import TYPE_CHECKING, Any, Dict, List, Optional

import numpy as np
from stable_baselines3.common.callbacks import BaseCallback as SB3BaseCallback

from adaptive_rl.metrics import (
    EpisodeMetrics,
    EpisodeMetricsAccumulator,
    OutcomePolicy,
    compute_rate,
    extract_episode_metrics,
)

if TYPE_CHECKING:
    from adaptive_rl.algorithms.base import BaseAlgorithm
    from adaptive_rl.training.checkpointing import CheckpointManager


class BaseCallback(ABC):
    """Abstract base class for monitoring, logging, and checkpointing during training."""

    def on_training_start(self, locals_dict: Optional[Dict[str, Any]] = None) -> None:
        """Called before the first training step."""
        pass

    def on_step(
        self,
        step: int,
        locals_dict: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Called after every environment step.

        Args:
            step: Current cumulative environment timestep.
            locals_dict: Dictionary of local variables from the training loop.

        Returns:
            True to continue training, False to abort early.
        """
        return True

    def on_episode_end(
        self,
        episode: int,
        episode_reward: float,
        episode_length: int,
        info: Optional[Dict[str, Any]] = None,
        metrics: Optional[EpisodeMetrics] = None,
    ) -> None:
        """Called upon completion of an environment episode.

        Args:
            episode: Cumulative episode number.
            episode_reward: Total reward accumulated during the episode.
            episode_length: Number of steps in the episode.
            info: Terminal environment information.
            metrics: Canonical episode metrics produced by the metrics pipeline.
        """
        pass

    def on_training_end(self) -> None:
        """Called after the final training step has executed."""
        pass


class MetricLoggerCallback(BaseCallback):
    """Tracks and logs episodic metrics (rewards, lengths, collisions, success)."""

    def __init__(self, window_size: int = 100) -> None:
        """Initialize metric logger.

        Args:
            window_size: Number of recent episodes used to calculate rolling statistics.
        """
        self.window_size = window_size
        self.episode_rewards: List[float] = []
        self.episode_lengths: List[int] = []
        self.total_episodes: int = 0
        self.successes: int = 0
        self.collisions: int = 0
        self.episode_metrics: List[EpisodeMetrics] = []

    def on_episode_end(
        self,
        episode: int,
        episode_reward: float,
        episode_length: int,
        info: Optional[Dict[str, Any]] = None,
        metrics: Optional[EpisodeMetrics] = None,
    ) -> None:
        """Record episode outcomes."""
        self.total_episodes += 1
        self.episode_rewards.append(episode_reward)
        self.episode_lengths.append(episode_length)

        # Compatibility path for callers that have not yet migrated to the
        # canonical EpisodeMetrics contract.
        if metrics is None:
            info_dict = dict(info or {})

            is_truncated = bool(
                info_dict.get("TimeLimit.truncated", False) or info_dict.get("truncated", False)
            )
            is_terminated = bool(info_dict.get("terminated", not is_truncated))

            metrics = extract_episode_metrics(
                reward=episode_reward,
                length=episode_length,
                terminated=is_terminated,
                truncated=is_truncated,
                info=info_dict,
            )

        self.episode_metrics.append(metrics)

        # Preserve nullable semantics:
        # None = undefined, False = explicit negative, True = explicit positive.
        if metrics.success is True:
            self.successes += 1

        if metrics.collision is True:
            self.collisions += 1

    @property
    def mean_reward(self) -> float:
        """Rolling mean episodic reward."""
        if not self.episode_rewards:
            return 0.0

        window = self.episode_rewards[-self.window_size :]
        return float(np.mean(window))

    @property
    def mean_length(self) -> float:
        """Rolling mean episode length."""
        if not self.episode_lengths:
            return 0.0

        window = self.episode_lengths[-self.window_size :]
        return float(np.mean(window))

    @property
    def success_rate(self) -> Optional[float]:
        """Success rate among episodes where success is defined."""
        if not self.episode_metrics:
            return None

        return compute_rate([metrics.success for metrics in self.episode_metrics])

    @property
    def collision_rate(self) -> Optional[float]:
        """Collision rate among episodes where collision is defined."""
        if not self.episode_metrics:
            return None

        return compute_rate([metrics.collision for metrics in self.episode_metrics])


class CheckpointCallback(BaseCallback):
    """Periodically saves model checkpoints using CheckpointManager."""

    def __init__(
        self,
        checkpoint_manager: CheckpointManager,
        save_freq: int,
        model: Optional[BaseAlgorithm] = None,
        verbose: int = 0,
    ) -> None:
        """Initialize checkpoint callback.

        Args:
            checkpoint_manager: CheckpointManager instance.
            save_freq: Number of environment steps between saves.
            model: Optional BaseAlgorithm model reference to save.
            verbose: Verbosity level.
        """
        self.manager = checkpoint_manager
        self.save_freq = save_freq
        self.model = model
        self.verbose = verbose
        self.last_save_step: int = 0

    def set_model(self, model: BaseAlgorithm) -> None:
        """Assign model reference to save."""
        self.model = model

    def on_step(
        self,
        step: int,
        locals_dict: Optional[Dict[str, Any]] = None,
    ) -> bool:
        """Check save frequency and persist checkpoint."""
        if self.save_freq > 0 and (step - self.last_save_step) >= self.save_freq:
            self.last_save_step = step

            if self.model is not None:
                self.manager.save_checkpoint(
                    model=self.model,
                    step=step,
                    metric_value=0.0,
                )

        return True


class SB3CallbackAdapter(SB3BaseCallback):
    """Adapter bridging AdaptiveRL callbacks to Stable-Baselines3 callbacks.

    Metrics Data Flow:
        PPO / SAC training step
            ↓
        SB3CallbackAdapter._on_step()
            ↓
        EpisodeMetricsAccumulator.record_step()
            ↓
        EpisodeMetricsAccumulator.finish() -> EpisodeMetrics
            ↓
        callback.on_episode_end(..., metrics=ep_metrics)
            ↓
        MetricLoggerCallback / CurriculumCallback
            ↓
        EpisodeRecord / TrainingResult / serialization
    """

    def __init__(
        self,
        callbacks: List[BaseCallback],
        algorithm: Optional[BaseAlgorithm] = None,
        verbose: int = 0,
        outcome_policy: Optional[OutcomePolicy] = None,
    ) -> None:
        """Initialize adapter with list of AdaptiveRL callbacks."""
        super().__init__(verbose)

        self.callbacks = callbacks
        self.algorithm = algorithm
        self.outcome_policy = outcome_policy
        self._current_rewards: Dict[int, float] = {}
        self._current_lengths: Dict[int, int] = {}
        self._accumulators: Dict[int, EpisodeMetricsAccumulator] = {}
        self._episode_count: int = 0

    def _on_training_start(self) -> None:
        """Propagate training start event."""
        for callback in self.callbacks:
            if isinstance(callback, CheckpointCallback) and self.algorithm is not None:
                callback.set_model(self.algorithm)

            callback.on_training_start(
                locals_dict=self.locals,
            )

    def _on_step(self) -> bool:
        """Track steps and episode outcomes from SB3 training step."""
        step = int(self.num_timesteps)

        # Check for episode endings across vectorized / single environments.
        dones = self.locals.get("dones", [False])
        infos = self.locals.get("infos", [{}])
        rewards = self.locals.get("rewards", [0.0])

        for i, done in enumerate(dones):
            reward = float(rewards[i]) if i < len(rewards) else 0.0
            info = infos[i] if i < len(infos) else {}

            self._current_rewards[i] = self._current_rewards.get(i, 0.0) + reward
            self._current_lengths[i] = self._current_lengths.get(i, 0) + 1

            if i not in self._accumulators:
                self._accumulators[i] = EpisodeMetricsAccumulator(
                    outcome_policy=self.outcome_policy,
                )

            is_truncated = bool(
                info.get("TimeLimit.truncated", False) or info.get("truncated", False)
            )

            is_terminated = bool(info.get("terminated", done and not is_truncated))

            accumulator = self._accumulators[i]

            accumulator.record_step(
                reward=reward,
                terminated=is_terminated,
                truncated=is_truncated,
                info=info,
            )

            if done:
                self._episode_count += 1

                self._current_rewards.pop(i, None)
                self._current_lengths.pop(i, None)

                accumulator = self._accumulators.pop(i)
                ep_metrics = accumulator.finish()

                ep_reward = ep_metrics.reward
                ep_length = ep_metrics.length

                for callback in self.callbacks:
                    callback.on_episode_end(
                        episode=self._episode_count,
                        episode_reward=ep_reward,
                        episode_length=ep_length,
                        info=info,
                        metrics=ep_metrics,
                    )

        continue_training = True

        for callback in self.callbacks:
            if not callback.on_step(
                step=step,
                locals_dict=self.locals,
            ):
                continue_training = False

        return continue_training

    def _on_training_end(self) -> None:
        """Propagate training end event."""
        for callback in self.callbacks:
            callback.on_training_end()
