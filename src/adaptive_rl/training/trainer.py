"""Training engine implementations for AdaptiveRL."""

from __future__ import annotations

import random
import time
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any, Dict, List, Optional

import gymnasium as gym
import numpy as np
import torch

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.algorithms.sac import SACAlgorithm
from adaptive_rl.config import ExperimentConfig
from adaptive_rl.environments.registry import make_env
from adaptive_rl.experiments.metadata import (
    EpisodeRecord,
    ExperimentMetadata,
    save_episodes_csv,
)
from adaptive_rl.metrics import (
    DefaultOutcomePolicy,
    EpisodeMetrics,
    EpisodeMetricsAccumulator,
    OutcomePolicy,
    TrafficOutcomePolicy,
)
from adaptive_rl.training.callbacks import (
    BaseCallback,
    CheckpointCallback,
    MetricLoggerCallback,
    SB3CallbackAdapter,
)
from adaptive_rl.training.checkpointing import CheckpointManager


@dataclass
class TrainingResult:
    """Structured summary and artifacts resulting from a training run."""

    experiment_name: str
    total_timesteps: int
    episodes_completed: int
    mean_reward: float
    final_model_path: Path
    checkpoints: List[Dict[str, Any]] = field(default_factory=list)
    episode_rewards: List[float] = field(default_factory=list)
    episode_lengths: List[int] = field(default_factory=list)
    success_rate: Optional[float] = None
    collision_rate: Optional[float] = None
    metadata_path: Optional[Path] = None
    episodes_csv_path: Optional[Path] = None


class BaseTrainer(ABC):
    """Abstract interface for RL training workflows in AdaptiveRL."""

    @abstractmethod
    def fit(self) -> TrainingResult:
        """Execute the training process and return structured results."""
        pass

    def close(self) -> None:
        """Clean up trainer resources and close environment."""
        if hasattr(self, "env") and self.env is not None:
            try:
                self.env.close()
            except Exception:
                pass


class PPOTrainer(BaseTrainer):
    """Concrete PPO training engine coordinating environment, algorithm, callbacks, and checkpoints."""

    def __init__(
        self,
        config: ExperimentConfig,
        env: Optional[gym.Env] = None,
        callbacks: Optional[List[BaseCallback]] = None,
    ) -> None:
        """Initialize PPOTrainer from experiment configuration.

        Args:
            config: Validated ExperimentConfig instance.
            env: Optional pre-instantiated Gymnasium environment.
            callbacks: Optional list of additional BaseCallback instances.
        """
        self.config = config
        self._set_deterministic_seed(self.config.seed)

        # 1. Environment initialization
        if env is not None:
            self.env = env
        else:
            self.env = make_env(
                self.config.environment.name,
                **self.config.environment.parameters,
            )

        if "traffic" in self.config.environment.name.lower():
            self.outcome_policy: OutcomePolicy = TrafficOutcomePolicy()
        else:
            self.outcome_policy = DefaultOutcomePolicy()

        # 2. Checkpoint management
        checkpoint_dir = self.config.output_dir / "checkpoints" / self.config.name
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        # 3. Callbacks setup
        if self.config.training is None:
            raise ValueError(
                f"Training configuration ('training') is required for {self.__class__.__name__}."
            )
        training_cfg = self.config.training

        self.metric_logger = MetricLoggerCallback()
        self._callbacks: List[BaseCallback] = [self.metric_logger]

        if training_cfg.checkpoint_freq > 0:
            checkpoint_cb = CheckpointCallback(
                checkpoint_manager=self.checkpoint_manager,
                save_freq=training_cfg.checkpoint_freq,
            )
            self._callbacks.append(checkpoint_cb)

        if callbacks:
            self._callbacks.extend(callbacks)

        # 4. Algorithm configuration
        algo_params = dict(self.config.algorithm.parameters)
        lr = (
            self.config.algorithm.learning_rate
            if self.config.algorithm.learning_rate is not None
            else 3e-4
        )
        gamma = self.config.algorithm.gamma if self.config.algorithm.gamma is not None else 0.99
        batch_size = (
            self.config.algorithm.batch_size if self.config.algorithm.batch_size is not None else 64
        )

        self.algorithm = PPOAlgorithm(
            env=self.env,
            learning_rate=lr,
            gamma=gamma,
            batch_size=batch_size,
            seed=self.config.seed,
            **algo_params,
        )

    @staticmethod
    def _set_deterministic_seed(seed: int) -> None:
        """Enforce deterministic random seeds across libraries."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fit(self) -> TrainingResult:
        """Execute end-to-end PPO training process.

        Returns:
            TrainingResult: Summary of training performance, episode outcomes, and artifact paths.
        """
        started_at = time.time()
        import adaptive_rl

        adapter = SB3CallbackAdapter(
            callbacks=self._callbacks,
            algorithm=self.algorithm,
            outcome_policy=self.outcome_policy,
        )

        assert self.config.training is not None
        # Run optimization
        self.algorithm.train(
            total_timesteps=self.config.training.total_timesteps,
            callback=adapter,
        )

        # Save final model
        models_dir = self.config.output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        final_model_path = models_dir / f"{self.config.name}_final.zip"
        self.algorithm.save(final_model_path)

        finished_at = time.time()
        duration = finished_at - started_at

        # Build per-episode records for CSV export from canonical EpisodeMetrics
        episode_records: List[EpisodeRecord] = []
        cumulative_ts = 0
        if self.metric_logger.episode_metrics:
            for i, m in enumerate(self.metric_logger.episode_metrics):
                cumulative_ts += m.length
                episode_records.append(
                    EpisodeRecord(
                        episode=i + 1,
                        reward=float(m.reward),
                        length=int(m.length),
                        success=m.success,
                        collision=m.collision,
                        timestep=cumulative_ts,
                    )
                )
        else:
            for i, (rew, length) in enumerate(
                zip(self.metric_logger.episode_rewards, self.metric_logger.episode_lengths)
            ):
                cumulative_ts += length
                episode_records.append(
                    EpisodeRecord(
                        episode=i + 1,
                        reward=float(rew),
                        length=int(length),
                        success=None,
                        collision=None,
                        timestep=cumulative_ts,
                    )
                )

        # Save metadata.json and episodes.csv in metadata subdir
        metadata_dir = self.config.output_dir / "metadata"
        metadata = ExperimentMetadata(
            experiment_name=self.config.name,
            algorithm=self.config.algorithm.name,
            environment=self.config.environment.name,
            seed=self.config.seed,
            total_timesteps=self.config.training.total_timesteps,
            actual_timesteps=self.algorithm.num_timesteps,
            episodes_completed=self.metric_logger.total_episodes,
            mean_reward=self.metric_logger.mean_reward,
            success_rate=self.metric_logger.success_rate,
            collision_rate=self.metric_logger.collision_rate,
            final_model_path=str(final_model_path),
            checkpoint_paths=[cp["path"] for cp in self.checkpoint_manager.list_checkpoints()],
            config_snapshot=self.config.model_dump(mode="python"),
            adaptive_rl_version=adaptive_rl.__version__,
            finished_at=__import__("datetime")
            .datetime.fromtimestamp(finished_at, tz=__import__("datetime").timezone.utc)
            .isoformat(),
            duration_seconds=round(duration, 3),
        )
        metadata_path = metadata.save(metadata_dir, name=self.config.name)
        episodes_csv_path = save_episodes_csv(
            records=episode_records,
            output_dir=metadata_dir,
            name=self.config.name,
        )

        result = TrainingResult(
            experiment_name=self.config.name,
            total_timesteps=self.config.training.total_timesteps,
            episodes_completed=self.metric_logger.total_episodes,
            mean_reward=self.metric_logger.mean_reward,
            final_model_path=final_model_path,
            checkpoints=self.checkpoint_manager.list_checkpoints(),
            episode_rewards=list(self.metric_logger.episode_rewards),
            episode_lengths=list(self.metric_logger.episode_lengths),
            success_rate=self.metric_logger.success_rate,
            collision_rate=self.metric_logger.collision_rate,
            metadata_path=metadata_path,
            episodes_csv_path=episodes_csv_path,
        )
        return result

    def evaluate(
        self,
        episodes: int = 10,
        deterministic: bool = True,
    ) -> tuple[float, float]:
        """Evaluate current policy on the environment using canonical EpisodeMetrics.

        Args:
            episodes: Number of evaluation episodes.
            deterministic: Whether to use deterministic action selection.

        Returns:
            tuple[float, float]: (mean_reward, std_reward)
        """
        metrics_list: List[EpisodeMetrics] = []
        for ep in range(episodes):
            obs, info = self.env.reset(seed=self.config.seed + ep if self.config.seed else None)
            acc = EpisodeMetricsAccumulator(outcome_policy=self.outcome_policy)
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
            metrics_list.append(m)

        rewards = [m.reward for m in metrics_list]
        return float(np.mean(rewards)), float(np.std(rewards))


class SACTrainer(BaseTrainer):
    """Concrete SAC training engine coordinating environment, algorithm, callbacks, and checkpoints."""

    def __init__(
        self,
        config: ExperimentConfig,
        env: Optional[gym.Env] = None,
        callbacks: Optional[List[BaseCallback]] = None,
    ) -> None:
        """Initialize SACTrainer from experiment configuration.

        Args:
            config: Validated ExperimentConfig instance.
            env: Optional pre-instantiated Gymnasium environment.
            callbacks: Optional list of additional BaseCallback instances.
        """
        self.config = config
        self._set_deterministic_seed(self.config.seed)

        # 1. Environment initialization
        if env is not None:
            self.env = env
        else:
            self.env = make_env(
                self.config.environment.name,
                **self.config.environment.parameters,
            )

        if "traffic" in self.config.environment.name.lower():
            self.outcome_policy: OutcomePolicy = TrafficOutcomePolicy()
        else:
            self.outcome_policy = DefaultOutcomePolicy()

        # 2. Checkpoint management
        checkpoint_dir = self.config.output_dir / "checkpoints" / self.config.name
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        # 3. Callbacks setup
        if self.config.training is None:
            raise ValueError(
                f"Training configuration ('training') is required for {self.__class__.__name__}."
            )
        training_cfg = self.config.training

        self.metric_logger = MetricLoggerCallback()
        self._callbacks: List[BaseCallback] = [self.metric_logger]

        if training_cfg.checkpoint_freq > 0:
            checkpoint_cb = CheckpointCallback(
                checkpoint_manager=self.checkpoint_manager,
                save_freq=training_cfg.checkpoint_freq,
            )
            self._callbacks.append(checkpoint_cb)

        if callbacks:
            self._callbacks.extend(callbacks)

        # 4. Algorithm configuration
        algo_params = dict(self.config.algorithm.parameters)
        lr = (
            self.config.algorithm.learning_rate
            if self.config.algorithm.learning_rate is not None
            else 3e-4
        )
        gamma = self.config.algorithm.gamma if self.config.algorithm.gamma is not None else 0.99
        batch_size = (
            self.config.algorithm.batch_size if self.config.algorithm.batch_size is not None else 64
        )

        self.algorithm = SACAlgorithm(
            env=self.env,
            learning_rate=lr,
            gamma=gamma,
            batch_size=batch_size,
            seed=self.config.seed,
            **algo_params,
        )

    @staticmethod
    def _set_deterministic_seed(seed: int) -> None:
        """Enforce deterministic random seeds across libraries."""
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)

    def fit(self) -> TrainingResult:
        """Execute end-to-end SAC training process.

        Returns:
            TrainingResult: Summary of training performance, episode outcomes, and artifact paths.
        """
        started_at = time.time()
        import adaptive_rl

        adapter = SB3CallbackAdapter(
            callbacks=self._callbacks,
            algorithm=self.algorithm,
            outcome_policy=self.outcome_policy,
        )

        assert self.config.training is not None
        # Run optimization
        self.algorithm.train(
            total_timesteps=self.config.training.total_timesteps,
            callback=adapter,
        )

        # Save final model
        models_dir = self.config.output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        final_model_path = models_dir / f"{self.config.name}_final.zip"
        self.algorithm.save(final_model_path)

        finished_at = time.time()
        duration = finished_at - started_at

        # Build per-episode records for CSV export from canonical EpisodeMetrics
        episode_records: List[EpisodeRecord] = []
        cumulative_ts = 0
        if self.metric_logger.episode_metrics:
            for i, m in enumerate(self.metric_logger.episode_metrics):
                cumulative_ts += m.length
                episode_records.append(
                    EpisodeRecord(
                        episode=i + 1,
                        reward=float(m.reward),
                        length=int(m.length),
                        success=m.success,
                        collision=m.collision,
                        timestep=cumulative_ts,
                    )
                )
        else:
            for i, (rew, length) in enumerate(
                zip(self.metric_logger.episode_rewards, self.metric_logger.episode_lengths)
            ):
                cumulative_ts += length
                episode_records.append(
                    EpisodeRecord(
                        episode=i + 1,
                        reward=float(rew),
                        length=int(length),
                        success=None,
                        collision=None,
                        timestep=cumulative_ts,
                    )
                )

        # Save metadata.json and episodes.csv
        metadata_dir = self.config.output_dir / "metadata"
        metadata = ExperimentMetadata(
            experiment_name=self.config.name,
            algorithm=self.config.algorithm.name,
            environment=self.config.environment.name,
            seed=self.config.seed,
            total_timesteps=self.config.training.total_timesteps,
            actual_timesteps=self.algorithm.num_timesteps,
            episodes_completed=self.metric_logger.total_episodes,
            mean_reward=self.metric_logger.mean_reward,
            success_rate=self.metric_logger.success_rate,
            collision_rate=self.metric_logger.collision_rate,
            final_model_path=str(final_model_path),
            checkpoint_paths=[cp["path"] for cp in self.checkpoint_manager.list_checkpoints()],
            config_snapshot=self.config.model_dump(mode="python"),
            adaptive_rl_version=adaptive_rl.__version__,
            finished_at=__import__("datetime")
            .datetime.fromtimestamp(finished_at, tz=__import__("datetime").timezone.utc)
            .isoformat(),
            duration_seconds=round(duration, 3),
        )
        metadata_path = metadata.save(metadata_dir, name=self.config.name)
        episodes_csv_path = save_episodes_csv(
            records=episode_records,
            output_dir=metadata_dir,
            name=self.config.name,
        )

        result = TrainingResult(
            experiment_name=self.config.name,
            total_timesteps=self.config.training.total_timesteps,
            episodes_completed=self.metric_logger.total_episodes,
            mean_reward=self.metric_logger.mean_reward,
            final_model_path=final_model_path,
            checkpoints=self.checkpoint_manager.list_checkpoints(),
            episode_rewards=list(self.metric_logger.episode_rewards),
            episode_lengths=list(self.metric_logger.episode_lengths),
            success_rate=self.metric_logger.success_rate,
            collision_rate=self.metric_logger.collision_rate,
            metadata_path=metadata_path,
            episodes_csv_path=episodes_csv_path,
        )
        return result

    def evaluate(
        self,
        episodes: int = 10,
        deterministic: bool = True,
    ) -> tuple[float, float]:
        """Evaluate current policy on the environment using canonical EpisodeMetrics.

        Args:
            episodes: Number of evaluation episodes.
            deterministic: Whether to use deterministic action selection.

        Returns:
            tuple[float, float]: (mean_reward, std_reward)
        """
        metrics_list: List[EpisodeMetrics] = []
        for ep in range(episodes):
            obs, info = self.env.reset(seed=self.config.seed + ep if self.config.seed else None)
            acc = EpisodeMetricsAccumulator(outcome_policy=self.outcome_policy)
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
            metrics_list.append(m)

        rewards = [m.reward for m in metrics_list]
        return float(np.mean(rewards)), float(np.std(rewards))


def get_trainer(
    config: ExperimentConfig,
    env: Optional[gym.Env] = None,
    callbacks: Optional[List[BaseCallback]] = None,
) -> BaseTrainer:
    """Create appropriate trainer instance based on algorithm and curriculum in config."""
    if config.curriculum is not None and config.curriculum.enabled:
        from adaptive_rl.curriculum.trainer import CurriculumTrainer

        return CurriculumTrainer(config=config, env=env, callbacks=callbacks)

    algo_name = config.algorithm.name.lower()
    if algo_name == "ppo":
        return PPOTrainer(config=config, env=env, callbacks=callbacks)
    elif algo_name == "sac":
        return SACTrainer(config=config, env=env, callbacks=callbacks)
    else:
        raise ValueError(
            f"Unsupported algorithm '{config.algorithm.name}'. Supported algorithms: 'ppo', 'sac'"
        )
