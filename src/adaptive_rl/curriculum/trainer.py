"""Curriculum training engine coordinating staged environment progression and RL optimization."""

from __future__ import annotations

import json
import random
from typing import List, Optional

import gymnasium as gym
import numpy as np
import torch

from adaptive_rl.algorithms.base import BaseAlgorithm
from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.algorithms.sac import SACAlgorithm
from adaptive_rl.config import ExperimentConfig
from adaptive_rl.curriculum.callbacks import CurriculumCallback
from adaptive_rl.curriculum.curriculum import Curriculum
from adaptive_rl.curriculum.presets import get_curriculum_preset
from adaptive_rl.curriculum.stage import CurriculumStage
from adaptive_rl.curriculum.wrapper import CurriculumEnvWrapper
from adaptive_rl.environments.registry import make_env
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
from adaptive_rl.training.trainer import BaseTrainer, TrainingResult


class CurriculumTrainer(BaseTrainer):
    """Orchestrates end-to-end reinforcement learning training with automated curriculum progression."""

    def __init__(
        self,
        config: ExperimentConfig,
        curriculum: Optional[Curriculum] = None,
        env: Optional[gym.Env] = None,
        callbacks: Optional[List[BaseCallback]] = None,
    ) -> None:
        """Initialize CurriculumTrainer.

        Args:
            config: Validated ExperimentConfig.
            curriculum: Optional explicit Curriculum instance.
            env: Optional base Gymnasium environment.
            callbacks: Optional additional callbacks.
        """
        self.config = config
        self._set_deterministic_seed(self.config.seed)

        # 1. Resolve curriculum
        if curriculum is not None:
            self.curriculum = curriculum
        elif self.config.curriculum is not None and self.config.curriculum.stages:
            stages = [
                CurriculumStage(
                    stage_id=i,
                    name=s.name,
                    environment_parameters=dict(s.environment_parameters),
                    success_threshold=s.success_threshold,
                    mean_reward_threshold=s.mean_reward_threshold,
                    max_timesteps=s.max_timesteps,
                    min_episodes=s.min_episodes,
                    description=s.description,
                )
                for i, s in enumerate(self.config.curriculum.stages)
            ]
            self.curriculum = Curriculum(
                name=f"{self.config.name}_curriculum",
                stages=stages,
                eval_window=self.config.curriculum.eval_window,
            )
        elif self.config.curriculum is not None and self.config.curriculum.preset:
            self.curriculum = get_curriculum_preset(
                self.config.curriculum.preset,
                eval_window=self.config.curriculum.eval_window,
            )
        else:
            # Default to preset matching environment name
            env_name = self.config.environment.name.lower()
            if "disturb" in env_name or "constrain" in env_name:
                self.curriculum = get_curriculum_preset("drone_disturbed")
            elif "drone" in env_name:
                self.curriculum = get_curriculum_preset("drone")
            elif "traffic" in env_name:
                self.curriculum = get_curriculum_preset("traffic")
            elif "nav" in env_name:
                self.curriculum = get_curriculum_preset("navigation")
            elif "grid" in env_name:
                self.curriculum = get_curriculum_preset("gridworld")
            else:
                raise ValueError(
                    f"No curriculum provided and no default preset found for env '{self.config.environment.name}'."
                )

        # 2. Environment creation & wrapper
        if env is not None:
            raw_env = env
        else:
            env_params = dict(self.config.environment.parameters)
            if "max_steps" not in env_params:
                env_params["max_steps"] = self.config.environment.max_steps
            raw_env = make_env(
                self.config.environment.name,
                **env_params,
            )

        self.env_wrapper = CurriculumEnvWrapper(env=raw_env, curriculum=self.curriculum)
        self.env: gym.Env = self.env_wrapper

        # 3. Checkpointing setup
        checkpoint_dir = self.config.output_dir / "checkpoints" / self.config.name
        self.checkpoint_manager = CheckpointManager(checkpoint_dir=checkpoint_dir)

        # 4. Callbacks setup
        self.metric_logger = MetricLoggerCallback()
        self.curriculum_callback = CurriculumCallback(
            curriculum=self.curriculum,
            env_wrapper=self.env_wrapper,
            verbose=1,
        )
        if self.config.training is None:
            raise ValueError(
                "Training configuration ('training') is required for curriculum training."
            )
        training_cfg = self.config.training

        if "traffic" in self.config.environment.name.lower():
            self.outcome_policy: OutcomePolicy = TrafficOutcomePolicy()
        else:
            self.outcome_policy = DefaultOutcomePolicy()

        self._callbacks: List[BaseCallback] = [self.metric_logger, self.curriculum_callback]

        if training_cfg.checkpoint_freq > 0:
            checkpoint_cb = CheckpointCallback(
                checkpoint_manager=self.checkpoint_manager,
                save_freq=training_cfg.checkpoint_freq,
            )
            self._callbacks.append(checkpoint_cb)

        if callbacks:
            self._callbacks.extend(callbacks)

        # 5. Algorithm initialization
        algo_name = self.config.algorithm.name.lower()
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

        self.algorithm: BaseAlgorithm
        if algo_name == "ppo":
            self.algorithm = PPOAlgorithm(
                env=self.env,
                learning_rate=lr,
                gamma=gamma,
                batch_size=batch_size,
                seed=self.config.seed,
                **algo_params,
            )
        elif algo_name == "sac":
            self.algorithm = SACAlgorithm(
                env=self.env,
                learning_rate=lr,
                gamma=gamma,
                batch_size=batch_size,
                seed=self.config.seed,
                **algo_params,
            )
        else:
            raise ValueError(
                f"Unsupported algorithm '{self.config.algorithm.name}' for curriculum training."
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
        """Execute end-to-end curriculum training process.

        Returns:
            TrainingResult: Structured summary of curriculum training run.
        """
        adapter = SB3CallbackAdapter(
            callbacks=self._callbacks,
            algorithm=self.algorithm,
            outcome_policy=self.outcome_policy,
        )

        assert self.config.training is not None
        # Train algorithm with automated stage progression
        self.algorithm.train(
            total_timesteps=self.config.training.total_timesteps,
            callback=adapter,
        )

        # Save final model
        models_dir = self.config.output_dir / "models"
        models_dir.mkdir(parents=True, exist_ok=True)
        final_model_path = models_dir / f"{self.config.name}_final.zip"
        self.algorithm.save(final_model_path)

        # Save curriculum report
        curriculum_dir = self.config.output_dir / "curriculum"
        curriculum_dir.mkdir(parents=True, exist_ok=True)
        curriculum_report_path = curriculum_dir / f"{self.config.name}_curriculum.json"
        curriculum_report_path.write_text(
            json.dumps(self.curriculum.to_dict(), indent=2),
            encoding="utf-8",
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
        )
        return result

    def evaluate(
        self,
        episodes: int = 10,
        deterministic: bool = True,
    ) -> tuple[float, float]:
        """Evaluate policy on current curriculum stage using canonical EpisodeMetrics."""
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
