"""Comprehensive tests for PPO algorithm wrapper, callbacks, checkpointing, and PPOTrainer."""

from pathlib import Path

import numpy as np
import pytest
import torch

from adaptive_rl.algorithms.ppo import PPOAlgorithm
from adaptive_rl.config import (
    AlgorithmConfig,
    EnvironmentConfig,
    EvaluationConfig,
    ExperimentConfig,
    TrainingConfig,
)
from adaptive_rl.environments.gridworld.grid import GridWorldEnv
from adaptive_rl.environments.testing import DummyTestEnv
from adaptive_rl.training.callbacks import (
    MetricLoggerCallback,
)
from adaptive_rl.training.checkpointing import CheckpointManager
from adaptive_rl.training.trainer import PPOTrainer, TrainingResult


def test_ppo_algorithm_init_default() -> None:
    """Verify default initialization of PPOAlgorithm with environment."""
    env = DummyTestEnv(step_limit=10)
    algo = PPOAlgorithm(env=env)
    assert algo.model is not None
    assert algo.hyperparameters["learning_rate"] == 3e-4
    assert algo.hyperparameters["gamma"] == 0.99
    assert algo.num_timesteps == 0
    env.close()


def test_ppo_algorithm_custom_hyperparameters() -> None:
    """Verify custom hyperparameters are forwarded correctly to PPO."""
    env = DummyTestEnv(step_limit=10)
    algo = PPOAlgorithm(
        env=env,
        learning_rate=1e-3,
        n_steps=64,
        batch_size=32,
        n_epochs=5,
        gamma=0.95,
        ent_coef=0.02,
        clip_range=0.1,
    )
    assert algo.hyperparameters["learning_rate"] == 1e-3
    assert algo.hyperparameters["n_steps"] == 64
    assert algo.hyperparameters["batch_size"] == 32
    assert algo.hyperparameters["n_epochs"] == 5
    assert algo.hyperparameters["gamma"] == 0.95
    assert algo.hyperparameters["ent_coef"] == 0.02
    assert algo.hyperparameters["clip_range"] == 0.1
    env.close()


def test_ppo_train_and_predict_dummy_env() -> None:
    """Verify training for small step budget on DummyTestEnv and predicting actions."""
    env = DummyTestEnv(step_limit=10)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32)
    algo.train(total_timesteps=128)

    assert algo.num_timesteps >= 128
    obs, _ = env.reset(seed=42)
    action, _ = algo.predict(obs, deterministic=True)
    assert env.action_space.contains(action)
    env.close()


def test_ppo_train_and_predict_gridworld() -> None:
    """Verify training PPO directly on GridWorld environment."""
    env = GridWorldEnv(width=5, height=5, num_obstacles=2, max_steps=20)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32)
    algo.train(total_timesteps=128)

    assert algo.num_timesteps >= 128
    obs, _ = env.reset(seed=100)
    action, _ = algo.predict(obs, deterministic=True)
    assert action in (0, 1, 2, 3)

    next_obs, reward, terminated, truncated, _ = env.step(action)
    assert next_obs.shape == (4,)
    assert isinstance(reward, float)
    env.close()


def test_ppo_save_and_load_roundtrip(tmp_path: Path) -> None:
    """Verify model save and reload produces identical deterministic action predictions."""
    env = DummyTestEnv(step_limit=10)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32, seed=42)
    algo.train(total_timesteps=64)

    test_obs = np.array([0.5, 0.5], dtype=np.float32)
    action1, _ = algo.predict(test_obs, deterministic=True)

    save_path = tmp_path / "saved_ppo.zip"
    algo.save(save_path)
    assert save_path.exists()

    # Load into new instance via from_pretrained
    loaded_algo = PPOAlgorithm.from_pretrained(save_path, env=env)
    action2, _ = loaded_algo.predict(test_obs, deterministic=True)

    assert action1 == action2
    env.close()


def test_ppo_uninitialized_model_raises() -> None:
    """Verify calling methods on uninitialized PPOAlgorithm raises appropriate exceptions."""
    algo = PPOAlgorithm(env=None)
    with pytest.raises(RuntimeError, match="model is not initialized"):
        algo.train(100)

    with pytest.raises(RuntimeError, match="model is not initialized"):
        algo.predict(np.zeros(2))

    with pytest.raises(RuntimeError, match="model is not initialized"):
        algo.save("non_existent_path.zip")

    with pytest.raises(FileNotFoundError, match="Model file not found"):
        algo.load("totally_missing_model_file.zip")


def test_checkpoint_manager_lifecycle(tmp_path: Path) -> None:
    """Verify CheckpointManager saving, tracking, latest, best, and clearing."""
    cp_dir = tmp_path / "checkpoints"
    manager = CheckpointManager(checkpoint_dir=cp_dir)
    assert cp_dir.exists()
    assert manager.list_checkpoints() == []
    assert manager.get_latest_checkpoint() is None
    assert manager.get_best_checkpoint() is None

    env = DummyTestEnv(step_limit=10)
    algo = PPOAlgorithm(env=env, n_steps=64, batch_size=32)

    # Save first checkpoint
    path1 = manager.save_checkpoint(model=algo, step=100, metric_value=-10.0)
    assert path1.exists()
    assert len(manager.list_checkpoints()) == 1
    assert manager.get_latest_checkpoint()["step"] == 100

    # Save second checkpoint with better metric
    path2 = manager.save_checkpoint(model=algo, step=200, metric_value=25.0)
    assert path2.exists()
    assert len(manager.list_checkpoints()) == 2
    assert manager.get_latest_checkpoint()["step"] == 200
    assert manager.get_best_checkpoint()["step"] == 200
    assert manager.get_best_checkpoint()["metric_value"] == 25.0

    manager.clear()
    assert manager.list_checkpoints() == []
    env.close()


def test_metric_logger_callback() -> None:
    """Verify MetricLoggerCallback computes rolling statistics, success, and collision rates."""
    logger = MetricLoggerCallback(window_size=3)
    assert logger.total_episodes == 0
    assert logger.mean_reward == 0.0

    # Episode 1: Success
    logger.on_episode_end(
        episode=1, episode_reward=100.0, episode_length=10, info={"success": True}
    )
    # Episode 2: Collision
    logger.on_episode_end(
        episode=2, episode_reward=-100.0, episode_length=5, info={"collision": True}
    )
    # Episode 3: Normal finish
    logger.on_episode_end(episode=3, episode_reward=50.0, episode_length=20, info={})

    assert logger.total_episodes == 3
    assert logger.successes == 1
    assert logger.collisions == 1
    # Defined episodes for success: 1 (Ep 1: True, Ep 2: None, Ep 3: None) -> 1/1 = 1.0
    # Defined episodes for collision: 1 (Ep 1: None, Ep 2: True, Ep 3: None) -> 1/1 = 1.0
    assert logger.success_rate == 1.0
    assert logger.collision_rate == 1.0
    assert pytest.approx(logger.mean_reward) == (100.0 - 100.0 + 50.0) / 3
    assert pytest.approx(logger.mean_length) == (10 + 5 + 20) / 3


def test_ppo_trainer_fit_end_to_end(tmp_path: Path) -> None:
    """Verify PPOTrainer trains agent, logs metrics, and produces valid artifacts."""
    config = ExperimentConfig(
        name="test_trainer_run",
        seed=123,
        algorithm=AlgorithmConfig(
            name="ppo",
            learning_rate=3e-4,
            gamma=0.99,
            batch_size=32,
            parameters={"n_steps": 64, "n_epochs": 2},
        ),
        environment=EnvironmentConfig(
            name="gridworld",
            max_steps=20,
            parameters={"width": 5, "height": 5, "num_obstacles": 2},
        ),
        training=TrainingConfig(
            total_timesteps=128,
            checkpoint_freq=64,
            log_interval=10,
        ),
        evaluation=EvaluationConfig(eval_episodes=2, deterministic=True),
        output_dir=tmp_path / "results",
        log_dir=tmp_path / "logs",
    )

    trainer = PPOTrainer(config=config)
    result = trainer.fit()

    assert isinstance(result, TrainingResult)
    assert result.experiment_name == "test_trainer_run"
    assert result.total_timesteps == 128
    assert result.final_model_path.exists()
    assert len(result.checkpoints) >= 1

    # Verify trainer evaluation method
    mean_eval, std_eval = trainer.evaluate(episodes=2)
    assert isinstance(mean_eval, float)
    assert isinstance(std_eval, float)


def test_ppo_deterministic_seeding() -> None:
    """Verify deterministic seed initialization produces identical initial weights."""
    env1 = DummyTestEnv(step_limit=10)
    PPOTrainer._set_deterministic_seed(42)
    algo1 = PPOAlgorithm(env=env1, seed=42)

    env2 = DummyTestEnv(step_limit=10)
    PPOTrainer._set_deterministic_seed(42)
    algo2 = PPOAlgorithm(env=env2, seed=42)

    # Policy network weights should be bitwise identical
    for p1, p2 in zip(algo1.model.policy.parameters(), algo2.model.policy.parameters()):
        torch.testing.assert_close(p1, p2)

    env1.close()
    env2.close()
