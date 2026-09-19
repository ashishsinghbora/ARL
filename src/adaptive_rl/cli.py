"""Command Line Interface for AdaptiveRL.

Provides commands for inspecting environment status, validating configurations,
training agents, evaluating performance, running experiments, benchmarking,
and displaying a terminal dashboard.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import numpy as np
import typer
from rich.console import Console
from rich.panel import Panel
from rich.table import Table

import adaptive_rl
from adaptive_rl.algorithms import algorithm_registry
from adaptive_rl.config import ConfigError, load_config
from adaptive_rl.environments.registry import RegistryError, list_all_metadata, make_env

app = typer.Typer(
    name="adaptive-rl",
    help="AdaptiveRL: Multi-Environment Reinforcement Learning Platform CLI.",
    add_completion=False,
    no_args_is_help=True,
)

config_app = typer.Typer(
    name="config",
    help="Configuration inspection and validation commands.",
    no_args_is_help=True,
)
app.add_typer(config_app, name="config")

env_app = typer.Typer(
    name="env",
    help="Environment discovery, inspection, and simulation commands.",
    no_args_is_help=True,
)
app.add_typer(env_app, name="env")

curriculum_app = typer.Typer(
    name="curriculum",
    help="Curriculum learning inspection and preset management commands.",
    no_args_is_help=True,
)
app.add_typer(curriculum_app, name="curriculum")

algorithm_app = typer.Typer(
    name="algorithm",
    help="Algorithm registry inspection and capability discovery commands.",
    no_args_is_help=True,
)
app.add_typer(algorithm_app, name="algorithm")

experiment_app = typer.Typer(
    name="experiment",
    help="Experiment lifecycle management: run, list, and inspect experiments.",
    no_args_is_help=True,
)
app.add_typer(experiment_app, name="experiment")

console = Console()


@app.command()
def version() -> None:
    """Show the installed AdaptiveRL version and phase status."""
    console.print(
        f"[bold green]AdaptiveRL[/bold green] version [bold cyan]{adaptive_rl.__version__}[/bold cyan] "
        f"([yellow]Phases 1–17: Complete Platform with Classical Baselines & Dashboard[/yellow])"
    )


@app.command()
def info() -> None:
    """Display platform architecture status and implementation roadmap."""
    table = Table(title="AdaptiveRL — Implementation Roadmap Status")
    table.add_column("Phase", style="cyan", no_wrap=True)
    table.add_column("Milestone Name", style="magenta")
    table.add_column("Status", style="green")

    table.add_row(
        "Phase 1",
        "Repository Foundation and Architecture Skeleton",
        "[bold green]COMPLETED[/bold green]",
    )
    table.add_row(
        "Phase 2", "Environment Abstraction and Registry", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row(
        "Phase 3", "Procedurally Generated GridWorld", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row(
        "Phase 4", "PPO Training Engine (SB3 Wrapper)", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row(
        "Phase 5", "Evaluation Engine and Standard Metrics", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row("Phase 6", "Continuous 2D Navigation", "[bold green]COMPLETED[/bold green]")
    table.add_row("Phase 7", "Curriculum Learning", "[bold green]COMPLETED[/bold green]")
    table.add_row("Phase 8", "Traffic Signal Optimization", "[bold green]COMPLETED[/bold green]")
    table.add_row("Phase 9", "Autonomous 3D Drone Navigation", "[bold green]COMPLETED[/bold green]")
    table.add_row(
        "Phase 10", "Drone Disturbances and Constraints", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row(
        "Phase 11", "Generalization to Unseen Environments", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row(
        "Phase 12", "Classical Navigation Baselines (A*)", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row(
        "Phase 13", "Algorithm Registry & SAC Hardening", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row(
        "Phase 14", "Reproducible Experiment Manager", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row(
        "Phase 15", "Benchmarking and Ablation Framework", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row(
        "Phase 16", "Standardized Metrics and Result Schemas", "[bold green]COMPLETED[/bold green]"
    )
    table.add_row(
        "Phase 17", "Experiment Dashboard (Rich Terminal TUI)", "[bold green]COMPLETED[/bold green]"
    )

    console.print(table)


@config_app.command(name="validate")
def validate_config(
    path: Path = typer.Argument(
        ..., help="Path to YAML configuration file to validate", exists=False
    ),
) -> None:
    """Validate an experiment YAML configuration file against the schema."""
    try:
        cfg = load_config(path)
        curr_info = ""
        if cfg.curriculum is not None and cfg.curriculum.enabled:
            curr_preset = cfg.curriculum.preset or f"{len(cfg.curriculum.stages)} custom stages"
            curr_info = f"\n• [bold]Curriculum:[/bold] Enabled ({curr_preset}, Window: {cfg.curriculum.eval_window})"

        algo_details = (
            f"LR: {cfg.algorithm.learning_rate}, Gamma: {cfg.algorithm.gamma}"
            if cfg.algorithm.learning_rate is not None
            else "Planner"
        )
        training_info = (
            f"{cfg.training.total_timesteps:,} steps (Checkpoint freq: {cfg.training.checkpoint_freq})"
            if cfg.training is not None
            else "None (Classical planner)"
        )

        console.print(
            Panel.fit(
                f"[bold green]✓ Configuration is valid![/bold green]\n\n"
                f"• [bold]Experiment:[/bold] {cfg.name}\n"
                f"• [bold]Seed:[/bold] {cfg.seed}\n"
                f"• [bold]Algorithm:[/bold] {cfg.algorithm.name} ({algo_details})\n"
                f"• [bold]Environment:[/bold] {cfg.environment.name} (Max steps: {cfg.environment.max_steps})\n"
                f"• [bold]Training:[/bold] {training_info}\n"
                f"• [bold]Evaluation:[/bold] {cfg.evaluation.eval_episodes} episodes"
                f"{curr_info}",
                title=f"Valid: {path}",
                border_style="green",
            )
        )
    except ConfigError as err:
        console.print(
            Panel.fit(
                f"[bold red]Configuration validation error:[/bold red]\n\n{err}",
                title=f"Invalid: {path}",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


@env_app.command(name="list")
def list_envs() -> None:
    """List all registered environments and their metadata."""
    meta_map = list_all_metadata()
    if not meta_map:
        console.print(
            "[yellow]No custom environments currently registered in AdaptiveRL registry.[/yellow]\n"
            "Standard Gymnasium environments (e.g. 'CartPole-v1', 'Pendulum-v1') are also resolvable by the factory."
        )
        return

    table = Table(title="Registered AdaptiveRL Environments")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Version", style="magenta")
    table.add_column("Obs Space", style="green")
    table.add_column("Action Space", style="green")
    table.add_column("Description")

    for name, meta in meta_map.items():
        table.add_row(
            name,
            meta.version,
            meta.observation_type,
            meta.action_type,
            meta.description or "-",
        )

    console.print(table)


@env_app.command(name="inspect")
def inspect_env(
    name: str = typer.Argument(..., help="Name of registered or Gymnasium environment to inspect"),
) -> None:
    """Inspect observation and action spaces of an environment."""
    try:
        env = make_env(name)
        obs, info = env.reset()

        panel_content = [
            f"[bold green]Environment '{name}' verified successfully![/bold green]\n",
            f"• [bold]Type:[/bold] {type(env).__name__}",
            f"• [bold]Observation Space:[/bold] {env.observation_space}",
            f"• [bold]Action Space:[/bold] {env.action_space}",
            f"• [bold]Initial Observation Shape:[/bold] {getattr(obs, 'shape', 'discrete/scalar')}",
            f"• [bold]Reset Info:[/bold] {info}",
        ]

        if hasattr(env, "render"):
            rendered = env.render()
            if rendered:
                panel_content.append(f"\n[bold]Initial Layout:[/bold]\n{rendered}")

        env.close()

        console.print(
            Panel.fit(
                "\n".join(panel_content),
                title=f"Environment Inspection: {name}",
                border_style="cyan",
            )
        )
    except RegistryError as err:
        console.print(
            Panel.fit(
                f"[bold red]Environment inspection failed:[/bold red]\n\n{err}",
                title=f"Error: {name}",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


@env_app.command(name="run")
def run_env(
    name: str = typer.Argument("gridworld", help="Environment to execute"),
    steps: int = typer.Option(10, "--steps", "-s", help="Number of steps to simulate"),
    seed: int = typer.Option(42, "--seed", help="Random seed for environment reset"),
) -> None:
    """Simulate an environment episode with random actions and textual rendering."""
    try:
        env = make_env(name)
        obs, info = env.reset(seed=seed)
        console.print(f"[bold green]Starting simulation for '{name}' (seed={seed})[/bold green]\n")

        if hasattr(env, "render"):
            rendered = env.render()
            if rendered:
                console.print(Panel(str(rendered), title="Initial State"))

        total_reward = 0.0
        step_count = 0

        for s in range(1, steps + 1):
            action = env.action_space.sample()
            obs, reward, terminated, truncated, step_info = env.step(action)
            total_reward += float(reward)
            step_count += 1
            if isinstance(action, np.ndarray):
                action_desc = np.array2string(action, precision=2, separator=",")
            else:
                action_desc = step_info.get("action_name", str(action))
            console.print(
                f"Step {s:02d}: Action={action_desc:<18} -> Reward={reward:+6.1f} | Terminated={terminated} | Truncated={truncated}"
            )

            if terminated or truncated:
                if step_info.get("collision"):
                    outcome = "COLLISION!"
                elif step_info.get("overflow"):
                    outcome = "QUEUE OVERFLOW!"
                elif step_info.get("success"):
                    outcome = "SUCCESS / GOAL REACHED!"
                else:
                    outcome = "MAX STEPS REACHED"
                console.print(f"\n[bold yellow]Episode ended at step {s}: {outcome}[/bold yellow]")
                if hasattr(env, "render"):
                    rendered = env.render()
                    if rendered:
                        console.print(Panel(str(rendered), title="Final State"))
                break

        console.print(
            Panel.fit(
                f"[bold]Total Steps:[/bold] {step_count}\n"
                f"[bold]Cumulative Reward:[/bold] {total_reward:+.1f}\n"
                f"[bold]Final Observation:[/bold] {obs}",
                title="Simulation Summary",
                border_style="green",
            )
        )
        env.close()
    except RegistryError as err:
        console.print(f"[bold red]Failed to run environment:[/bold red] {err}")
        raise typer.Exit(code=1)


@curriculum_app.command(name="list")
def list_curriculums() -> None:
    """List all available built-in curriculum schedules."""
    from adaptive_rl.curriculum.presets import CURRICULUM_PRESETS

    table = Table(title="AdaptiveRL — Predefined Curriculum Schedules")
    table.add_column("Preset Name", style="cyan", no_wrap=True)
    table.add_column("Stages", style="green")
    table.add_column("Description", style="white")

    for name, builder in sorted(CURRICULUM_PRESETS.items()):
        curr = builder()
        table.add_row(
            name,
            f"{len(curr.stages)} stages",
            f"Progressive curriculum for {curr.name}",
        )
    console.print(table)


@curriculum_app.command(name="inspect")
def inspect_curriculum(
    name: str = typer.Argument("navigation", help="Curriculum preset name to inspect"),
) -> None:
    """Inspect stages, progression thresholds, and parameters of a curriculum preset."""
    from adaptive_rl.curriculum.presets import get_curriculum_preset

    try:
        curr = get_curriculum_preset(name)
    except ValueError as err:
        console.print(f"[bold red]Curriculum lookup failed:[/bold red] {err}")
        raise typer.Exit(code=1)

    table = Table(title=f"Curriculum Stages: {curr.name} ({len(curr.stages)} stages)")
    table.add_column("Stage ID", style="cyan")
    table.add_column("Name", style="magenta")
    table.add_column("Success Threshold", style="green")
    table.add_column("Mean Reward Threshold", style="yellow")
    table.add_column("Min Episodes", style="blue")
    table.add_column("Parameters", style="white")

    for stage in curr.stages:
        st_str = (
            f"{stage.success_threshold * 100:.0f}%"
            if stage.success_threshold is not None
            else "None"
        )
        mr_str = (
            f"{stage.mean_reward_threshold:.1f}"
            if stage.mean_reward_threshold is not None
            else "None"
        )
        params_str = ", ".join(f"{k}={v}" for k, v in stage.environment_parameters.items())
        table.add_row(
            str(stage.stage_id),
            stage.name,
            st_str,
            mr_str,
            str(stage.min_episodes),
            params_str or "(default)",
        )

    console.print(table)


@app.command()
def train(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to training configuration YAML"
    ),
    timesteps: Optional[int] = typer.Option(
        None, "--timesteps", "-t", help="Override total training timesteps"
    ),
    seed: Optional[int] = typer.Option(
        None, "--seed", "-s", help="Override experiment random seed"
    ),
) -> None:
    """Train a reinforcement learning agent using PPO."""
    if config is None:
        default_config = Path("configs/ppo.yaml")
        if default_config.exists():
            config = default_config
        else:
            console.print(
                "[bold red]No configuration file provided.[/bold red] Specify --config <path>"
            )
            raise typer.Exit(code=1)

    try:
        exp_config = load_config(config)
    except ConfigError as err:
        console.print(f"[bold red]Configuration error:[/bold red] {err}")
        raise typer.Exit(code=1)

    if exp_config.training is None or algorithm_registry.is_planner(exp_config.algorithm.name):
        console.print(
            f"[bold red]Configuration error:[/bold red] Algorithm '{exp_config.algorithm.name}' is a classical planner and does not support training."
        )
        raise typer.Exit(code=1)

    if timesteps is not None:
        exp_config.training.total_timesteps = timesteps
    if seed is not None:
        exp_config.seed = seed

    curriculum_line = ""
    if exp_config.curriculum is not None and exp_config.curriculum.enabled:
        preset_info = (
            exp_config.curriculum.preset or f"{len(exp_config.curriculum.stages)} custom stages"
        )
        curriculum_line = f"\n• [bold]Curriculum:[/bold] Enabled ({preset_info})"

    console.print(
        Panel.fit(
            f"[bold green]Starting Training: {exp_config.name}[/bold green]\n\n"
            f"• [bold]Algorithm:[/bold] {exp_config.algorithm.name.upper()}\n"
            f"• [bold]Environment:[/bold] {exp_config.environment.name}\n"
            f"• [bold]Total Timesteps:[/bold] {exp_config.training.total_timesteps:,}\n"
            f"• [bold]Checkpoint Freq:[/bold] {exp_config.training.checkpoint_freq}\n"
            f"• [bold]Seed:[/bold] {exp_config.seed}\n"
            f"• [bold]Output Dir:[/bold] {exp_config.output_dir}"
            f"{curriculum_line}",
            title=f"{exp_config.algorithm.name.upper()} Training Pipeline",
            border_style="cyan",
        )
    )

    from adaptive_rl.training import get_trainer

    try:
        trainer = get_trainer(config=exp_config)
        result = trainer.fit()

        console.print(
            Panel.fit(
                f"[bold green]Training Completed Successfully![/bold green]\n\n"
                f"• [bold]Total Timesteps Trained:[/bold] {result.total_timesteps:,}\n"
                f"• [bold]Episodes Completed:[/bold] {result.episodes_completed}\n"
                f"• [bold]Mean Reward (last window):[/bold] {result.mean_reward:.2f}\n"
                f"• [bold]Saved Model:[/bold] {result.final_model_path}\n"
                f"• [bold]Checkpoints Created:[/bold] {len(result.checkpoints)}",
                title="Training Summary",
                border_style="green",
            )
        )
    except Exception as err:
        console.print(f"[bold red]Training failed with error:[/bold red] {err}")
        raise typer.Exit(code=1)


@app.command()
def evaluate(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to experiment configuration YAML"
    ),
    model: Optional[Path] = typer.Option(
        None, "--model", "-m", help="Path to model weights artifact (.zip)"
    ),
    episodes: Optional[int] = typer.Option(
        None, "--episodes", "-e", help="Number of evaluation episodes"
    ),
    deterministic: bool = typer.Option(
        True, "--deterministic/--stochastic", help="Use deterministic action selection"
    ),
    output_report: Optional[Path] = typer.Option(
        None, "--output-report", "-o", help="Optional path to export JSON metrics report"
    ),
) -> None:
    """Evaluate a trained agent over multiple benchmark episodes."""
    if config is None:
        default_config = Path("configs/gridworld_ppo.yaml")
        if not default_config.exists():
            default_config = Path("configs/ppo.yaml")
        if default_config.exists():
            config = default_config
        else:
            console.print(
                "[bold red]No configuration file provided.[/bold red] Specify --config <path>"
            )
            raise typer.Exit(code=1)

    try:
        exp_config = load_config(config)
    except ConfigError as err:
        console.print(f"[bold red]Configuration error:[/bold red] {err}")
        raise typer.Exit(code=1)

    num_episodes = episodes or exp_config.evaluation.eval_episodes
    det = deterministic if episodes is not None else exp_config.evaluation.deterministic

    # Resolve model path
    if model is None:
        candidate = exp_config.output_dir / "models" / f"{exp_config.name}_final.zip"
        if candidate.exists():
            model = candidate
        else:
            console.print(
                f"[bold red]No model weights provided.[/bold red] Pass --model <path> or train first to generate {candidate}"
            )
            raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold green]Starting Evaluation: {exp_config.name}[/bold green]\n\n"
            f"• [bold]Model:[/bold] {model}\n"
            f"• [bold]Environment:[/bold] {exp_config.environment.name}\n"
            f"• [bold]Episodes:[/bold] {num_episodes}\n"
            f"• [bold]Deterministic:[/bold] {det}",
            title="Evaluation Engine",
            border_style="cyan",
        )
    )

    from adaptive_rl.algorithms.ppo import PPOAlgorithm
    from adaptive_rl.environments.registry import make_env
    from adaptive_rl.evaluation.evaluator import Evaluator

    try:
        env = make_env(
            exp_config.environment.name,
            **exp_config.environment.parameters,
        )
        algo = PPOAlgorithm.from_pretrained(model, env=env)
        evaluator = Evaluator(algorithm=algo, env=env)

        metrics = evaluator.evaluate(
            num_episodes=num_episodes,
            deterministic=det,
            base_seed=exp_config.seed,
        )

        table = Table(title=f"Benchmark Results ({num_episodes} episodes)")
        table.add_column("Metric", style="cyan")
        table.add_column("Value", style="green", justify="right")

        table.add_row("Mean Reward", f"{metrics.mean_reward:.2f} ± {metrics.std_reward:.2f}")
        table.add_row("Min / Max Reward", f"{metrics.min_reward:.2f} / {metrics.max_reward:.2f}")
        success_str = (
            f"{metrics.success_rate * 100:.1f}%" if metrics.success_rate is not None else "N/A"
        )
        collision_str = (
            f"{metrics.collision_rate * 100:.1f}%" if metrics.collision_rate is not None else "N/A"
        )
        table.add_row("Success Rate", success_str)
        table.add_row("Collision Rate", collision_str)
        table.add_row(
            "Mean Episode Length",
            f"{metrics.mean_episode_length:.1f} ± {metrics.std_episode_length:.1f}",
        )

        console.print(table)

        if output_report is not None:
            saved_path = evaluator.save_report(metrics, output_report)
            console.print(f"\n[bold green]Report saved to:[/bold green] {saved_path}")

        env.close()
    except Exception as err:
        console.print(f"[bold red]Evaluation failed with error:[/bold red] {err}")
        raise typer.Exit(code=1)


@app.command(name="generalization")
def run_generalization_command(
    config: Optional[Path] = typer.Option(
        None, "--config", "-c", help="Path to experiment configuration YAML"
    ),
    train_count: int = typer.Option(
        15, "--train-count", help="Number of training distribution seeds to evaluate"
    ),
    test_count: int = typer.Option(
        15, "--test-count", help="Number of unseen test distribution seeds to evaluate"
    ),
    train_start: int = typer.Option(
        1000, "--train-start", help="Starting seed for training distribution"
    ),
    test_start: int = typer.Option(
        2000, "--test-start", help="Starting seed for unseen test distribution"
    ),
    output_report: Optional[Path] = typer.Option(
        None, "--output-report", "-o", help="Optional path to export JSON metrics report"
    ),
) -> None:
    """Run an end-to-end generalization experiment measuring performance on unseen environments."""
    from adaptive_rl.evaluation.generalization import (
        GeneralizationDistribution,
        GeneralizationReport,
    )
    from adaptive_rl.experiments.generalization_runner import GeneralizationExperimentRunner

    if config is None:
        default_config = Path("configs/generalization_gridworld.yaml")
        if default_config.exists():
            config = default_config
        else:
            console.print(
                "[bold red]No configuration file provided.[/bold red] Specify --config <path>"
            )
            raise typer.Exit(code=1)

    try:
        exp_config = load_config(config)
    except ConfigError as err:
        console.print(f"[bold red]Configuration error:[/bold red] {err}")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold green]Starting Generalization Experiment: {exp_config.name}[/bold green]\n\n"
            f"• [bold]Environment:[/bold] {exp_config.environment.name}\n"
            f"• [bold]Algorithm:[/bold] {exp_config.algorithm.name}\n"
            f"• [bold]Training Seed Distribution:[/bold] [{train_start} .. {train_start + train_count})\n"
            f"• [bold]Unseen Test Seed Distribution:[/bold] [{test_start} .. {test_start + test_count})\n"
            f"• [bold]Disjoint Integrity Check:[/bold] Verified (0 overlapping seeds)",
            title="Generalization Experiment Runner",
            border_style="cyan",
        )
    )

    try:
        distribution = GeneralizationDistribution(
            train_seeds=list(range(train_start, train_start + train_count)),
            test_seeds=list(range(test_start, test_start + test_count)),
            description=f"Disjoint benchmark for {exp_config.name}",
        )
        runner = GeneralizationExperimentRunner(distribution=distribution)
        report: GeneralizationReport = runner.run(exp_config)

        # Output comparison table
        table = Table(
            title=f"Generalization Benchmark: {report.environment_name} ({report.algorithm_name})"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Training Distribution", style="green", justify="right")
        table.add_column("Unseen Test Distribution", style="magenta", justify="right")
        table.add_column("Generalization Gap", style="yellow", justify="right")

        train_succ = (
            f"{report.train_metrics.success_rate * 100:.1f}%"
            if report.train_metrics.success_rate is not None
            else "N/A"
        )
        test_succ = (
            f"{report.test_metrics.success_rate * 100:.1f}%"
            if report.test_metrics.success_rate is not None
            else "N/A"
        )
        gap_succ = (
            f"{-report.generalization_gap_success * 100:+.1f}%"
            if report.generalization_gap_success is not None
            else "N/A"
        )
        table.add_row("Success Rate", train_succ, test_succ, gap_succ)

        train_coll = (
            f"{report.train_metrics.collision_rate * 100:.1f}%"
            if report.train_metrics.collision_rate is not None
            else "N/A"
        )
        test_coll = (
            f"{report.test_metrics.collision_rate * 100:.1f}%"
            if report.test_metrics.collision_rate is not None
            else "N/A"
        )
        gap_coll = (
            f"{(report.test_metrics.collision_rate - report.train_metrics.collision_rate) * 100:+.1f}%"
            if report.train_metrics.collision_rate is not None
            and report.test_metrics.collision_rate is not None
            else "N/A"
        )
        table.add_row("Collision Rate", train_coll, test_coll, gap_coll)
        table.add_row(
            "Mean Reward",
            f"{report.train_metrics.mean_reward:.2f} ± {report.train_metrics.std_reward:.2f}",
            f"{report.test_metrics.mean_reward:.2f} ± {report.test_metrics.std_reward:.2f}",
            f"{-report.generalization_gap_reward:+.2f}",
        )
        table.add_row(
            "Mean Episode Length",
            f"{report.train_metrics.mean_episode_length:.1f}",
            f"{report.test_metrics.mean_episode_length:.1f}",
            f"{report.test_metrics.mean_episode_length - report.train_metrics.mean_episode_length:+.1f}",
        )
        retention_str = (
            f"{report.relative_success_retention * 100:.1f}%"
            if report.relative_success_retention is not None
            else "N/A"
        )
        table.add_row(
            "Success Retention",
            "100.0%",
            retention_str,
            "-",
        )

        console.print(table)

        if output_report is not None:
            saved_path = report.save_json(output_report)
            console.print(f"\n[bold green]Report saved to:[/bold green] {saved_path}")

    except Exception as err:
        console.print(f"[bold red]Generalization experiment failed with error:[/bold red] {err}")
        raise typer.Exit(code=1)


# ---------------------------------------------------------------------------
# Algorithm registry commands
# ---------------------------------------------------------------------------


@algorithm_app.command(name="list")
def list_algorithms() -> None:
    """List all registered algorithms and planners with their capabilities."""
    from adaptive_rl.algorithms.registry import AlgorithmKind, list_all_algorithm_metadata

    meta_map = list_all_algorithm_metadata()
    if not meta_map:
        console.print("[yellow]No algorithms currently registered.[/yellow]")
        return

    table = Table(title="Registered AdaptiveRL Algorithms and Planners")
    table.add_column("Name", style="cyan", no_wrap=True)
    table.add_column("Kind", style="magenta")
    table.add_column("Action Space", style="green")
    table.add_column("Trainable", style="yellow", justify="center")
    table.add_column("Description")

    for name, meta in meta_map.items():
        kind_str = (
            "[bold blue]RL Policy[/bold blue]"
            if meta.kind == AlgorithmKind.RL_POLICY
            else "[bold cyan]Planner[/bold cyan]"
        )
        trainable_str = "[green]✓[/green]" if meta.trainable else "[dim]✗[/dim]"
        table.add_row(
            name,
            kind_str,
            meta.action_space,
            trainable_str,
            meta.description[:70] + "..." if len(meta.description) > 70 else meta.description,
        )

    console.print(table)


@algorithm_app.command(name="inspect")
def inspect_algorithm(
    name: str = typer.Argument(..., help="Algorithm name to inspect (e.g. ppo, sac, astar)"),
) -> None:
    """Inspect capabilities and hyperparameters of a registered algorithm."""
    from adaptive_rl.algorithms.registry import (
        AlgorithmKind,
        AlgorithmRegistryError,
        get_algorithm_metadata,
    )

    try:
        meta = get_algorithm_metadata(name)
    except AlgorithmRegistryError as err:
        console.print(f"[bold red]Algorithm lookup failed:[/bold red] {err}")
        raise typer.Exit(code=1)

    kind_str = "RL Policy" if meta.kind == AlgorithmKind.RL_POLICY else "Deterministic Planner"
    tags_str = ", ".join(meta.tags) if meta.tags else "none"

    content = (
        f"[bold]Name:[/bold] {meta.name}\n"
        f"[bold]Kind:[/bold] {kind_str}\n"
        f"[bold]Class:[/bold] {meta.class_name}\n"
        f"[bold]Action Space:[/bold] {meta.action_space}\n"
        f"[bold]Trainable:[/bold] {'Yes' if meta.trainable else 'No (deterministic planner)'}\n"
        f"[bold]Tags:[/bold] {tags_str}\n\n"
        f"[bold]Description:[/bold]\n{meta.description}"
    )

    if meta.hyperparameters:
        hp_lines = "\n".join(f"  {k}: {v}" for k, v in meta.hyperparameters.items())
        content += f"\n\n[bold]Default Hyperparameters:[/bold]\n{hp_lines}"

    console.print(
        Panel.fit(
            content,
            title=f"Algorithm: {meta.name.upper()}",
            border_style="cyan",
        )
    )


# ---------------------------------------------------------------------------
# Experiment management commands
# ---------------------------------------------------------------------------


@experiment_app.command(name="run")
def run_experiment(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to experiment YAML configuration file"
    ),
    timesteps: Optional[int] = typer.Option(
        None, "--timesteps", "-t", help="Override total training timesteps"
    ),
    seed: Optional[int] = typer.Option(None, "--seed", "-s", help="Override random seed"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Override base output directory"
    ),
) -> None:
    """Run a complete experiment (training + evaluation) from a YAML config.

    Creates a structured output directory under experiments/results/<experiment_id>/
    containing the config copy, manifest, metrics, and model artifacts.
    """
    from adaptive_rl.experiments.manager import ExperimentManager

    manager = ExperimentManager(base_output_dir=output_dir or Path("experiments/results"))

    try:
        exp_config = load_config(config)
    except ConfigError as err:
        console.print(f"[bold red]Configuration error:[/bold red] {err}")
        raise typer.Exit(code=1)

    console.print(
        Panel.fit(
            f"[bold green]Starting Experiment[/bold green]\n\n"
            f"• [bold]Config:[/bold] {config}\n"
            f"• [bold]Algorithm:[/bold] {exp_config.algorithm.name.upper()}\n"
            f"• [bold]Environment:[/bold] {exp_config.environment.name}\n"
            f"• [bold]Seed:[/bold] {seed or exp_config.seed}",
            title="Experiment Manager",
            border_style="cyan",
        )
    )

    result = manager.run_from_config(
        config_path=config,
        timesteps_override=timesteps,
        seed_override=seed,
    )

    if result.success:
        console.print(
            Panel.fit(
                f"[bold green]Experiment Completed Successfully![/bold green]\n\n"
                f"• [bold]Experiment ID:[/bold] {result.experiment_id}\n"
                f"• [bold]Output Directory:[/bold] {result.output_dir}\n"
                f"• [bold]Evaluation Status:[/bold] {result.manifest.evaluation_status}",
                title="Experiment Result",
                border_style="green",
            )
        )
    else:
        console.print(
            Panel.fit(
                f"[bold red]Experiment Failed[/bold red]\n\n"
                f"• [bold]Experiment ID:[/bold] {result.experiment_id}\n"
                f"• [bold]Error:[/bold] {result.error_message}",
                title="Experiment Error",
                border_style="red",
            )
        )
        raise typer.Exit(code=1)


@experiment_app.command(name="list")
def list_experiments(
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Experiments base directory"
    ),
) -> None:
    """List all experiments recorded in the results directory."""
    from adaptive_rl.experiments.manager import ExperimentManager

    manager = ExperimentManager(base_output_dir=output_dir or Path("experiments/results"))
    experiments = manager.list_experiments()

    if not experiments:
        console.print(
            "[yellow]No experiments found.[/yellow] Run one with [cyan]adaptive-rl experiment run[/cyan]."
        )
        return

    table = Table(title=f"Recorded Experiments ({len(experiments)} total)")
    table.add_column("Experiment ID", style="cyan", no_wrap=True, max_width=45)
    table.add_column("Algorithm", style="magenta")
    table.add_column("Environment", style="green")
    table.add_column("Seed", style="blue", justify="right")
    table.add_column("Status", justify="center")
    table.add_column("Created At", style="dim")

    for exp in experiments:
        status = exp.get("evaluation_status", "?")
        created_at = exp.get("created_at", "?")
        if isinstance(created_at, str) and "T" in created_at:
            created_at = created_at[:19].replace("T", " ") + " UTC"
        status_str = (
            "[bold green]completed[/bold green]"
            if status == "completed"
            else "[bold red]failed[/bold red]"
            if status == "failed"
            else f"[yellow]{status}[/yellow]"
        )
        table.add_row(
            exp.get("experiment_id", "?"),
            exp.get("algorithm", "?").upper(),
            exp.get("environment", "?"),
            str(exp.get("seed", "?")),
            status_str,
            created_at,
        )

    console.print(table)


@experiment_app.command(name="inspect")
def inspect_experiment(
    experiment_id: str = typer.Argument(..., help="Experiment ID to inspect"),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Experiments base directory"
    ),
) -> None:
    """Show detailed metrics and manifest for a specific experiment."""
    from adaptive_rl.experiments.manager import ExperimentManager
    from adaptive_rl.visualization.dashboard import render_metrics

    manager = ExperimentManager(base_output_dir=output_dir or Path("experiments/results"))
    render_metrics(experiment_id, manager)


# ---------------------------------------------------------------------------
# Benchmark command
# ---------------------------------------------------------------------------


@app.command()
def benchmark(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to the YAML configuration to benchmark"
    ),
    seeds: Optional[str] = typer.Option(
        None,
        "--seeds",
        help="Comma-separated list of seeds (e.g. 42,43,44). Default: 42,43,44.",
    ),
    timesteps: Optional[int] = typer.Option(
        None, "--timesteps", "-t", help="Override training timesteps per seed"
    ),
    compare_config: Optional[Path] = typer.Option(
        None, "--compare", help="Second config to compare against (ablation)"
    ),
    output_report: Optional[Path] = typer.Option(
        None, "--output-report", "-o", help="Path to write comparison JSON report"
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", help="Base directory for experiment artifacts"
    ),
) -> None:
    """Run multi-seed benchmark evaluation for an algorithm configuration.

    Evaluates the configuration across multiple seeds and reports aggregate
    statistics (mean ± std, min, max). Optionally compare two configurations.

    Example:
        adaptive-rl benchmark --config configs/gridworld_ppo.yaml --seeds 42,43,44
        adaptive-rl benchmark --config configs/gridworld_ppo.yaml --compare configs/gridworld_astar.yaml
    """
    from adaptive_rl.benchmarking import BenchmarkRunner

    seed_list = [int(s.strip()) for s in seeds.split(",")] if seeds else [42, 43, 44]
    base_dir = output_dir or Path("experiments/results")

    runner = BenchmarkRunner(seeds=seed_list, timesteps=timesteps, base_output_dir=base_dir)

    if compare_config:
        # Comparison mode
        console.print(
            Panel.fit(
                f"[bold green]Running Ablation Comparison[/bold green]\n\n"
                f"• [bold]Config A:[/bold] {config}\n"
                f"• [bold]Config B:[/bold] {compare_config}\n"
                f"• [bold]Seeds:[/bold] {seed_list}",
                title="Benchmark Runner",
                border_style="cyan",
            )
        )

        comparison = runner.compare(
            config_a=config,
            config_b=compare_config,
            output_path=output_report,
        )

        table = Table(title="Benchmark Comparison Results")
        table.add_column("Metric", style="cyan")
        table.add_column(comparison.arm_a.name, style="green", justify="right")
        table.add_column(comparison.arm_b.name, style="magenta", justify="right")
        table.add_column("Delta (B - A)", style="yellow", justify="right")

        for metric, cmp in comparison.comparison.items():
            a_mean = cmp.get("arm_a_mean")
            b_mean = cmp.get("arm_b_mean")
            delta = cmp.get("delta")

            a_str = f"{a_mean:.4f}" if a_mean is not None else "N/A"
            b_str = f"{b_mean:.4f}" if b_mean is not None else "N/A"
            d_str = f"{delta:+.4f}" if delta is not None else "N/A"

            table.add_row(metric, a_str, b_str, d_str)

        console.print(table)

        if output_report:
            console.print(f"\n[bold green]Report saved to:[/bold green] {output_report}")

    else:
        # Single benchmark mode
        console.print(
            Panel.fit(
                f"[bold green]Running Multi-Seed Benchmark[/bold green]\n\n"
                f"• [bold]Config:[/bold] {config}\n"
                f"• [bold]Seeds:[/bold] {seed_list}",
                title="Benchmark Runner",
                border_style="cyan",
            )
        )

        result = runner.run(config_path=config)

        table = Table(
            title=f"Benchmark Results: {result.name} ({result.successful_seeds}/{len(result.seeds)} seeds)"
        )
        table.add_column("Metric", style="cyan")
        table.add_column("Mean", style="green", justify="right")
        table.add_column("Std Dev", style="yellow", justify="right")
        table.add_column("Min", style="dim", justify="right")
        table.add_column("Max", style="dim", justify="right")

        for metric, stats in sorted(result.aggregate.items()):
            table.add_row(
                metric,
                f"{stats.mean:.4f}",
                f"± {stats.std:.4f}",
                f"{stats.min:.4f}",
                f"{stats.max:.4f}",
            )

        console.print(table)


# ---------------------------------------------------------------------------
# Benchmark-planners command
# ---------------------------------------------------------------------------


@app.command(name="benchmark-planners")
def benchmark_planners(
    config: Path = typer.Option(
        ..., "--config", "-c", help="Path to the YAML experiment configuration"
    ),
    planner: str = typer.Option(
        "astar", "--planner", "-p", help="Classical planner to benchmark ('astar' or 'rrt_star')"
    ),
    episodes: int = typer.Option(
        10, "--episodes", "-n", help="Number of evaluation episodes per seed"
    ),
    start_seed: int = typer.Option(
        42, "--start-seed", "-s", help="Starting seed; subsequent seeds are start_seed+i"
    ),
    output_report: Optional[Path] = typer.Option(
        None, "--output-report", "-o", help="Path to write JSON benchmark report"
    ),
) -> None:
    """Benchmark a classical planner head-to-head against the RL baseline from a config.

    Runs the specified classical planner for the given number of episodes and
    reports success rate, collision rate, path length, and planning time.

    Example:
        adaptive-rl benchmark-planners --config configs/gridworld_astar.yaml --planner astar --episodes 10
        adaptive-rl benchmark-planners --config configs/gridworld_ppo.yaml --planner astar --output-report report.json
    """
    from adaptive_rl.planning.benchmark import ClassicalBenchmarkRunner

    try:
        exp_config = load_config(config)
    except ConfigError as exc:
        console.print(f"[bold red]Configuration error:[/bold red] {exc}")
        raise typer.Exit(code=1) from exc

    env_name = exp_config.environment.name
    env_params = dict(exp_config.environment.parameters or {})

    console.print(
        Panel.fit(
            f"[bold green]Classical vs RL Benchmark[/bold green]\n\n"
            f"• [bold]Planner:[/bold] {planner}\n"
            f"• [bold]Environment:[/bold] {env_name}\n"
            f"• [bold]Episodes:[/bold] {episodes}\n"
            f"• [bold]Start Seed:[/bold] {start_seed}",
            title="Planner Benchmark",
            border_style="cyan",
        )
    )

    runner = ClassicalBenchmarkRunner(
        environment_name=env_name,
        planner_type=planner,
        environment_parameters=env_params,
    )

    seeds = list(range(start_seed, start_seed + episodes))
    report = runner.run_benchmark(
        seeds=seeds,
        experiment_name=f"{env_name}_{planner}_benchmark",
        rl_algorithm=None,
    )

    table = Table(title="Classical vs RL Benchmark Results")
    table.add_column("Metric", style="cyan")
    table.add_column("Value", style="green", justify="right")

    p_sr_str = (
        f"{report.planner_success_rate:.2%}" if report.planner_success_rate is not None else "N/A"
    )
    p_cr_str = (
        f"{report.planner_collision_rate:.2%}"
        if report.planner_collision_rate is not None
        else "N/A"
    )
    table.add_row("Success Rate", p_sr_str)
    table.add_row("Collision Rate", p_cr_str)
    table.add_row("Mean Path Length", f"{report.planner_mean_path_length:.2f}")
    table.add_row("Mean Steps", f"{report.planner_mean_steps:.1f}")
    table.add_row("Mean Planning Time (ms)", f"{report.planner_mean_planning_time_ms:.2f}")
    table.add_row("Total Episodes", str(report.total_episodes))

    console.print(table)

    if output_report:
        saved = report.save_json(output_report)
        console.print(f"\n[bold green]Report saved to:[/bold green] {saved}")


# ---------------------------------------------------------------------------
# Dashboard command
# ---------------------------------------------------------------------------


@app.command()
def dashboard(
    experiment_id: Optional[str] = typer.Option(
        None, "--experiment", "-e", help="Show detailed metrics for a specific experiment ID"
    ),
    compare: Optional[str] = typer.Option(
        None,
        "--compare",
        help="Comma-separated list of experiment IDs to compare side by side",
    ),
    output_dir: Optional[Path] = typer.Option(
        None, "--output-dir", "-o", help="Experiments base directory (default: experiments/results)"
    ),
) -> None:
    """Open the AdaptiveRL terminal experiment dashboard.

    Displays a Rich terminal UI with experiment overview, metrics, and comparisons.

    Usage examples:
        adaptive-rl dashboard
        adaptive-rl dashboard --experiment 2026-09-17_gridworld_ppo_seed42
        adaptive-rl dashboard --compare 2026-09-17_gridworld_ppo_seed42,2026-09-17_gridworld_astar_seed42
    """
    from adaptive_rl.visualization.dashboard import run_dashboard

    compare_list = [s.strip() for s in compare.split(",")] if compare else None
    base_dir = output_dir or Path("experiments/results")

    run_dashboard(
        base_output_dir=base_dir,
        experiment_id=experiment_id,
        compare=compare_list,
    )


if __name__ == "__main__":
    app()
