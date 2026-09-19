"""AdaptiveRL Studio, an optional desktop control center for experiments."""

from __future__ import annotations

import json
import sys
from pathlib import Path
from threading import Event
from typing import Any, Dict, List, Optional, cast

from PySide6.QtCore import QObject, QPointF, Qt, QThread, Signal, Slot
from PySide6.QtGui import QColor, QPainter, QPen, QPolygonF
from PySide6.QtWidgets import (
    QApplication,
    QComboBox,
    QFrame,
    QGridLayout,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QListWidget,
    QListWidgetItem,
    QMainWindow,
    QMessageBox,
    QProgressBar,
    QPushButton,
    QSpinBox,
    QStackedWidget,
    QTableWidget,
    QTableWidgetItem,
    QVBoxLayout,
    QWidget,
)

from adaptive_rl.training.callbacks import BaseCallback

try:
    from adaptive_rl.experiments.manager import ExperimentManager
except ImportError:

    class ExperimentManager:  # type: ignore
        """Fallback manager for artifact reading when running without experiment manager package."""

        def __init__(self, base_output_dir: Any = None) -> None:
            self.base_output_dir = Path(base_output_dir or "experiments/results")

        def list_experiments(self) -> List[Dict[str, Any]]:
            experiments: List[Dict[str, Any]] = []
            if not self.base_output_dir.exists():
                return experiments
            for exp_dir in sorted(self.base_output_dir.iterdir()):
                if not exp_dir.is_dir():
                    continue
                manifest_path = exp_dir / "manifest.json"
                if manifest_path.exists():
                    try:
                        with open(manifest_path, encoding="utf-8") as f:
                            experiments.append(json.load(f))
                    except Exception:
                        experiments.append({"experiment_id": exp_dir.name})
            return sorted(experiments, key=lambda x: str(x.get("created_at", "")), reverse=True)

        def get_metrics(self, experiment_id: str) -> Optional[Dict[str, Any]]:
            metrics_path = self.base_output_dir / experiment_id / "metrics.json"
            if not metrics_path.exists():
                return None
            try:
                with open(metrics_path, encoding="utf-8") as f:
                    return cast(Dict[str, Any], json.load(f))
            except Exception:
                return None


NAV_ITEMS = [
    ("Overview", "What is my RL system doing right now?"),
    ("Environment Lab", "Inspect layouts, trajectories, and episode behavior."),
    ("Training Center", "Launch training through the existing trainer API."),
    ("Evaluation Lab", "Review measured performance, not training proxies."),
    ("Generalization", "Compare train and unseen distributions."),
    ("Experiments", "Inspect reproducible artifacts and provenance."),
    ("Benchmarks", "Compare policies and classical planners."),
]


class MetricCard(QFrame):
    """Compact metric display used across the Studio overview."""

    def __init__(self, label: str, value: str, accent: str) -> None:
        super().__init__()
        self.setObjectName("metricCard")
        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 14, 16, 14)
        eyebrow = QLabel(label.upper())
        eyebrow.setObjectName("metricLabel")
        metric = QLabel(value)
        metric.setObjectName("metricValue")
        metric.setStyleSheet(f"color: {accent};")
        layout.addWidget(eyebrow)
        layout.addWidget(metric)


class TrajectoryCanvas(QWidget):
    """Simple top-down trajectory view for navigation and drone environments."""

    def __init__(self) -> None:
        super().__init__()
        self.setMinimumSize(560, 330)
        self.start = QPointF(0.15, 0.78)
        self.goal = QPointF(0.84, 0.2)
        self.trajectory: list[QPointF] = []
        self.obstacles: list[QPointF] = []

    def set_episode(
        self, trajectory: list[tuple[float, float]], obstacles: list[tuple[float, float]]
    ) -> None:
        self.trajectory = [QPointF(x, y) for x, y in trajectory]
        self.obstacles = [QPointF(x, y) for x, y in obstacles]
        self.update()

    def paintEvent(self, _event: Any) -> None:
        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)
        painter.fillRect(self.rect(), QColor("#101820"))
        margin = 28
        width = self.width() - 2 * margin
        height = self.height() - 2 * margin

        painter.setPen(QPen(QColor("#263847"), 1))
        for index in range(1, 10):
            x = margin + width * index / 10
            y = margin + height * index / 10
            painter.drawLine(QPointF(x, margin), QPointF(x, margin + height))
            painter.drawLine(QPointF(margin, y), QPointF(margin + width, y))

        def project(point: QPointF) -> QPointF:
            return QPointF(margin + point.x() * width, margin + point.y() * height)

        painter.setBrush(QColor("#d15b48"))
        painter.setPen(Qt.PenStyle.NoPen)
        for obstacle in self.obstacles:
            center = project(obstacle)
            painter.drawEllipse(center, 9, 9)

        painter.setBrush(QColor("#e7b65a"))
        painter.drawEllipse(project(self.start), 7, 7)
        painter.setBrush(QColor("#62c7a5"))
        painter.drawPolygon(
            QPolygonF(
                [
                    project(self.goal) + QPointF(0, -10),
                    project(self.goal) + QPointF(9, 8),
                    project(self.goal) + QPointF(-9, 8),
                ]
            )
        )

        if len(self.trajectory) > 1:
            painter.setPen(QPen(QColor("#79b8ff"), 3))
            points = [project(point) for point in self.trajectory]
            for first, second in zip(points, points[1:]):
                painter.drawLine(first, second)
            painter.setBrush(QColor("#f4f7fb"))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawEllipse(points[-1], 6, 6)


class EpisodeWorker(QObject):
    """Run a short environment episode away from the GUI thread."""

    finished = Signal(object, object)
    failed = Signal(str)

    def __init__(self, environment: str, seed: int) -> None:
        super().__init__()
        self.environment = environment
        self.seed = seed

    @Slot()
    def run(self) -> None:
        try:
            from adaptive_rl.environments import make_env

            env = make_env(self.environment)
            observation, _ = env.reset(seed=self.seed)
            trajectory: list[tuple[float, float]] = []
            obstacles: list[tuple[float, float]] = []
            if hasattr(env, "obstacles"):
                obstacles = [(float(item[0]), float(item[1])) for item in env.obstacles]
            for _ in range(80):
                position = getattr(env, "agent_pos", getattr(env, "position", None))
                if position is not None:
                    trajectory.append((float(position[0]), float(position[1])))
                observation, _, terminated, truncated, _ = env.step(env.action_space.sample())
                if terminated or truncated:
                    break
            env.close()
            self.finished.emit(trajectory, obstacles)
        except Exception as err:
            self.failed.emit(str(err))


class TrainingWorker(QObject):
    """Run the existing AdaptiveRL trainer without blocking the window."""

    finished = Signal(object)
    failed = Signal(str)

    def __init__(self, config_path: str, timesteps: int, seed: int) -> None:
        super().__init__()
        self.config_path = config_path
        self.timesteps = timesteps
        self.seed = seed
        self.stop_event = Event()

    def request_stop(self) -> None:
        """Ask the cooperative training callback to stop after the current step."""
        self.stop_event.set()

    @Slot()
    def run(self) -> None:
        trainer = None
        try:
            from adaptive_rl.config import load_config
            from adaptive_rl.training import get_trainer

            config = load_config(Path(self.config_path))
            if config.training is None:
                raise ValueError("The selected configuration has no training section.")
            config.training.total_timesteps = self.timesteps
            config.seed = self.seed
            trainer = get_trainer(config=config, callbacks=[TrainingStopCallback(self.stop_event)])
            self.finished.emit(trainer.fit())
        except Exception as err:
            self.failed.emit(str(err))
        finally:
            if trainer is not None:
                trainer.close()


class TrainingStopCallback(BaseCallback):
    """Stop training cooperatively without terminating a Qt worker thread."""

    def __init__(self, stop_event: Event) -> None:
        self.stop_event = stop_event

    def on_training_start(self, locals_dict: Optional[dict[str, Any]] = None) -> None:
        return None

    def on_step(self, step: int, locals_dict: Optional[dict[str, Any]] = None) -> bool:
        return not self.stop_event.is_set()

    def on_episode_end(
        self,
        episode: int,
        episode_reward: float,
        episode_length: int,
        info: Optional[dict[str, Any]] = None,
        metrics: Optional[Any] = None,
    ) -> None:
        return None

    def on_training_end(self) -> None:
        return None


class StudioWindow(QMainWindow):
    """Main AdaptiveRL Studio window."""

    def __init__(self, output_dir: Path) -> None:
        super().__init__()
        self.output_dir = output_dir
        self.manager = ExperimentManager(base_output_dir=output_dir)
        self.episode_thread: Optional[QThread] = None
        self.training_thread: Optional[QThread] = None
        self.training_worker: Optional[TrainingWorker] = None
        self.environment: Any = None
        self.environment_observation: Any = None
        self.environment_trajectory: list[tuple[float, float]] = []
        self.setWindowTitle("AdaptiveRL Studio")
        self.resize(1420, 860)
        self._apply_theme()
        self._build_ui()
        self._refresh_experiments()

    def _apply_theme(self) -> None:
        application = QApplication.instance()
        if not isinstance(application, QApplication):
            return
        application.setStyleSheet(
            """
            QWidget#root { background: #0d141b; color: #e7edf4; }
            QFrame#sidebar { background: #111d26; border-right: 1px solid #243746; }
            QFrame#content { background: #0d141b; }
            QLabel#brand { color: #f4f7fb; font-size: 22px; font-weight: 800; letter-spacing: 2px; }
            QLabel#subtitle, QLabel#muted, QLabel#sidebarStatus { color: #8295a5; }
            QLabel#pageTitle { color: #f4f7fb; font-size: 28px; font-weight: 700; }
            QLabel#intro { color: #9db0bf; font-size: 14px; }
            QLabel#readyPill { color: #62c7a5; background: #17342f; padding: 7px 12px; border-radius: 5px; font-weight: 700; }
            QListWidget#navigation { background: transparent; border: none; color: #9db0bf; font-size: 14px; }
            QListWidget#navigation::item { padding: 12px 10px; border-radius: 5px; }
            QListWidget#navigation::item:selected { background: #1d3445; color: #f4f7fb; }
            QFrame#metricCard, QFrame#panel { background: #14232e; border: 1px solid #243746; border-radius: 7px; }
            QLabel#metricLabel { color: #8295a5; font-size: 10px; font-weight: 700; }
            QLabel#metricValue { font-size: 26px; font-weight: 700; }
            QLineEdit#pathField, QComboBox, QSpinBox { background: #101c25; border: 1px solid #385064; border-radius: 4px; padding: 8px; color: #e7edf4; }
            QPushButton { background: #1d3445; color: #e7edf4; border: 1px solid #385064; border-radius: 4px; padding: 9px 14px; }
            QPushButton:hover { background: #27465b; }
            QPushButton#primaryButton { background: #2f7d72; border-color: #62c7a5; font-weight: 700; }
            QTableWidget { background: #101c25; alternate-background-color: #14232e; border: 1px solid #243746; gridline-color: #243746; color: #d9e2ea; }
            QHeaderView::section { background: #1a2c39; color: #9db0bf; padding: 8px; border: none; }
            QLabel#emptyState { color: #9db0bf; font-size: 16px; padding: 30px; }
            """
        )

    def _build_ui(self) -> None:
        root = QWidget()
        root.setObjectName("root")
        self.setCentralWidget(root)
        shell = QHBoxLayout(root)
        shell.setContentsMargins(0, 0, 0, 0)
        shell.setSpacing(0)

        sidebar = QFrame()
        sidebar.setObjectName("sidebar")
        sidebar.setFixedWidth(252)
        sidebar_layout = QVBoxLayout(sidebar)
        sidebar_layout.setContentsMargins(22, 24, 18, 22)
        brand = QLabel("ADAPTIVERL\nSTUDIO")
        brand.setObjectName("brand")
        sidebar_layout.addWidget(brand)
        subtitle = QLabel("Research control center")
        subtitle.setObjectName("subtitle")
        sidebar_layout.addWidget(subtitle)
        sidebar_layout.addSpacing(28)
        self.navigation = QListWidget()
        self.navigation.setObjectName("navigation")
        for name, hint in NAV_ITEMS:
            item = QListWidgetItem(name)
            item.setToolTip(hint)
            self.navigation.addItem(item)
        self.navigation.currentRowChanged.connect(self._change_page)
        sidebar_layout.addWidget(self.navigation)
        sidebar_layout.addStretch()
        status = QLabel("●  LOCAL\n\nArtifacts are read from\n" + str(self.output_dir))
        status.setObjectName("sidebarStatus")
        sidebar_layout.addWidget(status)
        shell.addWidget(sidebar)

        content = QFrame()
        content.setObjectName("content")
        content_layout = QVBoxLayout(content)
        content_layout.setContentsMargins(38, 30, 42, 32)
        content_layout.setSpacing(20)
        header = QHBoxLayout()
        self.page_title = QLabel("Training Overview")
        self.page_title.setObjectName("pageTitle")
        header.addWidget(self.page_title)
        header.addStretch()
        self.header_status = QLabel("READY")
        self.header_status.setObjectName("readyPill")
        header.addWidget(self.header_status)
        content_layout.addLayout(header)
        self.pages = QStackedWidget()
        self.pages.addWidget(self._overview_page())
        self.pages.addWidget(self._environment_page())
        self.pages.addWidget(self._training_page())
        self.pages.addWidget(self._evaluation_page())
        self.pages.addWidget(self._generalization_page())
        self.pages.addWidget(self._experiments_page())
        self.pages.addWidget(self._benchmark_page())
        content_layout.addWidget(self.pages)
        shell.addWidget(content, 1)
        self.navigation.setCurrentRow(0)

    def _overview_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        layout.setContentsMargins(0, 0, 0, 0)
        intro = QLabel("A calm view of what your RL system is doing.")
        intro.setObjectName("intro")
        layout.addWidget(intro)
        metrics = QGridLayout()
        metrics.setSpacing(12)
        cards = [
            ("Mean Reward", "—", "#79b8ff"),
            ("Success Rate", "—", "#62c7a5"),
            ("Experiments", "0", "#e7b65a"),
            ("Latest Seed", "—", "#d987b3"),
        ]
        self.metric_values: list[QLabel] = []
        for index, (label, value, accent) in enumerate(cards):
            card = MetricCard(label, value, accent)
            self.metric_values.append(card.findChild(QLabel, "metricValue"))  # type: ignore[arg-type]
            metrics.addWidget(card, 0, index)
        layout.addLayout(metrics)
        chart_frame = QFrame()
        chart_frame.setObjectName("panel")
        chart_layout = QVBoxLayout(chart_frame)
        chart_layout.addWidget(QLabel("RECENT EXPERIMENTS"))
        self.recent_table = QTableWidget(0, 4)
        self.recent_table.setHorizontalHeaderLabels(["Run", "Environment", "Algorithm", "Status"])
        self.recent_table.horizontalHeader().setStretchLastSection(True)
        self.recent_table.verticalHeader().setVisible(False)
        self.recent_table.setEditTriggers(QTableWidget.EditTrigger.NoEditTriggers)
        chart_layout.addWidget(self.recent_table)
        layout.addWidget(chart_frame)
        layout.addStretch()
        return page

    def _environment_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.environment_select = QComboBox()
        self.environment_select.addItems(["gridworld", "navigation", "traffic", "drone"])
        self.seed_select = QSpinBox()
        self.seed_select.setRange(0, 1_000_000)
        self.seed_select.setValue(42)
        run_button = QPushButton("Run Episode")
        run_button.setObjectName("primaryButton")
        run_button.clicked.connect(self._run_episode)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._reset_environment)
        step_button = QPushButton("Step")
        step_button.clicked.connect(self._step_environment)
        controls.addWidget(QLabel("Environment"))
        controls.addWidget(self.environment_select)
        controls.addWidget(QLabel("Seed"))
        controls.addWidget(self.seed_select)
        controls.addWidget(run_button)
        controls.addWidget(reset_button)
        controls.addWidget(step_button)
        controls.addStretch()
        layout.addLayout(controls)
        self.trajectory_canvas = TrajectoryCanvas()
        layout.addWidget(self.trajectory_canvas, 1)
        self.environment_status = QLabel("Choose an environment and run an episode.")
        self.environment_status.setObjectName("muted")
        layout.addWidget(self.environment_status)
        self.environment_observation_label = QLabel(
            "Observation: —\nReward: —\nTerminated: —  Truncated: —"
        )
        self.environment_observation_label.setObjectName("muted")
        layout.addWidget(self.environment_observation_label)
        return page

    def _evaluation_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        controls = QHBoxLayout()
        self.evaluation_select = QComboBox()
        evaluate_button = QPushButton("Load Metrics")
        evaluate_button.setObjectName("primaryButton")
        evaluate_button.clicked.connect(self._load_selected_metrics)
        controls.addWidget(QLabel("Experiment"))
        controls.addWidget(self.evaluation_select, 1)
        controls.addWidget(evaluate_button)
        layout.addLayout(controls)
        self.evaluation_table = QTableWidget(0, 2)
        self.evaluation_table.setHorizontalHeaderLabels(["Metric", "Measured value"])
        self.evaluation_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.evaluation_table)
        return page

    def _generalization_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        self.generalization_label = QLabel("No generalization reports found.")
        self.generalization_label.setObjectName("emptyState")
        layout.addWidget(self.generalization_label)
        layout.addStretch()
        return page

    def _training_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        panel = QFrame()
        panel.setObjectName("panel")
        form = QGridLayout(panel)
        self.training_config = QLineEdit("configs/smoke_gridworld_ppo.yaml")
        self.training_config.setObjectName("pathField")
        self.training_steps = QSpinBox()
        self.training_steps.setRange(1, 10_000_000)
        self.training_steps.setValue(256)
        self.training_seed = QSpinBox()
        self.training_seed.setRange(0, 1_000_000)
        self.training_seed.setValue(42)
        self.training_progress = QProgressBar()
        self.training_progress.setValue(0)
        train_button = QPushButton("Start Training")
        train_button.setObjectName("primaryButton")
        train_button.clicked.connect(self._start_training)
        stop_button = QPushButton("Stop")
        stop_button.clicked.connect(self._stop_training)
        reset_button = QPushButton("Reset")
        reset_button.clicked.connect(self._reset_training)
        form.addWidget(QLabel("Config"), 0, 0)
        form.addWidget(self.training_config, 0, 1, 1, 2)
        form.addWidget(QLabel("Timesteps"), 1, 0)
        form.addWidget(self.training_steps, 1, 1)
        form.addWidget(QLabel("Seed"), 2, 0)
        form.addWidget(self.training_seed, 2, 1)
        form.addWidget(train_button, 3, 0, 1, 2)
        form.addWidget(stop_button, 3, 2)
        form.addWidget(reset_button, 4, 2)
        form.addWidget(self.training_progress, 5, 0, 1, 3)
        layout.addWidget(panel)
        self.training_status = QLabel(
            "Training runs through adaptive_rl.training; Studio does not duplicate the RL engine."
        )
        self.training_status.setObjectName("muted")
        layout.addWidget(self.training_status)
        layout.addStretch()
        return page

    def _experiments_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        refresh = QPushButton("Refresh Experiments")
        refresh.clicked.connect(self._refresh_experiments)
        layout.addWidget(refresh, alignment=Qt.AlignmentFlag.AlignLeft)
        self.experiment_table = QTableWidget(0, 5)
        self.experiment_table.setHorizontalHeaderLabels(
            ["Experiment", "Environment", "Algorithm", "Seed", "Status"]
        )
        self.experiment_table.horizontalHeader().setStretchLastSection(True)
        self.experiment_table.verticalHeader().setVisible(False)
        layout.addWidget(self.experiment_table)
        return page

    def _benchmark_page(self) -> QWidget:
        page = QWidget()
        layout = QVBoxLayout(page)
        message = QLabel(
            "BENCHMARK CENTER\n\nSelect experiments from the table below to compare measured metrics."
        )
        message.setObjectName("emptyState")
        layout.addWidget(message)
        self.benchmark_table = QTableWidget(0, 4)
        self.benchmark_table.setHorizontalHeaderLabels(
            ["Experiment", "Environment", "Algorithm", "Mean reward"]
        )
        self.benchmark_table.horizontalHeader().setStretchLastSection(True)
        layout.addWidget(self.benchmark_table)
        layout.addStretch()
        return page

    @Slot(int)
    def _change_page(self, index: int) -> None:
        self.pages.setCurrentIndex(index)
        self.page_title.setText(NAV_ITEMS[index][0])

    @Slot()
    def _run_episode(self) -> None:
        if self.episode_thread is not None and self.episode_thread.isRunning():
            return
        self.environment_status.setText("Running episode in background...")
        self.header_status.setText("RUNNING")
        self.episode_thread = QThread(self)
        worker = EpisodeWorker(self.environment_select.currentText(), self.seed_select.value())
        worker.moveToThread(self.episode_thread)
        self.episode_thread.started.connect(worker.run)
        worker.finished.connect(self._episode_finished)
        worker.failed.connect(self._episode_failed)
        worker.finished.connect(self.episode_thread.quit)
        worker.failed.connect(self.episode_thread.quit)
        self.episode_thread.finished.connect(worker.deleteLater)
        self.episode_thread.finished.connect(self.episode_thread.deleteLater)
        self.episode_thread.start()

    @Slot()
    def _reset_environment(self) -> None:
        try:
            from adaptive_rl.environments import make_env

            if self.environment is not None:
                self.environment.close()
            self.environment = make_env(self.environment_select.currentText())
            self.environment_observation, _ = self.environment.reset(seed=self.seed_select.value())
            self.environment_trajectory = []
            self._update_environment_observation(0.0, False, False)
            self.environment_status.setText("Environment reset.")
        except Exception as err:
            self._show_error("Environment reset", err)

    @Slot()
    def _step_environment(self) -> None:
        try:
            if self.environment is None:
                self._reset_environment()
            if self.environment is None:
                return
            self.environment_observation, reward, terminated, truncated, _ = self.environment.step(
                self.environment.action_space.sample()
            )
            self._update_environment_observation(reward, terminated, truncated)
            if terminated or truncated:
                self.environment_status.setText("Episode ended. Reset to continue.")
        except Exception as err:
            self._show_error("Environment step", err)

    def _update_environment_observation(
        self, reward: float, terminated: bool, truncated: bool
    ) -> None:
        self.environment_observation_label.setText(
            f"Observation: {self.environment_observation}\n"
            f"Reward: {reward:.3f}\nTerminated: {terminated}  Truncated: {truncated}"
        )

    def _show_error(self, operation: str, error: Exception) -> None:
        message = f"{type(error).__name__}: {error}"
        self.header_status.setText("ERROR")
        QMessageBox.critical(self, f"{operation} failed", message)

    @Slot(object, object)
    def _episode_finished(
        self, trajectory: list[tuple[float, float]], obstacles: list[tuple[float, float]]
    ) -> None:
        self.trajectory_canvas.set_episode(trajectory, obstacles)
        self.environment_status.setText(f"Episode complete: {len(trajectory)} trajectory samples.")
        self.header_status.setText("READY")

    @Slot(str)
    def _episode_failed(self, message: str) -> None:
        self.environment_status.setText(f"Episode failed: {message}")
        self.header_status.setText("ERROR")
        QMessageBox.critical(self, "Episode failed", message)

    @Slot()
    def _start_training(self) -> None:
        if self.training_thread is not None and self.training_thread.isRunning():
            return
        self.training_progress.setValue(0)
        self.training_status.setText("Training in progress through adaptive_rl.training...")
        self.header_status.setText("TRAINING")
        self.training_thread = QThread(self)
        worker = TrainingWorker(
            self.training_config.text(),
            self.training_steps.value(),
            self.training_seed.value(),
        )
        self.training_worker = worker
        worker.moveToThread(self.training_thread)
        self.training_thread.started.connect(worker.run)
        worker.finished.connect(self._training_finished)
        worker.failed.connect(self._training_failed)
        worker.finished.connect(self.training_thread.quit)
        worker.failed.connect(self.training_thread.quit)
        self.training_thread.finished.connect(worker.deleteLater)
        self.training_thread.finished.connect(self.training_thread.deleteLater)
        self.training_thread.start()

    @Slot()
    def _stop_training(self) -> None:
        if self.training_worker is None:
            self.training_status.setText("No training run is active.")
            return
        self.training_worker.request_stop()
        self.training_status.setText("Stop requested; finishing the current environment step...")

    @Slot()
    def _reset_training(self) -> None:
        if self.training_thread is not None and self.training_thread.isRunning():
            self._stop_training()
            return
        self.training_config.setText("configs/smoke_gridworld_ppo.yaml")
        self.training_steps.setValue(256)
        self.training_seed.setValue(42)
        self.training_progress.setValue(0)
        self.training_status.setText("Training form reset.")
        self.header_status.setText("READY")

    @Slot(object)
    def _training_finished(self, result: Any) -> None:
        self.training_progress.setValue(100)
        self.training_status.setText(
            f"Training complete: {result.total_timesteps:,} timesteps. "
            f"Model: {result.final_model_path}"
        )
        self.header_status.setText("READY")
        self._refresh_experiments()

    @Slot(str)
    def _training_failed(self, message: str) -> None:
        self.training_status.setText(f"Training failed: {message}")
        self.header_status.setText("ERROR")
        QMessageBox.critical(self, "Training failed", message)

    def _refresh_experiments(self) -> None:
        experiments = self.manager.list_experiments()
        self.metric_values[2].setText(str(len(experiments)))
        if experiments:
            self.metric_values[3].setText(str(experiments[0].get("seed", "—")))
        if hasattr(self, "evaluation_select"):
            self.evaluation_select.clear()
            self.evaluation_select.addItems(
                [str(item.get("experiment_id", "?")) for item in experiments]
            )
        if hasattr(self, "benchmark_table"):
            self.benchmark_table.setRowCount(0)
        for table in (self.recent_table, self.experiment_table):
            table.setRowCount(0)
        for experiment in experiments:
            row = self.recent_table.rowCount()
            self.recent_table.insertRow(row)
            values = [
                str(experiment.get("experiment_id", "?")),
                str(experiment.get("environment", "?")),
                str(experiment.get("algorithm", "?")).upper(),
                str(experiment.get("evaluation_status", "unknown")),
            ]
            for column, value in enumerate(values):
                self.recent_table.setItem(row, column, QTableWidgetItem(value))
            row = self.experiment_table.rowCount()
            self.experiment_table.insertRow(row)
            detail_values = values[:3] + [str(experiment.get("seed", "?")), values[3]]
            for column, value in enumerate(detail_values):
                self.experiment_table.setItem(row, column, QTableWidgetItem(value))

            if hasattr(self, "benchmark_table"):
                metrics = self.manager.get_metrics(str(experiment.get("experiment_id", ""))) or {}
                row = self.benchmark_table.rowCount()
                self.benchmark_table.insertRow(row)
                benchmark_values = values[:3] + [str(metrics.get("mean_reward", "—"))]
                for column, value in enumerate(benchmark_values):
                    self.benchmark_table.setItem(row, column, QTableWidgetItem(value))

        self._load_generalization_reports()

    def _load_selected_metrics(self) -> None:
        self.evaluation_table.setRowCount(0)
        if not self.evaluation_select.currentText():
            return
        metrics = self.manager.get_metrics(self.evaluation_select.currentText()) or {}
        for name in (
            "mean_reward",
            "std_reward",
            "success_rate",
            "collision_rate",
            "mean_episode_length",
        ):
            if name in metrics:
                row = self.evaluation_table.rowCount()
                self.evaluation_table.insertRow(row)
                self.evaluation_table.setItem(row, 0, QTableWidgetItem(name))
                self.evaluation_table.setItem(row, 1, QTableWidgetItem(str(metrics[name])))

    def _load_generalization_reports(self) -> None:
        reports = sorted(self.output_dir.rglob("*report.json"))
        if not reports:
            self.generalization_label.setText("No generalization reports found.")
            return
        try:
            data = json.loads(reports[-1].read_text(encoding="utf-8"))
            self.generalization_label.setText(
                f"GENERALIZATION REPORT\n\n"
                f"Experiment: {data.get('experiment_name', reports[-1].stem)}\n"
                f"Train success: {data.get('train_metrics', {}).get('success_rate', '—')}\n"
                f"Unseen success: {data.get('test_metrics', {}).get('success_rate', '—')}\n"
                f"Success gap: {data.get('generalization_gap_success', '—')}\n"
                f"Reward gap: {data.get('generalization_gap_reward', '—')}"
            )
        except (OSError, json.JSONDecodeError) as err:
            self.generalization_label.setText(f"Could not read report: {type(err).__name__}: {err}")

    def closeEvent(self, event: Any) -> None:
        """Stop workers and release the active environment during shutdown."""
        if self.training_worker is not None:
            self.training_worker.request_stop()
        for thread in (self.episode_thread, self.training_thread):
            if thread is not None and thread.isRunning():
                thread.quit()
                thread.wait(5000)
        if self.environment is not None:
            self.environment.close()
            self.environment = None
        event.accept()


def launch_studio(output_dir: Optional[Path] = None) -> int:
    """Create one QApplication, show one strongly-held window, and run Qt."""
    application = QApplication.instance()
    if application is None:
        application = QApplication(sys.argv)
    window = StudioWindow(output_dir or Path("experiments/results"))
    application._adaptive_rl_studio_window = window  # type: ignore[attr-defined]
    window.show()
    window.raise_()
    window.activateWindow()
    return application.exec()
