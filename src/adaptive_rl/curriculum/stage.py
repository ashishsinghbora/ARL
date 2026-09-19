"""Curriculum stage representation and advancement evaluation."""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Any, Dict, Optional


@dataclass
class CurriculumStage:
    """Represents a discrete milestone in an environment curriculum."""

    stage_id: int
    name: str
    environment_parameters: Dict[str, Any] = field(default_factory=dict)
    success_threshold: Optional[float] = None
    mean_reward_threshold: Optional[float] = None
    max_timesteps: Optional[int] = None
    min_episodes: int = 10
    description: str = ""

    def can_advance(
        self,
        rolling_metrics: Dict[str, Any],
        stage_episodes: int,
        stage_timesteps: int,
    ) -> bool:
        """Evaluate whether criteria to advance past this stage have been satisfied.

        Advancement logic:
        1. If max_timesteps is specified and stage_timesteps >= max_timesteps, advance automatically.
        2. If stage_episodes < min_episodes, do not advance.
        3. If success_threshold is specified, rolling success_rate must reach or exceed it.
        4. If mean_reward_threshold is specified, rolling mean_reward must reach or exceed it.
        5. If at least one threshold was specified and passed, return True.

        Args:
            rolling_metrics: Dict with rolling performance metrics ('success_rate', 'mean_reward').
            stage_episodes: Episodes completed within the current stage.
            stage_timesteps: Environment steps executed within the current stage.

        Returns:
            bool: True if agent is eligible to advance to the next stage.
        """
        # Timeout override
        if self.max_timesteps is not None and stage_timesteps >= self.max_timesteps:
            return True

        # Enforce minimum training sample in stage
        if stage_episodes < self.min_episodes:
            return False

        has_threshold = False

        if self.success_threshold is not None:
            has_threshold = True
            sr_val = rolling_metrics.get("success_rate")
            current_sr = float(sr_val) if sr_val is not None else 0.0
            if current_sr < self.success_threshold:
                return False

        if self.mean_reward_threshold is not None:
            has_threshold = True
            current_mr = float(rolling_metrics.get("mean_reward", float("-inf")))
            if current_mr < self.mean_reward_threshold:
                return False

        return has_threshold
