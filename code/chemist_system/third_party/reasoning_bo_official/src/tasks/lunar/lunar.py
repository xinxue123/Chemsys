# references
# AABO, https://github.com/nataliemaus/aabo/blob/main/tasks/utils/lunar.py


#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-17 13:59:52
@File: src/tasks/aerospace/lunar.py
@IDE: vscode
@Description:
    Metric for evaluating Lunar Lander optimization performance.
"""

from __future__ import annotations
import numpy as np
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.types import TParameterization
from ax.utils.common.result import Err, Ok
from pyre_extensions import none_throws
import multiprocess as mp
from collections.abc import Iterable
from src.tasks.lunar.lunar_utils import simulate_lunar_lander
import pandas as pd


class LunarLanderMetric(Metric):
    """Metric for evaluating Lunar Lander optimization performance.

    Args:
        name: The name of the metric.
        seed: Random seeds for simulation (default: range(50))
        lower_is_better: If True, the metric should be minimized (default: False)
    """

    def __init__(
        self,
        name: str,
        seed: Iterable[int] = range(50),
        lower_is_better: bool = False,
        **kwargs,
    ):
        self.seed = [seed] if not isinstance(seed, Iterable) else seed
        self.pool = mp.Pool(mp.cpu_count())
        super().__init__(name=name, lower_is_better=lower_is_better, **kwargs)

    def _evaluate_lander(self, parameters: TParameterization) -> float:
        """Evaluate lander performance for given parameters."""
        # 硬编码
        # x = np.array([parameters[f"x{i+1}"] for i in range(12)])
        # 软编码
        # print(parameters)
        x = np.array(
            [
                parameters[param_name]
                for param_name in sorted(parameters.keys())
            ]
        )
        params = [[x, seed] for seed in self.seed]
        rewards = np.array(self.pool.map(simulate_lunar_lander, params))
        return np.mean(rewards)

    def fetch_trial_data(self, trial: BaseTrial) -> MetricFetchResult:
        try:
            arm_names = []
            mean = []
            sem = []
            trial_indices = []

            for name, arm in trial.arms_by_name.items():
                try:
                    val = self._evaluate_lander(arm.parameters)
                    arm_names.append(name)
                    mean.append(val)
                    sem.append(0.0)  # Noiseless evaluation
                    trial_indices.append(trial.index)
                except (KeyError, ValueError):
                    continue

            if not arm_names:
                return Err(
                    MetricFetchE(
                        message=f"No valid arms found for {self.name}",
                        exception=ValueError(
                            "All arms have invalid parameters"
                        ),
                    )
                )

            df = pd.DataFrame(
                {
                    "arm_name": arm_names,
                    "metric_name": self.name,
                    "mean": mean,
                    "sem": sem,
                    "trial_index": trial_indices,
                }
            )
            return Ok(value=Data(df=df))

        except Exception as e:
            return Err(
                MetricFetchE(
                    message=f"Failed to fetch {self.name}", exception=e
                )
            )

    def __del__(self):
        self.pool.close()
