#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-28 18:48:58
@File: src/tasks/maths/rosenbrock.py
@IDE: vscode
@Description:
    Metric for evaluating Rosenbrock synthetic function.
"""

from __future__ import annotations

import numpy as np
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.types import TParameterization
from ax.utils.common.result import Err, Ok
from pyre_extensions import none_throws
from typing import Any
import pandas as pd


class RosenbrockMetric(Metric):
    """Metric for evaluating Rosenbrock synthetic function.

    Args:
        name: The name of the metric.
        noiseless: If True, consider observations noiseless, otherwise
            assume unknown Gaussian observation noise.
        lower_is_better: If True, the metric should be minimized.
        dimension: Dimension of the Rosenbrock function (default=2).
    """

    def __init__(
        self,
        name: str,
        noiseless: bool = False,
        lower_is_better: bool = True,
        dimension: int = 2,
    ) -> None:
        self.noiseless = noiseless
        self.dimension = dimension
        super().__init__(name=name, lower_is_better=lower_is_better)

    def clone(self) -> RosenbrockMetric:
        return self.__class__(
            name=self._name,
            noiseless=self.noiseless,
            lower_is_better=none_throws(self.lower_is_better),
            dimension=self.dimension,
        )

    def _evaluate_rosenbrock(self, params: TParameterization) -> float:
        """Evaluate Rosenbrock function at given parameters."""
        # Convert parameters to numpy array
        x = np.array(
            [params[param_name] for param_name in sorted(params.keys())]
        )

        # Rosenbrock function calculation
        sum_val = 0.0
        for i in range(self.dimension - 1):
            term1 = 100 * (x[i + 1] - x[i] ** 2) ** 2
            term2 = (1 - x[i]) ** 2
            sum_val += term1 + term2

        return float(sum_val)

    def fetch_trial_data(
        self, trial: BaseTrial, **kwargs: Any
    ) -> MetricFetchResult:
        try:
            noise_sd = 0.0 if self.noiseless else float("nan")
            arm_names = []
            mean = []
            sem = []
            trial_indices = []

            for name, arm in trial.arms_by_name.items():
                try:
                    val = self._evaluate_rosenbrock(arm.parameters)
                    arm_names.append(name)
                    mean.append(val)
                    sem.append(noise_sd)
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


# ---------------------------------- use case ----------------------------------

# 经典2D Rosenbrock
# rosenbrock_metric = RosenbrockMetric(name="rosenbrock2d")

# 高维版本
# rosenbrock_10d = RosenbrockMetric(name="rosenbrock10d", dimension=10)
