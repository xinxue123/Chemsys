#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-28 16:50:55
@File: src/tasks/maths/hartmann6.py
@IDE: vscode
@Description:
    Metric for evaluating Hartmann6 synthetic function.
"""

from __future__ import annotations

import numpy as np
import numpy.typing as npt
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.types import TParameterization
from ax.utils.common.result import Err, Ok
from pyre_extensions import none_throws
from typing import Any
import pandas as pd


class Hartmann6Metric(Metric):
    """Metric for evaluating Hartmann6 synthetic function.

    Args:
        name: The name of the metric.
        noiseless: If True, consider observations noiseless, otherwise
            assume unknown Gaussian observation noise.
        lower_is_better: If True, the metric should be minimized.
    """

    def __init__(
        self,
        name: str,
        noiseless: bool = False,
        lower_is_better: bool = True,  # Typically we minimize Hartmann6
    ) -> None:
        self.noiseless = noiseless
        super().__init__(name=name, lower_is_better=lower_is_better)

    def clone(self) -> Hartmann6Metric:
        return self.__class__(
            name=self._name,
            noiseless=self.noiseless,
            lower_is_better=none_throws(self.lower_is_better),
        )

    def _evaluate_hartmann6(self, params: TParameterization) -> float:
        """Evaluate Hartmann6 function at given parameters."""
        # Hartmann6 function implementation
        alpha = np.array([1.0, 1.2, 3.0, 3.2])
        A = np.array(
            [
                [10, 3, 17, 3.5, 1.7, 8],
                [0.05, 10, 17, 0.1, 8, 14],
                [3, 3.5, 1.7, 10, 17, 8],
                [17, 8, 0.05, 10, 0.1, 14],
            ]
        )
        P = 1e-4 * np.array(
            [
                [1312, 1696, 5569, 124, 8283, 5886],
                [2329, 4135, 8307, 3736, 1004, 9991],
                [2348, 1451, 3522, 2883, 3047, 6650],
                [4047, 8828, 8732, 5743, 1091, 381],
            ]
        )

        # Convert parameters to numpy array in correct order
        x = np.array(
            [params[param_name] for param_name in sorted(params.keys())]
        )

        y = 0.0
        for j, alpha_j in enumerate(alpha):
            t = 0
            for k in range(6):
                t += A[j, k] * ((x[k] - P[j, k]) ** 2)
            y -= alpha_j * np.exp(-t)
        return float(y)

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
                    val = self._evaluate_hartmann6(arm.parameters)
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

# hartmann6_metric = Hartmann6Metric(dimension=2)
