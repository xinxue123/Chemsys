#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-28 18:48:28
@File: src/tasks/maths/levy.py
@IDE: vscode
@Description:
    Metric for evaluating Levy synthetic function.
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
import math


class LevyMetric(Metric):
    """Metric for evaluating Levy synthetic function.

    Args:
        name: The name of the metric.
        noiseless: If True, consider observations noiseless, otherwise
            assume unknown Gaussian observation noise.
        lower_is_better: If True, the metric should be minimized.
        dimension: Dimension of the Levy function (default=2).
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

    def clone(self) -> LevyMetric:
        return self.__class__(
            name=self._name,
            noiseless=self.noiseless,
            lower_is_better=none_throws(self.lower_is_better),
            dimension=self.dimension,
        )

    def _evaluate_levy(self, params: TParameterization) -> float:
        """Evaluate Levy function at given parameters."""
        # Convert parameters to numpy array
        x = np.array(
            [params[param_name] for param_name in sorted(params.keys())]
        )
        w = 1 + (x - 1) / 4

        # First term
        term1 = np.sin(np.pi * w[0]) ** 2

        # Middle terms
        term2 = 0.0
        for i in range(self.dimension - 1):
            term2 += (w[i] - 1) ** 2 * (1 + 10 * np.sin(np.pi * w[i] + 1) ** 2)

        # Last term
        term3 = (w[-1] - 1) ** 2 * (1 + np.sin(2 * np.pi * w[-1]) ** 2)

        return float(term1 + term2 + term3)

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
                    val = self._evaluate_levy(arm.parameters)
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

# 2D Levy
# levy_metric = LevyMetric(name="levy2d")

# 5D Levy
# levy_metric_5d = LevyMetric(name="levy5d", dimension=5)
