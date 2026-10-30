#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-28 16:52:43
@File: src/tasks/maths/ackley.py
@IDE: vscode
@Description:
    Metric for evaluating Ackley synthetic function.
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


class AckleyMetric(Metric):
    """Metric for evaluating Ackley synthetic function.

    Args:
        name: The name of the metric.
        noiseless: If True, consider observations noiseless, otherwise
            assume unknown Gaussian observation noise.
        lower_is_better: If True, the metric should be minimized.
        dimension: Dimension of the Ackley function (default=2).
        a: Parameter controlling depth of the basin (default=20).
        b: Parameter controlling width of the basin (default=0.2).
        c: Parameter controlling frequency of cosine term (default=2π).
    """

    def __init__(
        self,
        name: str,
        noiseless: bool = False,
        lower_is_better: bool = True,
        dimension: int = 2,
        a: float = 20.0,
        b: float = 0.2,
        c: float = 2 * math.pi,
    ) -> None:
        self.noiseless = noiseless
        self.dimension = dimension
        self.a = a
        self.b = b
        self.c = c
        super().__init__(name=name, lower_is_better=lower_is_better)

    def clone(self) -> AckleyMetric:
        return self.__class__(
            name=self._name,
            noiseless=self.noiseless,
            lower_is_better=none_throws(self.lower_is_better),
            dimension=self.dimension,
            a=self.a,
            b=self.b,
            c=self.c,
        )

    def _evaluate_ackley(self, params: TParameterization) -> float:
        """Evaluate Ackley function at given parameters."""
        # Convert parameters to numpy array
        x = np.array(
            [params[param_name] for param_name in sorted(params.keys())]
        )

        # First exponential term
        sum_sq = np.sum(x**2)
        term1 = -self.a * np.exp(-self.b * np.sqrt(sum_sq / self.dimension))

        # Second exponential term
        sum_cos = np.sum(np.cos(self.c * x))
        term2 = -np.exp(sum_cos / self.dimension)

        # Constant terms
        term3 = self.a + math.exp(1)

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
                    val = self._evaluate_ackley(arm.parameters)
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


# ---------------------------------- Use Example ----------------------------------
# 2D Ackley
# acley_metric = AckleyMetric(name="ackley2d", dimension=2)

# 10D Ackley with custom parameters
# acley_metric_10D = AckleyMetric(
#     name="ackley10D", dimension=10, a=15.0, b=0.1, c=3.0
# )
