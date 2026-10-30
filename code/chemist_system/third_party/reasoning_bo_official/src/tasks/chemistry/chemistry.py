#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-17 13:59:52
@File: src/metrics/chemistry.py
@IDE: vscode
@Description:
    class for optimizating yields from chemical reactions.
    adapted from https://github.com/facebook/Ax/blob/main/ax/metrics/chemistry.py
"""

from __future__ import annotations

"""
Classes for optimizing yields from chemical reactions.

References

.. [Perera2018]
    D. Perera, J. W. Tucker, S. Brahmbhatt, C. Helal, A. Chong, W. Farrell,
    P. Richardson, N. W. Sach. A platform for automated nanomole-scale
    reaction screening and micromole-scale synthesis in flow. Science, 26.
    2018.

.. [Shields2021]
   B. J. Shields, J. Stevens, J. Li, et al. Bayesian reaction optimization
   as a tool for chemical synthesis. Nature 590, 89–96 (2021).

"SUZUKI" involves optimization solvent, ligand, and base combinations
in a Suzuki-Miyaura coupling to optimize carbon-carbon bond formation.
See _[Perera2018] for details.

"DIRECT_ARYLATION" involves optimizing the solvent, base, and ligand chemicals
as well as the temperature and concentration for a direct arylation reaction.
See _[Shields2021] for details.
"""


from dataclasses import dataclass
from enum import Enum
from functools import lru_cache
from pathlib import Path
from typing import Any
from zipfile import ZipFile

import pandas as pd
from ax.core.base_trial import BaseTrial
from ax.core.data import Data
from ax.core.metric import Metric, MetricFetchE, MetricFetchResult
from ax.core.types import TParameterization, TParamValue
from ax.utils.common.result import Err, Ok
from pyre_extensions import none_throws


class ChemistryProblemType(Enum):
    SUZUKI: str = "suzuki"
    DIRECT_ARYLATION: str = "direct_arylation"
    Buchwald_Hartwig: str = "Buchwald_Hartwig"
    CPA: str = "CPA"


@dataclass(frozen=True)
class ChemistryData:
    param_names: list[str]
    objective_dict: dict[tuple[TParamValue, ...], float]

    def evaluate(self, params: TParameterization) -> float:
        k = tuple(params[pname] for pname in self.param_names)
        return self.objective_dict[k]


@lru_cache(maxsize=8)
def _get_data(problem_type: ChemistryProblemType) -> ChemistryData:
    file_path = (
        Path(__file__)
        .parent.parent.parent.parent.joinpath("data/chemistry_data.zip")
        .absolute()
    )

    with ZipFile(file_path) as zf:
        with zf.open(f"{problem_type.value}.csv") as f:
            df = pd.read_csv(f, index_col=0)
    param_names = sorted(col for col in df.columns if col != "yield")
    return ChemistryData(
        param_names=param_names,
        objective_dict=df.set_index(param_names)["yield"].to_dict(),
    )


class ChemistryMetric(Metric):
    """Metric for modeling chemical reactions.

    Metric describing the outcomes of chemical reactions. Based on tabulate data.
    Problems typically contain many discrete and categorical parameters.

    Args:
        name: The name of the metric.
        noiseless: If True, consider observations noiseless, otherwise
        sume unknown Gaussian observation noise.
        problem_type: The problem type.

    Attributes:
        noiseless: If True, consider observations noiseless, otherwise
            assume unknown Gaussian observation noise.
        lower_is_better: If True, the metric should be minimized.
    """

    def __init__(
        self,
        name: str,
        noiseless: bool = False,
        problem_type: ChemistryProblemType = ChemistryProblemType.SUZUKI,
        lower_is_better: bool = False,
    ) -> None:
        self.noiseless = noiseless
        self.problem_type = problem_type
        super().__init__(name=name, lower_is_better=lower_is_better)

    def clone(self) -> ChemistryMetric:
        return self.__class__(
            name=self._name,
            noiseless=self.noiseless,
            problem_type=self.problem_type,
            lower_is_better=none_throws(self.lower_is_better),
        )

    def fetch_trial_data(
        self, trial: BaseTrial, **kwargs: Any
    ) -> MetricFetchResult:
        try:
            noise_sd = 0.0 if self.noiseless else float("nan")
            data = _get_data(self.problem_type)
            arm_names = []
            mean = []
            sem = []
            trial_indices = []
            skipped_arms = 0
            print(f"\nDebug: Dataset parameter names: {data.param_names}")
            print(
                f"Debug: First trial arm parameters: {next(iter(trial.arms_by_name.values())).parameters.keys()}\n"
            )

            # Filter arms with valid parameters
            for name, arm in trial.arms_by_name.items():
                try:
                    # First validate all parameters exist in the data
                    missing_params = [
                        p for p in arm.parameters if p not in data.param_names
                    ]
                    if missing_params:
                        print(f"Debug: Skipping - Parameter mismatch")
                        print(f"Arm params: {sorted(arm.parameters.keys())}")
                        print(f"Data params: {sorted(data.param_names)}")
                        print(f"Missing params: {missing_params}")
                        skipped_arms += 1
                        continue

                    val = data.evaluate(params=arm.parameters)
                    print(f"Debug: Found yield value: {val}")

                    arm_names.append(name)
                    mean.append(val)
                    sem.append(noise_sd)
                    trial_indices.append(trial.index)
                except KeyError as e:
                    print(f"Debug: KeyError - {str(e)}")
                    # Skip arms with missing data
                    skipped_arms += 1
                    continue

            if skipped_arms > 0:
                print(
                    f"Warning: Skipped {skipped_arms}/{len(trial.arms_by_name)} "
                    f"arms due to missing data in {self.problem_type.value} dataset"
                )

            if not arm_names:
                print("\nDebug: All arms skipped. Details:")
                print(f"Total arms: {len(trial.arms_by_name)}")
                print(f"Skipped arms: {skipped_arms}")
                return Err(
                    MetricFetchE(
                        message=f"No valid arms found for {self.name}",
                        exception=ValueError("All arms have missing data"),
                    )
                )

            # Ensure all arrays have the same length
            assert (
                len(arm_names) == len(mean) == len(sem) == len(trial_indices)
            ), "All arrays must have the same length"

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
# chem_metric = ChemistryMetric()
