#!/usr/bin/env python
# -*- coding: utf-8 -*-
r"""
@DATE: 2025-03-04 10:03:29
@File: src/bo/models.py
@IDE: vscode
@Description:
    BO model
"""

from ax.modelbridge.registry import Models
from ax.models.torch.botorch_modular.surrogate import SurrogateSpec
from ax.models.torch.botorch_modular.utils import ModelConfig
from botorch.acquisition.logei import qLogNoisyExpectedImprovement
from botorch.models.gp_regression import SingleTaskGP

import numpy as np
from ax.plot.trace import optimization_trace_single_method
from ax.utils.notebook.plotting import render
import json
from src.config import Config

config = Config()


class BOModel:
    def __init__(self, experiment):
        self.experiment = experiment
        self.model_bridge = None

    def hot_start(self, experiment, num_sobol_trials):
        NUM_SOBOL_TRIALS = num_sobol_trials
        print(f"Running Sobol initialization trials...")
        sobol = Models.SOBOL(search_space=experiment.search_space)
        for i in range(NUM_SOBOL_TRIALS):
            generator_run = sobol.gen(n=1)
            trial = experiment.new_trial(generator_run=generator_run)
            trial.run()
            trial.mark_completed()

    def gen(self, n):
        print(
            f"Start using BO algorithms to generate bo_recommendations candidates..."
        )
        self.model_bridge = Models.BOTORCH_MODULAR(
            experiment=self.experiment,
            data=self.experiment.fetch_data(),
            surrogate_spec=SurrogateSpec(
                model_configs=[
                    ModelConfig(botorch_model_class=SingleTaskGP),
                ]
            ),
            botorch_acqf_class=qLogNoisyExpectedImprovement,
        )

        print("Done!\n")
        return self.model_bridge.gen(n=n)

    def easy_render_hartmann6(self):
        objective_means = np.array(
            [
                [
                    trial.objective_mean
                    for trial in self.experiment.trials.values()
                ]
            ]
        )
        best_objective_plot = optimization_trace_single_method(
            y=np.minimum.accumulate(objective_means, axis=1),
            optimum=-3.32237,  # Known minimum objective for Hartmann6 function.
        )
        render(best_objective_plot)
