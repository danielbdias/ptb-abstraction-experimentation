import optax
from dataclasses import dataclass
from typing import Dict, List, Set

from _utils import PlannerParameters, PlanningModelParameters, OptimizerParameters, TrainingParameters

from pyRDDLGym_jax.core.logic import ProductTNorm, FuzzyLogic

@dataclass(frozen=True)
class DomainExperiment:
    name:                                str
    instance:                            str
    state_fluents:                       List[str]
    ground_fluents_to_freeze:            Set[str]
    experiment_params:                   PlannerParameters

jax_seeds = [
    42
    # 42, 101, 967, 103, 61, 
    # 107, 647, 109, 347, 113, 
    # 139, 127, 367, 131, 13, 137, 971, 139, 31, 149
]

domains = [
    ########################################################################################################################
    # Mars Rover
    # {'std': 0.4281115676561741, 'lr': 100.0, 'w': 100000.0, 'wa': 1.0999397093732934}
    ########################################################################################################################
    DomainExperiment(
        name                     = 'MarsRover',
        instance                 = 'instance3',
        state_fluents            = [ 'mineral-harvested', 'vel-x', 'pos-x', 'vel-y', 'pos-y' ],
        ground_fluents_to_freeze = set([]),
        experiment_params=PlannerParameters(
            epsilon_error          = 0.01,
            epsilon_iteration_stop = 200,
            model_params=PlanningModelParameters(
                logic=FuzzyLogic(
                    tnorm  = ProductTNorm(),
                    weight = 1000.0
                )
            ),
            optimizer_params=OptimizerParameters(
                plan             = None,
                optimizer        = optax.rmsprop,
                learning_rate    = 1.0,
                batch_size_train = 256,
                batch_size_test  = 256,
                action_bounds    = {
                    'power-x': (-0.09999, 0.09999), 
                    'power-y': (-0.09999, 0.09999)
                },
            ),
            training_params=TrainingParameters(
                seed               = 42,
                epochs             = 1000,
                train_seconds      = 120,
                policy_hyperparams = { 'harvest': 5.0 }
            )
        )
    ),
    ########################################################################################################################
    # Power Generator
    ########################################################################################################################
    # DomainExperiment(
    #     name                     = 'PowerGen',
    #     instance                 = 'instance3',
    #     state_fluents            = [ 'prevProd', 'prevOn', 'temperature' ],
    #     ground_fluents_to_freeze = set([ 'prevOn___p1', 'prevOn___p2', 'prevOn___p3', 'prevOn___p4', 'prevOn___p5', 'temperature' ]),
    #     experiment_params=PlannerParameters(
    #         epsilon_error          = 0.01,
    #         epsilon_iteration_stop = 200,
    #         model_params=PlanningModelParameters(
    #             logic=FuzzyLogic(
    #                 tnorm  = ProductTNorm(),
    #                 weight = 10
    #             )
    #         ),
    #         optimizer_params=OptimizerParameters(
    #             plan             = None,
    #             optimizer        = optax.rmsprop,
    #             learning_rate    = 0.05,
    #             batch_size_train = 32,
    #             batch_size_test  = 32,
    #             action_bounds    = None,
    #         ),
    #         training_params=TrainingParameters(
    #             seed               = 42,
    #             epochs             = 10000,
    #             train_seconds      = 30,
    #             policy_hyperparams = None
    #         )
    #     )
    # ),
    ########################################################################################################################
    # Recommender Systems
    # {'std': 0.0008535555659132102, 'lr': 11.021651612071706, 'w': 1.0, 'wa': 144.82535968177223}
    ########################################################################################################################
    # DomainExperiment(
    #     name                     = 'RecSim',
    #     instance                 = 'instance1',
    #     state_fluents            = [ 'provider-satisfaction', 'consumer-satisfaction', 'item-feature', 'item-by' ],
    #     ground_fluents_to_freeze = set([]),
    #     experiment_params=PlannerParameters(
    #         epsilon_error          = 0.01,
    #         epsilon_iteration_stop = 200,
    #         model_params=PlanningModelParameters(
    #             logic=FuzzyLogic(
    #                 tnorm  = ProductTNorm(),
    #                 weight = 1.0
    #             )
    #         ),
    #         optimizer_params=OptimizerParameters(
    #             plan             = None, # To be defined on each experiment
    #             optimizer        = optax.rmsprop,
    #             learning_rate    = 11,
    #             batch_size_train = 32,
    #             batch_size_test  = 32,
    #             action_bounds    = {
    #                 'power-x': (-0.09999, 0.09999), 
    #                 'power-y': (-0.09999, 0.09999)
    #             },
    #         ),
    #         training_params=TrainingParameters(
    #             seed               = 42,
    #             epochs             = 1000,
    #             train_seconds      = 120,
    #             policy_hyperparams = None
    #         )
    #     )
    # ),
]

silent = True