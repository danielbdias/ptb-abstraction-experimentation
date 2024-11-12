import optax
from dataclasses import dataclass
from typing import List, Set

from _utils import PlannerParameters, PlanningModelParameters, OptimizerParameters, TrainingParameters

from pyRDDLGym_jax.core.logic import ProductTNorm, FuzzyLogic
from pyRDDLGym.core.intervals import IntervalAnalysisStrategy

@dataclass(frozen=True)
class DomainExperiment:
    name:                                str
    instance:                            str
    state_fluents:                       List[str]
    ground_fluents_to_freeze:            Set[str]
    bound_strategies:                    dict
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
    # DomainExperiment(
    #     name                     = 'MarsRover',
    #     instance                 = 'instance3',
    #     state_fluents            = [ 'mineral-harvested', 'vel-x', 'pos-x', 'vel-y', 'pos-y' ],
    #     ground_fluents_to_freeze = set([]),
    #     experiment_params=PlannerParameters(
    #         epsilon_error          = 0.01,
    #         epsilon_iteration_stop = 200,
    #         model_params=PlanningModelParameters(
    #             logic=FuzzyLogic(
    #                 tnorm  = ProductTNorm(),
    #                 weight = 1000.0
    #             )
    #         ),
    #         optimizer_params=OptimizerParameters(
    #             plan             = None,
    #             optimizer        = optax.rmsprop,
    #             learning_rate    = 1.0,
    #             batch_size_train = 256,
    #             batch_size_test  = 256,
    #             action_bounds    = {
    #                 'power-x': (-0.09999, 0.09999), 
    #                 'power-y': (-0.09999, 0.09999)
    #             },
    #         ),
    #         training_params=TrainingParameters(
    #             seed               = 42,\\\
        
    #             train_seconds      = 120,
    #             policy_hyperparams = { 'harvest': 5.0 }
    #         )
    #     )
    # ),
    ########################################################################################################################
    # Power Generator
    ########################################################################################################################
    DomainExperiment(
        name                     = 'PowerGen',
        instance                 = 'instance3',
        state_fluents            = [ 'prevProd', 'prevOn', 'temperature' ],
        ground_fluents_to_freeze = set([ 'prevOn___p1', 'prevOn___p2', 'prevOn___p3', 'prevOn___p4', 'prevOn___p5' ]),
        bound_strategies         = {
            'support': (IntervalAnalysisStrategy.SUPPORT, {}),
            'mean': (IntervalAnalysisStrategy.MEAN, {}),
            'percentiles': (IntervalAnalysisStrategy.PERCENTILE, { 'percentiles': [0.05, 0.95] }),
        },
        experiment_params=PlannerParameters(
            # epsilon_error          = 0.01,
            # epsilon_iteration_stop = 3000,
            epsilon_error          = None,
            epsilon_iteration_stop = None,
            model_params=PlanningModelParameters(
                logic=FuzzyLogic(
                    tnorm  = ProductTNorm(),
                    weight = 10
                )
            ),
            optimizer_params=OptimizerParameters(
                plan             = None,
                optimizer        = optax.rmsprop,
                learning_rate    = 0.05,
                batch_size_train = 32,
                batch_size_test  = 32,
                action_bounds    = None,
            ),
            training_params=TrainingParameters(
                seed               = 42,
                epochs             = 15000,
                train_seconds      = 120,
                policy_hyperparams = None
            )
        )
    ),
    ########################################################################################################################
    # HVAC
    ########################################################################################################################
    # DomainExperiment(
    #     name                     = 'HVAC',
    #     instance                 = 'instance3',
    #     state_fluents            = [ 'temp-zone', 'temp-heater', 'occupied' ],
    #     ground_fluents_to_freeze = set([ 'occupied___z1', 'occupied___z2', 'occupied___z3', 'occupied___z4', 'occupied___z5' ]),
    #     bound_strategies         = {
    #         'support': (IntervalAnalysisStrategy.SUPPORT, {}),
    #         'mean': (IntervalAnalysisStrategy.MEAN, {}),
    #         'percentiles': (IntervalAnalysisStrategy.PERCENTILE, { 'percentiles': [0.05, 0.95] }),
    #     },
    #     experiment_params=PlannerParameters(
    #         # epsilon_error          = 0.01,
    #         # epsilon_iteration_stop = 3000,
    #         epsilon_error          = None,
    #         epsilon_iteration_stop = None,
    #         model_params=PlanningModelParameters(
    #             logic=FuzzyLogic(
    #                 tnorm  = ProductTNorm(),
    #                 weight = 10
    #             )
    #         ),
    #         optimizer_params=OptimizerParameters(
    #             plan             = None,
    #             optimizer        = optax.rmsprop,
    #             learning_rate    = 0.01,
    #             batch_size_train = 32,
    #             batch_size_test  = 32,
    #             action_bounds    = None,
    #         ),
    #         training_params=TrainingParameters(
    #             seed               = 42,
    #             epochs             = 10000,
    #             train_seconds      = 120,
    #             policy_hyperparams = None
    #         )
    #     )
    # ),
]

silent = True