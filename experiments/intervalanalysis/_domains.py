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
                    weight = 100
                )
            ),
            optimizer_params=OptimizerParameters(
                optimizer        = optax.rmsprop,
                learning_rate    = 1.0,
                batch_size_train = 32,
                batch_size_test  = 32,
                action_bounds    = {
                    'power-x': (-0.09999, 0.09999), 
                    'power-y': (-0.09999, 0.09999)
                },
            ),
            training_params=TrainingParameters(
                epochs             = 500,
                train_seconds      = 30,
                policy_hyperparams = { 'harvest': 5.0 }
            )
        )
    ),
    ########################################################################################################################
    # Power Generator
    ########################################################################################################################
    DomainExperiment(
        name                     = 'PowerGen',
        instance                 = 'instance3',
        state_fluents            = [ 'prevProd', 'prevOn', 'temperature' ],
        ground_fluents_to_freeze = set([]),
        experiment_params=PlannerParameters(
            epsilon_error          = 0.01,
            epsilon_iteration_stop = 200,
            model_params=PlanningModelParameters(
                logic=FuzzyLogic(
                    tnorm  = ProductTNorm(),
                    weight = 10
                )
            ),
            optimizer_params=OptimizerParameters(
                optimizer        = optax.rmsprop,
                learning_rate    = 0.05,
                batch_size_train = 32,
                batch_size_test  = 32,
                action_bounds    = {
                    'power-x': (-0.09999, 0.09999), 
                    'power-y': (-0.09999, 0.09999)
                },
            ),
            training_params=TrainingParameters(
                epochs             = 10000,
                train_seconds      = 30,
                policy_hyperparams = None
            )
        )
    ),
    ########################################################################################################################
    # Recommender Systems
    ########################################################################################################################
    DomainExperiment(
        name                     = 'RecSim',
        instance                 = 'instance1',
        state_fluents            = [ 'provider-satisfaction', 'consumer-satisfaction', 'item-feature', 'item-by' ],
        ground_fluents_to_freeze = set([]),
        experiment_params=PlannerParameters(
            epsilon_error          = 0.01,
            epsilon_iteration_stop = 200,
            model_params=PlanningModelParameters(
                logic=FuzzyLogic(
                    tnorm  = ProductTNorm(),
                    weight = 10
                )
            ),
            optimizer_params=OptimizerParameters(
                optimizer        = optax.rmsprop,
                learning_rate    = 0.05,
                batch_size_train = 32,
                batch_size_test  = 32,
                action_bounds    = {
                    'power-x': (-0.09999, 0.09999), 
                    'power-y': (-0.09999, 0.09999)
                },
            ),
            training_params=TrainingParameters(
                epochs             = 10000,
                train_seconds      = 30,
                policy_hyperparams = None
            )
        )
    ),
]

silent = True