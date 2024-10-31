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
    #best parameters found - Reservoir: {'std': 1.7730401264602842e-05, 'lr': 0.0041021388862297345, 'w': 2499.926664413202}
    DomainExperiment(
        name                     = 'SplittedReservoir',
        instance                 = 'reservoir_full',
        state_fluents            = [ 'rlevel' ],
        ground_fluents_to_freeze = set([]),
        experiment_params=PlannerParameters(
            epsilon_error          = 0.01,
            epsilon_iteration_stop = 200,
            model_params=PlanningModelParameters(
                logic=FuzzyLogic(
                    tnorm  = ProductTNorm(),
                    weight = 10.0
                )
            ),
            optimizer_params=OptimizerParameters(
                plan             = None,
                optimizer        = optax.rmsprop,
                learning_rate    = 0.004,
                batch_size_train = 256,
                batch_size_test  = 256,
                action_bounds    = None,
            ),
            training_params=TrainingParameters(
                seed               = 42,
                epochs             = 1000,
                train_seconds      = 120,
                policy_hyperparams = None
            )
        )
    ),
    DomainExperiment(
        name                     = 'SplittedReservoir',
        instance                 = 'reservoir_right',
        state_fluents            = [ 'rlevel' ],
        ground_fluents_to_freeze = set([]),
        experiment_params=PlannerParameters(
            epsilon_error          = 0.01,
            epsilon_iteration_stop = 200,
            model_params=PlanningModelParameters(
                logic=FuzzyLogic(
                    tnorm  = ProductTNorm(),
                    weight = 10.0
                )
            ),
            optimizer_params=OptimizerParameters(
                plan             = None,
                optimizer        = optax.rmsprop,
                learning_rate    = 0.004,
                batch_size_train = 256,
                batch_size_test  = 256,
                action_bounds    = None,
            ),
            training_params=TrainingParameters(
                seed               = 42,
                epochs             = 1000,
                train_seconds      = 120,
                policy_hyperparams = None
            )
        )
    ),
    DomainExperiment(
        name                     = 'SplittedReservoir',
        instance                 = 'reservoir_left',
        state_fluents            = [ 'rlevel' ],
        ground_fluents_to_freeze = set([]),
        experiment_params=PlannerParameters(
            epsilon_error          = 0.01,
            epsilon_iteration_stop = 200,
            model_params=PlanningModelParameters(
                logic=FuzzyLogic(
                    tnorm  = ProductTNorm(),
                    weight = 10.0
                )
            ),
            optimizer_params=OptimizerParameters(
                plan             = None,
                optimizer        = optax.rmsprop,
                learning_rate    = 0.004,
                batch_size_train = 256,
                batch_size_test  = 256,
                action_bounds    = None,
            ),
            training_params=TrainingParameters(
                seed               = 42,
                epochs             = 1000,
                train_seconds      = 120,
                policy_hyperparams = None
            )
        )
    ),
]

silent = True