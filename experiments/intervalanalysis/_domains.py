import optax
from dataclasses import dataclass
from typing import Set

from _utils import PlannerParameters, PlanningModelParameters, OptimizerParameters, TrainingParameters

from pyRDDLGym_jax.core.logic import ProductTNorm, FuzzyLogic, SigmoidComparison, SoftRounding, SoftControlFlow

@dataclass(frozen=True)
class DomainExperiment:
    name:                                str
    instance:                            str
    ground_fluents_to_freeze:            Set[str]
    bound_strategies:                    dict
    experiment_params:                   PlannerParameters
    
    def get_state_fluents(self, rddl_model):
        return list(rddl_model.state_fluents.keys())

jax_seeds = [
    42
    # 42, 101, 967, 103, 61, 
    # 107, 647, 109, 347, 113, 
    # 139, 127, 367, 131, 13, 137, 971, 139, 31, 149
]

bound_strategies = {
    'support': None,
    'mean': None,
    'percentiles': (0.05, 0.95),
}

bound_strategy_to_choose_fluents = 'mean'
threshold_to_choose_fluents = 0.3 # 30% of the fluents

train_seconds = 3600 # 1 hour, JaxPlan stops training after this time or if the number of epochs is reached

def get_planner_parameters(model_weight : int, learning_rate : float, batch_size : int, epochs : int, train_seconds: int, policy_hyperparams: dict = None):
    return PlannerParameters(
        model_params = PlanningModelParameters(
            logic=FuzzyLogic(
                tnorm      = ProductTNorm(),
                comparison = SigmoidComparison(weight=model_weight),
                rounding   = SoftRounding(weight=model_weight),
                control    = SoftControlFlow(weight=model_weight),
            )
        ),
        optimizer_params = OptimizerParameters(
            plan             = None,
            optimizer        = optax.rmsprop,
            learning_rate    = learning_rate,
            batch_size_train = batch_size,
            batch_size_test  = batch_size,
            action_bounds    = None,
            guess            = None
        ),
        training_params = TrainingParameters(
            seed               = 42,
            epochs             = epochs,
            train_seconds      = train_seconds,
            policy_hyperparams = policy_hyperparams
        )
    )

domains = [
    ##################################################################
    # HVAC
    ##################################################################
    DomainExperiment(
        name                     = 'HVAC',
        instance                 = 'instance_h_100',
        ground_fluents_to_freeze = set(),
        bound_strategies         = bound_strategies,
        experiment_params        = get_planner_parameters(model_weight=5, learning_rate=0.02, batch_size=32, epochs=16_000, train_seconds=train_seconds)
    ),
    ##################################################################
    # PowerGen
    ##################################################################
    DomainExperiment(
        name                     = 'PowerGen',
        instance                 = 'instance_h_100',
        ground_fluents_to_freeze = set(),
        # ground_fluents_to_freeze = set([ 'prevOn___p1', 'prevOn___p2', 'prevOn___p3', 'prevOn___p4', 'prevOn___p5' ]),
        bound_strategies         = bound_strategies,
        experiment_params        = get_planner_parameters(model_weight=10, learning_rate=0.05, batch_size=32, epochs=40_000, train_seconds=train_seconds)
    ),
    ##################################################################
    # MountainCar
    ##################################################################
    DomainExperiment(
        name                     = 'MountainCar',
        instance                 = 'instance1',
        ground_fluents_to_freeze = set(),
        bound_strategies         = bound_strategies,
        experiment_params        = get_planner_parameters(model_weight=10, learning_rate=1.0, batch_size=32, epochs=10_000, train_seconds=train_seconds)
    ),
    
    ##################################################################
    # Reservoir
    ##################################################################
    DomainExperiment(
        name                     = 'Reservoir',
        instance                 = 'instance_h_100',
        ground_fluents_to_freeze = set(),
        # ground_fluents_to_freeze = set([ 'rlevel___t3', 'rlevel___t4', 'rlevel___t7', 'rlevel___t10' ]),
        bound_strategies         = bound_strategies,
        experiment_params        = get_planner_parameters(model_weight=10, learning_rate=0.2, batch_size=32, epochs=12_000, train_seconds=train_seconds)
    ),
]

silent = True