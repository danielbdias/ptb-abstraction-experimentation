import optax
from dataclasses import dataclass
from typing import Dict, List, Set

@dataclass(frozen=True)
class DomainExperiment:
    name:                                str
    instance:                            str
    action_bounds:                       Dict
    action_bounds_for_interval_analysis: Dict
    state_fluents:                       List[str]
    policy_hyperparams:                  Dict
    ground_fluents_to_freeze:            Set[str]
    experiment_params:                   Dict

jax_seeds = [
    42
    # 42, 101, 967, 103, 61, 
    # 107, 647, 109, 347, 113, 
    # 139, 127, 367, 131, 13, 137, 971, 139, 31, 149
]

domains = [
    ###########################################################
    # Modified domains
    ###########################################################

    #best parameters found - Reservoir: {'std': 1.7730401264602842e-05, 'lr': 0.0041021388862297345, 'w': 2499.926664413202}
    DomainExperiment(
        name='Reservoir',
        instance='instance_small',
        action_bounds={},
        action_bounds_for_interval_analysis=None,
        state_fluents=['rlevel'],
        policy_hyperparams=None,
        ground_fluents_to_freeze=set(['rlevel___t2']),
        
        experiment_params = {
            'batch_size_train': 256,
            'batch_size_test': 256,
            'optimizer': optax.rmsprop,
            'learning_rate': 0.004,
            'epochs': 1000,
            'epsilon_error': 0.01,
            'epsilon_iteration_stop': 200,
            'train_seconds': 120,
        }
    ),
    DomainExperiment(
        name='Reservoir',
        instance='instance_medium',
        action_bounds={},
        action_bounds_for_interval_analysis=None,
        state_fluents=['rlevel'],
        policy_hyperparams=None,
        ground_fluents_to_freeze=set(['rlevel___t3', 'rlevel___t6']),
        experiment_params = {
            'batch_size_train': 256,
            'batch_size_test': 256,
            'optimizer': optax.rmsprop,
            'learning_rate': 0.004,
            'epochs': 1000,
            'epsilon_error': 0.01,
            'epsilon_iteration_stop': 200,
            'train_seconds': 120,
        }
    ),
    DomainExperiment(
        name='Reservoir',
        instance='instance_large',
        action_bounds={},
        action_bounds_for_interval_analysis=None,
        state_fluents=['rlevel'],
        policy_hyperparams=None,
        ground_fluents_to_freeze=set(['rlevel___t3', 'rlevel___t4', 'rlevel___t6', 'rlevel___t7']),
        experiment_params = {
            'batch_size_train': 256,
            'batch_size_test': 256,
            'optimizer': optax.rmsprop,
            'learning_rate': 0.004,
            'epochs': 1000,
            'epsilon_error': 0.01,
            'epsilon_iteration_stop': 200,
            'train_seconds': 120,
        }
    ),
]

silent = True