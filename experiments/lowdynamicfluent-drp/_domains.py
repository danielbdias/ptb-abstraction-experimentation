import optax
from dataclasses import dataclass
from typing import Dict, List, Set

@dataclass(frozen=True)
class DomainExperiment:
    name:                      str
    instance:                  str
    action_bounds:             Dict
    state_fluents:             List[str]
    policy_hyperparams:        Dict
    ground_fluents_to_freeze:  Set[str]
    experiment_params:         Dict
    metadata:                  Dict = None     

jax_seeds = [
    42
    # 42, 101, 967, 103, 61, 
    # 107, 647, 109, 347, 113, 
    # 139, 127, 367, 131, 13, 137, 971, 139, 31, 149
]

domains = [
    DomainExperiment(
        name='HVAC',
        instance='instance1',
        action_bounds={},
        state_fluents=['occupied', 'temp-heater', 'temp-zone'],
        policy_hyperparams=None,
        ground_fluents_to_freeze=set(['occupied___z1', 'occupied___z2']),
        experiment_params = {
            'batch_size_train': 256,
            'optimizer': optax.rmsprop,
            'learning_rate': 0.1,
            'epochs': 6000,
            'epsilon_error': 0.001,
            'epsilon_iteration_stop': 100,
        },
        metadata={
            'topology': [64, 64]
        }
    ),
    DomainExperiment(
        name='Reservoir',
        instance='instance2',
        action_bounds={},
        state_fluents=['rlevel'],
        policy_hyperparams=None,
        ground_fluents_to_freeze=set(['rlevel___t2', 'rlevel___t5']),
        experiment_params = {
            'batch_size_train': 256,
            'optimizer': optax.rmsprop,
            'learning_rate': 0.0002,
            'epochs': 5000,
            'epsilon_error': 0.001,
            'epsilon_iteration_stop': 100,
        },
        metadata={
            'topology': [64, 32]
        }
    )
]

silent = True

bins = 100

# 'batch_size_train': 256,
# 'optimizer': optax.rmsprop,
# 'learning_rate': 0.1,
# 'epochs': 1000,
# 'epsilon_error': 0.001,
# 'epsilon_iteration_stop': 10,