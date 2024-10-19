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
    # test case to validate interval analysis
    ###########################################################

    # DomainExperiment( 
    #     name='SimplifiedReservoir',
    #     instance='instance1',
    #     action_bounds={},
    #     action_bounds_for_interval_analysis=None,
    #     state_fluents=['rlevel'],
    #     policy_hyperparams=None,
    #     ground_fluents_to_freeze=set([]),
    #     experiment_params = {
    #         'batch_size_train': 256,
    #         'optimizer': optax.rmsprop,
    #         'learning_rate': 0.2,
    #         'epochs': 1000,
    #         'epsilon_error': 0.001,
    #         'epsilon_iteration_stop': 100,
    #     }
    # ),

    ###########################################################
    # IPPC 23 domains
    ###########################################################

    # DomainExperiment(
    #     name='HVAC',
    #     instance='instance3',
    #     action_bounds={},
    #     action_bounds_for_interval_analysis=None,
    #     state_fluents=['occupied', 'temp-heater', 'temp-zone'],
    #     policy_hyperparams=None,
    #     ground_fluents_to_freeze=set([]),
    #     experiment_params = {
    #         'batch_size_train': 256,
    #         'optimizer': optax.rmsprop,
    #         'learning_rate': 0.1,
    #         'epochs': 2000,
    #         'epsilon_error': 0.001,
    #         'epsilon_iteration_stop': 50,
    #     }
    # ),
    # DomainExperiment(
    #     name='Reservoir',
    #     instance='instance3',
    #     action_bounds={},
    #     action_bounds_for_interval_analysis=None,
    #     state_fluents=['rlevel'],
    #     policy_hyperparams=None,
    #     ground_fluents_to_freeze=set([]),
    #     experiment_params = {
    #         'batch_size_train': 256,
    #         'optimizer': optax.rmsprop,
    #         'learning_rate': 0.2,
    #         'epochs': 1000,
    #         'epsilon_error': 0.001,
    #         'epsilon_iteration_stop': 100,
    #     }
    # ),
    # DomainExperiment(
    #     name='PowerGen',
    #     instance='instance3',
    #     action_bounds={},
    #     action_bounds_for_interval_analysis=None,
    #     state_fluents=['prevProd', 'prevOn', 'temperature'],
    #     policy_hyperparams=None,
    #     ground_fluents_to_freeze=set([]),
    #     experiment_params = {
    #         'batch_size_train': 256,
    #         'optimizer': optax.rmsprop,
    #         'learning_rate': 0.05,
    #         'epochs': 3000,
    #         'epsilon_error': 0.001,
    #         'epsilon_iteration_stop': 100,
    #     }
    # ),

    ###########################################################
    # Modified domains
    ###########################################################

    DomainExperiment(
        name='ModifiedReservoir',
        instance='instance_small',
        action_bounds={},
        action_bounds_for_interval_analysis=None,
        state_fluents=['rlevel'],
        policy_hyperparams=None,
        ground_fluents_to_freeze=set(['rlevel___t3']),
        experiment_params = {
            'batch_size_train': 256,
            'optimizer': optax.rmsprop,
            'learning_rate': 0.2,
            'epochs': 1000,
            'epsilon_error': 0.001,
            'epsilon_iteration_stop': 100,
        }
    ),
    # DomainExperiment(
    #     name='ModifiedReservoir',
    #     instance='instance_medium',
    #     action_bounds={},
    #     action_bounds_for_interval_analysis=None,
    #     state_fluents=['rlevel'],
    #     policy_hyperparams=None,
    #     ground_fluents_to_freeze=set(['rlevel___t3', 'rlevel___t6']),
    #     experiment_params = {
    #         'batch_size_train': 256,
    #         'optimizer': optax.rmsprop,
    #         'learning_rate': 0.2,
    #         'epochs': 1000,
    #         'epsilon_error': 0.001,
    #         'epsilon_iteration_stop': 100,
    #     }
    # ),
]

silent = True

bins = 100

# 'batch_size_train': 256,
# 'optimizer': optax.rmsprop,
# 'learning_rate': 0.1,
# 'epochs': 1000,
# 'epsilon_error': 0.001,
# 'epsilon_iteration_stop': 10,