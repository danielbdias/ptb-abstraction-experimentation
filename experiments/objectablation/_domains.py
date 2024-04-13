import optax
from dataclasses import dataclass

@dataclass(frozen=True)
class DomainExperiment:
    name:              str
    instance:          str
    action_bounds:     dict
    variables_removed: list

jax_seeds = [
    42, 101, 967, 103, 61, 107, 647, 109, 347, 113, 139, 127, 367, 131, 13, 137, 971, 139, 31, 149
]

domains = [
    DomainExperiment(
        name='HVAC',
        instance='instance1',
        action_bounds={'fan-in': (0.05001, None), 'heat-input': (0.0, None)},
        variables_removed=['occupied', 'tempzone']
    ),
    DomainExperiment(
        name='UAV',
        instance='instance1',
        action_bounds={'set-acc': (-1, 1), 'set-phi': (-1, 1), 'set-theta': (-1, 1)},
        variables_removed=['pos-x', 'pos-y', 'pos-z']
    )
]

silent = True

experiment_params = {
    'batch_size_train': 256,
    'optimizer': optax.rmsprop,
    'learning_rate': 0.1,
    'epochs': 1000,
    'report_statistics_interval': 1,
    'epsilon_error': 0.001,
    'epsilon_iteration_stop': 10,
}