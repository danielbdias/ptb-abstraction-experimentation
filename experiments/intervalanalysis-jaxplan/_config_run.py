from functools import partial
from _experiment import get_domain_instance_experiment

#####################################################################################################################################
# This file provides configuration for the experiments directly in python variables,
# as a DSL (domain-specific language)
#####################################################################################################################################

jax_seeds = [
    13, 31, 42, 61, 101, 103, 107, 109, 113, 127
    # 1 run:  42
    # 3 runs: 13, 31, 42
    # 10 runs: 13, 31, 42, 61, 101, 103, 107, 109, 113, 127
    # 20 runs: 13, 31, 42, 61, 101, 103, 107, 109, 113, 127, 131, 137, 139, 139, 149, 347, 367, 647, 967, 971
]

bound_strategies = {
    'support': None,
    'mean': None,
    'percentiles': (0.05, 0.95),
}

threshold_to_choose_fluents = [
    0.0, # 0% of the fluents
    0.1,
    0.2,
    0.3, # 30% of the fluents
    0.4,
    0.5,
    0.6,
    0.7, # 70% of the fluents
] 

silent = True

run_drp = True
run_slp = False

domain_instance_experiment = partial(
    get_domain_instance_experiment, 
    bound_strategies=bound_strategies,
    iter_cutting_point=10_000,
    train_seconds=30, # 1 hour, JaxPlan stops training after this time or if the number of epochs is reached,
    eval_trials=1,
    batch_size=1,
    epochs=10,
)

experiments = {
    # Continuous Domains
    'uav': domain_instance_experiment('UAV_ippc2023', '3', model_weight=64.95630307266005, learning_rate=0.01662497762967736, topology=[23, 163], policy_hyperparams=0.013498252680281307, eval_trials=1),
    'mountaincar': domain_instance_experiment('MountainCar_ippc2023', '1', model_weight=98.46738873614564, learning_rate=0.04570563099801451, topology=[29, 215], policy_hyperparams=0.04207988669606635, eval_trials=1),
    'reservoir': domain_instance_experiment('Reservoir_ippc2023', '3', model_weight=101.29197956845731, learning_rate=0.2142302175774106, topology=[13, 9], policy_hyperparams=6.7965780907581514, eval_trials=5),
    
    # Continuous and Discrete (Mixed) Domains
    'marsrover': domain_instance_experiment('MarsRover_ippc2023', '3', model_weight=9733.585251841243, learning_rate=0.010597465543046915, topology=[174, 92], policy_hyperparams=60.07439949582946126, eval_trials=1),
    'hvac': domain_instance_experiment('HVAC', 'inst_5_zones_5_heaters', model_weight=156.24303824917484, learning_rate=0.04034411767966345, topology=[30, 220], policy_hyperparams=0.04589450016024262, eval_trials=5, iter_cutting_point=10_000), 
    'powergen': domain_instance_experiment('PowerGen', 'inst_5_gen', model_weight=1.1526449540315609, learning_rate=0.14528246637516035, topology=[8, 230], policy_hyperparams=0.05337032762603955, eval_trials=5, iter_cutting_point=8_000),
    
    # Discrete Domains
    'wildfire': domain_instance_experiment('Wildfire_MDP_ippc2014', '5', model_weight=1062.8867303429126, learning_rate=0.013984638078471815, topology=[43, 121], policy_hyperparams=0.9278118476829581, eval_trials=1, iter_cutting_point=10_000),
    'sysadmin': domain_instance_experiment('SysAdmin', 'instance2', model_weight=100, learning_rate=0.001, topology=[128, 64], eval_trials=1, iter_cutting_point=10_000),
    'tireworld': domain_instance_experiment('TriangleTireworld', 'instance4', model_weight=100, learning_rate=0.001, topology=[128, 64], eval_trials=1, iter_cutting_point=10_000),
}

def get_experiments():
    from argparse import ArgumentParser 
    parser = ArgumentParser()
    parser.add_argument('--domains', type=str, nargs='+', default=None)
    args = parser.parse_args()
    
    if not args.domains:
        return experiments.values()
    
    return [experiments[domain] for domain in args.domains]