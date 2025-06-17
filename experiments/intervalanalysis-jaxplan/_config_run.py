from functools import partial
from _experiment import get_domain_instance_experiment

#####################################################################################################################################
# This file provides configuration for the experiments directly in python variables,
# as a DSL (domain-specific language)
#####################################################################################################################################

jax_seeds = [
    13, 31, 42, 61, 101,
    # 13, 31, 42, 61, 101, 103, 107, 109, 113, 127
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
    train_seconds=7_200, # 2 hours, JaxPlan stops training after this time or if the number of epochs is reached,
    eval_trials=5,
    batch_size=32,
    epochs=10_000,
)

experiments = {
    # Continuous Domains
    # 'uav': domain_instance_experiment('UAV_ippc2023', '3', 
    #                                   model_weight=571.09858689201, policy_hyperparams=0.9991885068952715,
    #                                   learning_rate=0.6558898576585526, policy_variance=0.052044741103344686, 
    #                                   topology=[20], eval_trials=1),
    'reservoir': domain_instance_experiment('Reservoir_ippc2023', '3', model_weight=1325.7003451079236, policy_hyperparams=0.11271918964372518,
                                            learning_rate=0.3353098796250857, policy_variance=0.034019297121515385,
                                            topology=[8]),
    
    # Continuous and Discrete (Mixed) Domains
    # 'marsrover': domain_instance_experiment('MarsRover', 'inst_6_rovers_7_minerals', model_weight=9733.585251841243, learning_rate=0.010597465543046915, topology=[174, 92], policy_hyperparams=60.07439949582946126, eval_trials=1),
    'hvac': domain_instance_experiment('HVAC', 'inst_5_zones_5_heaters', model_weight=0.1, policy_hyperparams=0.01,
                                       learning_rate=1e-05, policy_variance=0.01, 
                                       topology=[8], iter_cutting_point=4_000), 
    'powergen': domain_instance_experiment('PowerGen', 'inst_5_gen', model_weight=10000.0, policy_hyperparams=100.0,
                                       learning_rate=1e-05, policy_variance=100.0, 
                                       topology=[256], iter_cutting_point=8_000),
    
    # Discrete Domains
    # 'wildfire': domain_instance_experiment('Wildfire_MDP_ippc2014', '5', model_weight=1062.8867303429126, learning_rate=0.013984638078471815, topology=[43, 121], policy_hyperparams=0.9278118476829581, iter_cutting_point=10_000),
    'sysadmin': domain_instance_experiment('SysAdmin', 'instance2', model_weight=104.67829342328949, policy_hyperparams=2.9671078266598125,
                                           learning_rate=0.00040480533477448575, policy_variance=0.06033709609302027, 
                                           topology=[78], iter_cutting_point=10_000),
    'tireworld': domain_instance_experiment('TriangleTireworld', 'instance4', model_weight=1452.8246637516033, policy_hyperparams=0.07068974950624601,
                                           learning_rate=0.7072114131472232, policy_variance=0.05337032762603955, 
                                           topology=[8], iter_cutting_point=10_000),
}

def get_experiments():
    from argparse import ArgumentParser 
    parser = ArgumentParser()
    parser.add_argument('--domains', type=str, nargs='+', default=None)
    args = parser.parse_args()
    
    if not args.domains:
        return experiments.values()
    
    return [experiments[domain] for domain in args.domains]