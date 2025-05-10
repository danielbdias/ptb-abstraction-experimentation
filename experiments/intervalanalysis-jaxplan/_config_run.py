from _experiment import DomainInstanceExperiment, get_planner_parameters

#####################################################################################################################################
# This file provides configuration for the experiments directly in python variables,
# as a DSL (domain-specific language)
#####################################################################################################################################

jax_seeds = [
    42
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

train_seconds = 3600 # 1 hour, JaxPlan stops training after this time or if the number of epochs is reached

silent = True

run_drp = True
run_slp = False

def domain_instance_experiment(domain_name, instance_name, iter_cutting_point, slp_experiment_params=None, drp_experiment_params=None):
    experiment = DomainInstanceExperiment(
        domain_name = domain_name, 
        instance_name = instance_name, 
        iter_cutting_point = iter_cutting_point,
        ground_fluents_to_freeze = set(),
        bound_strategies = bound_strategies,
        slp_experiment_params = slp_experiment_params,
        drp_experiment_params = drp_experiment_params
    )

    if slp_experiment_params is not None:
        experiment.slp_experiment_params.training_params.train_seconds = train_seconds
    if drp_experiment_params is not None:
        experiment.drp_experiment_params.training_params.train_seconds = train_seconds
    
    return experiment

experiments = [
    # Continuous Domains
    domain_instance_experiment(
        'UAV_ippc2023', '3', iter_cutting_point=4_000,
        drp_experiment_params = get_planner_parameters(model_weight=64.95630307266005, learning_rate=0.01662497762967736, batch_size=32, epochs=10_000, topology=[23, 163], policy_hyperparams=0.013498252680281307)
    ),
    domain_instance_experiment(
        'MountainCar_ippc2023', '1', iter_cutting_point=4_000,
        drp_experiment_params = get_planner_parameters(model_weight=98.46738873614564, learning_rate=0.04570563099801451, batch_size=32, epochs=10_000, topology=[29, 215], policy_hyperparams=0.04207988669606635)
    ),
    domain_instance_experiment(
        'Reservoir_ippc2023', '3', iter_cutting_point=4_000,
        drp_experiment_params = get_planner_parameters(model_weight=101.29197956845731, learning_rate=0.2142302175774106, batch_size=32, epochs=10_000, topology=[13, 9], policy_hyperparams=6.7965780907581514)
    ),
    
    # domain_instance_experiment(
    #     'Reservoir', 'inst_10_reservoirs', iter_cutting_point=4_000,
    #     drp_experiment_params = get_planner_parameters(model_weight=10, learning_rate=0.0002, batch_size=32, epochs=10_000, topology=[64, 32])
    # ),
    
    # Continuous and Discrete (Mixed) Domains
    # domain_instance_experiment(
    #     'HVAC', 'inst_5_zones_5_heaters', iter_cutting_point=10_000,
    #     drp_experiment_params = get_planner_parameters(model_weight=5, learning_rate=0.001, batch_size=32, epochs=10_000, topology=[64, 64])
    # ),
    # domain_instance_experiment(
    #     'PowerGen', 'inst_5_gen', iter_cutting_point=8_000,
    #     drp_experiment_params = get_planner_parameters(model_weight=10, learning_rate=0.0001, batch_size=32, epochs=10_000, topology=[256, 128])
    # ),
    
    # Continuous Domains
    # domain_instance_experiment(
    #     'UAV', 'instance4', iter_cutting_point=10_000,
    #     drp_experiment_params = get_planner_parameters(model_weight=1, learning_rate=0.0004, batch_size=32, epochs=10_000, topology=[64, 32])
    # ),
    # domain_instance_experiment(
    #     'MountainCar', 'instance1', iter_cutting_point=10_000,
    #     drp_experiment_params = get_planner_parameters(model_weight=10, learning_rate=0.1, batch_size=32, epochs=10_000, topology=[64, 32])
    # ),
    
    # Continuous and Discrete (Mixed) Domains
    # domain_instance_experiment(
    #     'MarsRover', 'inst_6_rovers_7_minerals',
    #     # slp_experiment_params = get_planner_parameters(model_weight=10, learning_rate=0.2, batch_size=32, epochs=1_000),
    #     drp_experiment_params = get_planner_parameters(model_weight=10, learning_rate=0.0002, batch_size=32, epochs=10_000, topology=[64, 32])
    # ),
    
    # Discrete Domains
    # domain_instance_experiment(
    #     'Wildfire', 'inst_5x5_grid',
    #     drp_experiment_params = get_planner_parameters(model_weight=100, learning_rate=0.001, batch_size=32, epochs=10_000, topology=[128, 64])
    # ),
    # domain_instance_experiment(
    #     'SysAdmin', 'instance2',
    #     drp_experiment_params = get_planner_parameters(model_weight=100, learning_rate=0.001, batch_size=32, epochs=10_000, topology=[128, 64])
    # ),
    # domain_instance_experiment(
    #     'TriangleTireworld', 'instance4',
    #     drp_experiment_params = get_planner_parameters(model_weight=100, learning_rate=0.001, batch_size=32, epochs=10_000, topology=[128, 64])
    # ),
    # 
]