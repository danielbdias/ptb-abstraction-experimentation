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
run_slp = True

def domain_instance_experiment(domain_name, instance_name, slp_experiment_params, drp_experiment_params):
    experiment = DomainInstanceExperiment(
        domain_name = domain_name, 
        instance_name = instance_name, 
        ground_fluents_to_freeze = set(),
        bound_strategies = bound_strategies,
        slp_experiment_params = slp_experiment_params,
        drp_experiment_params = drp_experiment_params
    )
    
    experiment.slp_experiment_params.training_params.train_seconds = train_seconds
    experiment.drp_experiment_params.training_params.train_seconds = train_seconds
    
    return experiment

experiments = [
    domain_instance_experiment(
        'HVAC', 'inst_5_zones_5_heaters',
        slp_experiment_params = get_planner_parameters(model_weight=5, learning_rate=0.02, batch_size=32, epochs=4_000),
        drp_experiment_params = get_planner_parameters(model_weight=5, learning_rate=0.001, batch_size=32, epochs=10_000, topology=[64, 64])
    ),
    domain_instance_experiment(
        'PowerGen', 'inst_5_gen',
        slp_experiment_params = get_planner_parameters(model_weight=10, learning_rate=0.05, batch_size=32, epochs=10_000),
        drp_experiment_params = get_planner_parameters(model_weight=10, learning_rate=0.0001, batch_size=32, epochs=12_000, topology=[256, 128])
    ),
    # domain_instance_experiment(
    #     'Reservoir', 'inst_10_reservoirs',
    #     slp_experiment_params = get_planner_parameters(model_weight=10, learning_rate=0.2, batch_size=32, epochs=1_000),
    #     drp_experiment_params = get_planner_parameters(model_weight=10, learning_rate=0.0002, batch_size=32, epochs=10_000, topology=[64, 32])
    # ),
    # domain_instance_experiment(
    #      'MarsRover', 'inst_6_rovers_7_minerals',
    #     slp_experiment_params = get_planner_parameters(model_weight=10, learning_rate=0.2, batch_size=32, epochs=1_000),
    #     drp_experiment_params = get_planner_parameters(model_weight=10, learning_rate=0.0002, batch_size=32, epochs=10_000, topology=[64, 32])
    # ),
    # domain_instance_experiment(
    #      'Wildfire', 'inst_5x5_grid',
    #     slp_experiment_params = get_planner_parameters(model_weight=10, learning_rate=0.2, batch_size=32, epochs=1_000),
    #     drp_experiment_params = get_planner_parameters(model_weight=10, learning_rate=0.0002, batch_size=32, epochs=10_000, topology=[64, 32])
    # ),
]