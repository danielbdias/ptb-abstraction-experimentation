from _experiment import DomainInstanceExperiment

#####################################################################################################################################
# This file provides configuration for the experiments directly in python variables,
# as a DSL (domain-specific language)
#####################################################################################################################################

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

solver_timeout = 600 # 10 minutes, GurobiPlan stops running after this time

silent = True

run_slp = True

def domain_instance_experiment(domain_name, instance_name):
    return DomainInstanceExperiment(
        domain_name = domain_name, 
        instance_name = instance_name, 
        ground_fluents_to_freeze = set(),
        bound_strategies = bound_strategies,
        solver_timeout = solver_timeout,
    )

experiments = [
    # domain_instance_experiment(
    #     'HVAC', 'inst_5_zones_5_heaters',
    # ),
    domain_instance_experiment(
        'PowerGen', 'inst_5_gen',
    ),
    domain_instance_experiment(
        'Reservoir', 'inst_10_reservoirs',
    ),
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