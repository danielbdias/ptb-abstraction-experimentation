import os

import time
import jax

import jax.nn.initializers as initializers

import pyRDDLGym

from pyRDDLGym_jax.core.planner import JaxStraightLinePlan

from _domains import domains, jax_seeds, silent, experiment_params
from _utils import run_experiment, save_data, load_data, PlannerParameters

root_folder = os.path.dirname(__file__)

print('--------------------------------------------------------------------------------')
print('Experiment Part 3 - Create Warm Start policies and Run with Warm Start')
print('--------------------------------------------------------------------------------')
print()

start_time = time.time()

for domain in domains:
    print('--------------------------------------------------------------------------------')
    print('Domain: ', domain)
    print('--------------------------------------------------------------------------------')
    print()

    #########################################################################################################
    # Runs PtB with modified domain (that has ground fluents frozen with initial state values)
    #########################################################################################################

    domain_path = f"{root_folder}/domains/{domain.name}"
    
    regular_domain_file_path = f'{domain_path}/regular/domain.rddl'
    regular_instance_file_path = f'{domain_path}/regular/{domain.instance}.rddl'

    regular_environment = pyRDDLGym.make(domain=regular_domain_file_path, instance=regular_instance_file_path)
    
    ablated_model_file_path = f'{domain_path}/ablated/domain_{domain.instance}.model'
    grounded_model = load_data(ablated_model_file_path)

    warm_start_creation_experiment_stats = []
    warm_start_run_experiment_stats = []

    for jax_seed in jax_seeds:
        experiment_params['plan'] = JaxStraightLinePlan()
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds
        experiment_params['policy_hyperparams'] = domain.policy_hyperparams
        experiment_params['ground_fluents_to_freeze'] = domain.ground_fluents_to_freeze

        abstraction_env_params = PlannerParameters(**experiment_params)

        abstraction_env_experiment_summary = run_experiment(f"{domain.name} (abstraction) - Straight line", rddl_model=grounded_model, planner_parameters=abstraction_env_params, silent=silent)
        warm_start_creation_experiment_stats.append(abstraction_env_experiment_summary)
        
        initializers_per_action = {}
        for key in abstraction_env_experiment_summary.final_policy_weights.keys():
            initializers_per_action[key] = initializers.constant(abstraction_env_experiment_summary.final_policy_weights[key])

        experiment_params['plan'] = JaxStraightLinePlan(initializer_per_action=initializers_per_action)
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds

        warm_start_env_params = PlannerParameters(**experiment_params)

        warm_start_env_experiment_summary = run_experiment(f"{domain.name} (warm start) - Straight line", rddl_model=regular_environment.model, planner_parameters=warm_start_env_params, silent=silent)
        warm_start_run_experiment_stats.append(warm_start_env_experiment_summary)

    save_data(warm_start_creation_experiment_stats, f'{root_folder}/_results/warmstart_creation_run_data_{domain.name}.pickle')
    save_data(warm_start_run_experiment_stats, f'{root_folder}/_results/warmstart_execution_run_data_{domain.name}.pickle')

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()