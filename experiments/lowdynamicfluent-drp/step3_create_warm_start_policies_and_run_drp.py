import os

import time
import jax

import haiku as hk

import pyRDDLGym
from pyRDDLGym.core.grounder import RDDLGrounder

from pyRDDLGym_jax.core.planner import JaxDeepReactivePolicy

from _domains import domains, jax_seeds, silent
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
    grounder = RDDLGrounder(regular_environment.model.ast)
    regular_grounded_model = grounder.ground()
    
    ablated_model_file_path = f'{domain_path}/ablated/domain_{domain.instance}.model'
    ablated_grounded_model = load_data(ablated_model_file_path)

    warm_start_creation_experiment_stats = []
    warm_start_run_experiment_stats = []

    for jax_seed in jax_seeds:
        experiment_params = domain.experiment_params.copy()
        experiment_params['plan'] = JaxDeepReactivePolicy(domain.metadata['topology'])
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds
        experiment_params['policy_hyperparams'] = domain.policy_hyperparams
        experiment_params['ground_fluents_to_freeze'] = domain.ground_fluents_to_freeze

        abstraction_env_params = PlannerParameters(**experiment_params)

        abstraction_env_experiment_summary = run_experiment(f"{domain.name} (abstraction) - Straight line", rddl_model=ablated_grounded_model, planner_parameters=abstraction_env_params, silent=silent)
        warm_start_creation_experiment_stats.append(abstraction_env_experiment_summary)
        
        initializers_per_layer = {}
        for layer_name in abstraction_env_experiment_summary.final_policy_weights.keys():
            # print(layer_name)
            # print(abstraction_env_experiment_summary.final_policy_weights[layer_name])
            # initializers_per_layer[layer_name] = hk.initializers.Constant(abstraction_env_experiment_summary.final_policy_weights[layer_name]['w'])
            initializers_per_layer[layer_name] = abstraction_env_experiment_summary.final_policy_weights[layer_name]

        experiment_params = domain.experiment_params.copy()
        experiment_params['plan'] = JaxDeepReactivePolicy(domain.metadata['topology'], initializer_per_layer=initializers_per_layer)
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds
        experiment_params['policy_hyperparams'] = domain.policy_hyperparams
        experiment_params['ground_fluents_to_freeze'] = set()

        warm_start_env_params = PlannerParameters(**experiment_params)

        warm_start_env_experiment_summary = run_experiment(f"{domain.name} (warm start) - Straight line", rddl_model=regular_grounded_model, planner_parameters=warm_start_env_params, silent=silent)
        warm_start_run_experiment_stats.append(warm_start_env_experiment_summary)

    save_data(warm_start_creation_experiment_stats, f'{root_folder}/_results/warmstart_creation_run_data_{domain.name}.pickle')
    save_data(warm_start_run_experiment_stats, f'{root_folder}/_results/warmstart_execution_run_data_{domain.name}.pickle')

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()