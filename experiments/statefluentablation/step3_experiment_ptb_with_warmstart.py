import os
import jax

import time

import jax.nn.initializers as initializers

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan

from _domains import domains, jax_seeds, silent, experiment_params
from _utils import run_experiment, save_data, PlannerParameters

root_folder = os.path.dirname(__file__)

print('--------------------------------------------------------------------------------')
print('Experiment Part 3 - Running with Warm Start')
print('--------------------------------------------------------------------------------')
print()

start_time = time.time()

#########################################################################################################
# Runs with regular domain
#########################################################################################################

for domain in domains:
    print('--------------------------------------------------------------------------------')
    print('Domain: ', domain)
    print('--------------------------------------------------------------------------------')
    print()

    #########################################################################################################
    # Runs with regular domain (just to use as comparison)
    #########################################################################################################

    regular_environment = RDDLEnv.RDDLEnv(domain=f'{root_folder}/domains/{domain.name}/regular/domain.rddl', instance=f'{root_folder}/domains/{domain.name}/regular/{domain.instance}.rddl')
    regular_env_experiment_stats = []

    regular_experiment_name = f"{domain.name} (regular) - Straight line"

    for jax_seed in jax_seeds:
        experiment_params['plan'] = JaxStraightLinePlan()
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds
        experiment_params['policy_hyperparams'] = domain.policy_hyperparams

        env_params = PlannerParameters(**experiment_params)

        experiment_summary = run_experiment(regular_experiment_name, environment=regular_environment, planner_parameters=env_params, silent=silent)
        regular_env_experiment_stats.append(experiment_summary)

    save_data(regular_env_experiment_stats, f'{root_folder}/_results/{domain.name}_regular_statistics.pickle')

    #########################################################################################################
    # Runs with abstracted domain (Regular domain with ablated state variable set as initial state)
    #########################################################################################################

    abstraction_environment = RDDLEnv.RDDLEnv(domain=f'{root_folder}/domains/{domain.name}/abstraction/domain.rddl', instance=f'{root_folder}/domains/{domain.name}/abstraction/{domain.instance}.rddl')
    env_experiment_stats = []

    for jax_seed in jax_seeds:
        experiment_params['plan'] = JaxStraightLinePlan()
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds

        abstraction_env_params = PlannerParameters(**experiment_params)

        abstraction_env_experiment_summary = run_experiment(f"{domain.name} (abstraction) - Straight line", environment=abstraction_environment, planner_parameters=abstraction_env_params, silent=silent)

        initializers_per_action = {}
        for key in abstraction_env_experiment_summary.final_policy_weights.keys():
            initializers_per_action[key] = initializers.constant(abstraction_env_experiment_summary.final_policy_weights[key])

        experiment_params['plan'] = JaxStraightLinePlan(initializer_per_action=initializers_per_action)
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds

        warm_start_env_params = PlannerParameters(**experiment_params)

        warm_start_env_experiment_summary = run_experiment(f"{domain.name} (warm start) - Straight line", environment=regular_environment, planner_parameters=warm_start_env_params, silent=silent)
        env_experiment_stats.append(warm_start_env_experiment_summary)

    save_data(env_experiment_stats, f'{root_folder}/_results/{domain.name}_warmstart_statistics.pickle')

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()