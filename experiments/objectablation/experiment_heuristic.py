import os
import jax

import time

import jax.nn.initializers as initializers

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan

from _domains import domains, jax_seeds, silent, experiment_params
from _utils import run_experiment, save_data, PlannerParameters

root_folder = os.path.dirname(__file__)

start_time = time.time()

#########################################################################################################
# Runs with regular domain
#########################################################################################################

for domain in domains:
    print('--------------------------------------------------------------------------------')
    print('Domain: ', domain)
    print('--------------------------------------------------------------------------------')
    print()

    regular_environment = RDDLEnv.RDDLEnv(domain=f'{root_folder}/domains/{domain.name}/regular/domain.rddl', instance=f'{root_folder}/domains/{domain.name}/regular/{domain.instance}.rddl')

    #########################################################################################################
    # Runs with heuristic-ready domain
    #########################################################################################################

    heuristic_environment = RDDLEnv.RDDLEnv(domain=f'{root_folder}/domains/{domain.name}/heuristic/domain.rddl', instance=f'{root_folder}/domains/{domain.name}/heuristic/{domain.instance}.rddl')
    env_experiment_stats = []

    for jax_seed in jax_seeds:
        experiment_params['plan'] = JaxStraightLinePlan()
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds

        heuristic_env_params = PlannerParameters(**experiment_params)

        heuristic_env_experiment_summary = run_experiment(f"{domain.name} (heuristic) - Straight line", environment=heuristic_environment, planner_parameters=heuristic_env_params, silent=silent)

        initializers_per_action = {}
        for key in heuristic_env_experiment_summary.final_policy_weights.keys():
            initializers_per_action[key] = initializers.constant(heuristic_env_experiment_summary.final_policy_weights[key])

        experiment_params['plan'] = JaxStraightLinePlan(initializer_per_action=initializers_per_action)
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds

        initialized_env_params = PlannerParameters(**experiment_params)

        initialized_env_experiment_summary = run_experiment(f"{domain.name} (initialized) - Straight line", environment=regular_environment, planner_parameters=initialized_env_params, silent=silent)
        env_experiment_stats.append(initialized_env_experiment_summary)

    save_data(env_experiment_stats, f'{root_folder}/_results/{domain.name}_heuristic_statistics.pickle')

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()