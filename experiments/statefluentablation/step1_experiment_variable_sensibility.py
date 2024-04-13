import os
import jax

import time

from pyRDDLGym import RDDLEnv
from pyRDDLGym.Core.Jax.JaxRDDLBackpropPlanner import JaxStraightLinePlan

from _domains import domains, jax_seeds, silent, experiment_params
from _utils import run_experiment, save_data, PlannerParameters

root_folder = os.path.dirname(__file__)

print('--------------------------------------------------------------------------------')
print('Experiment Part 1 - Sensibility Analysis')
print('--------------------------------------------------------------------------------')
print()

start_time = time.time()

def run_planner_experiment(domain, bound_name, experiment_name):
    domain_path = f"{root_folder}/domains/{domain.name}/abstraction-{state_fluent}-{bound_name}"

    environment = RDDLEnv.RDDLEnv(domain=f'{domain_path}/domain.rddl', instance=f'{domain_path}/{domain.instance}.rddl')
    experiment_stats = []

    for jax_seed in jax_seeds:
        experiment_params['plan'] = JaxStraightLinePlan()
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds
        experiment_params['policy_hyperparams'] = domain.policy_hyperparams

        env_params = PlannerParameters(**experiment_params)

        experiment_summary = run_experiment(experiment_name, environment=environment, planner_parameters=env_params, silent=silent)
        experiment_stats.append(experiment_summary)

    return experiment_stats

#########################################################################################################
# Runs with simplified domains
#########################################################################################################

print('--------------------------------------------------------------------------------')

for domain in domains:
    for state_fluent in domain.state_fluents:
        
        print(f'Domain: {domain.name} State Fluent: {state_fluent}')
        
        lower_bound_experiment_name = f"{domain.name} ({state_fluent} lower bound) - Straight line"

        lower_bound_experiment_stats = run_planner_experiment(domain, 'lower-bound', lower_bound_experiment_name)
        save_data(lower_bound_experiment_stats, f'{root_folder}/_results/{domain.name}_abstraction_{state_fluent}_lower_bound_statistics.pickle')

        upper_bound_experiment_name = f"{domain.name} ({state_fluent} upper bound) - Straight line"

        upper_bound_experiment_stats = run_planner_experiment(domain, 'upper-bound', upper_bound_experiment_name)
        save_data(upper_bound_experiment_stats, f'{root_folder}/_results/{domain.name}_abstraction_{state_fluent}_upper_bound_statistics.pickle')

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print()
print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()