import os

import time
import jax

import pyRDDLGym
from pyRDDLGym.core.grounder import RDDLGrounder

from pyRDDLGym_jax.core.planner import JaxStraightLinePlan

from _domains import domains, jax_seeds, silent
from _utils import run_experiment, save_data, PlannerParameters

root_folder = os.path.dirname(__file__)

print('--------------------------------------------------------------------------------')
print('Experiment Part 0 - Create baseline Run with random policy')
print('--------------------------------------------------------------------------------')
print()

start_time = time.time()

for domain in domains:
    print('--------------------------------------------------------------------------------')
    print('Domain: ', domain)
    print('--------------------------------------------------------------------------------')
    print()

    #########################################################################################################
    # Runs with regular domain (just to use as comparison)
    #########################################################################################################

    domain_path = f"{root_folder}/domains/{domain.name}"
    domain_file_path = f'{domain_path}/domain.rddl'
    instance_file_path = f'{domain_path}/{domain.instance}.rddl'

    regular_environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path)
    grounder = RDDLGrounder(regular_environment.model.ast)
    grounded_model = grounder.ground() # we need to run the base model on the same way as the other models

    regular_env_experiment_stats = []

    regular_experiment_name = f"{domain.name} (regular) - Straight line"

    for jax_seed in jax_seeds:
        experiment_params = domain.experiment_params.copy()
        experiment_params['plan'] = JaxStraightLinePlan()
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds
        experiment_params['policy_hyperparams'] = domain.policy_hyperparams
        experiment_params['ground_fluents_to_freeze'] = set()

        env_params = PlannerParameters(**experiment_params)

        experiment_summary = run_experiment(regular_experiment_name, rddl_model=grounded_model, planner_parameters=env_params, silent=silent)
        regular_env_experiment_stats.append(experiment_summary)

    save_data(regular_env_experiment_stats, f'{root_folder}/_results/baseline_run_data_{domain.name}_{domain.instance}.pickle')

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()