import os
import jax
import time

from _domains import domains, experiment_params, jax_seeds, silent
from _utils import run_simulations, compute_statistics, record_csv, run_jaxplanner, compute_jax_simulation_statistics, PlannerParameters

import pyRDDLGym
from pyRDDLGym_jax.core.planner import JaxStraightLinePlan

root_folder = os.path.dirname(__file__)

print('--------------------------------------------------------------------------------')
print('Experiment Part 1 - Analysis of Fluent Dynamics')
print('--------------------------------------------------------------------------------')
print()

# possible analysis - per grounded fluent, per lifted fluent

start_time = time.time()

#########################################################################################################
# Runs with simplified domains
#########################################################################################################

print('--------------------------------------------------------------------------------')

batch_size = experiment_params['batch_size_train']
bins = 100

for domain in domains:
    domain_path = f"{root_folder}/domains/{domain.name}/regular"
    domain_file_path = f'{domain_path}/domain.rddl'
    instance_file_path = f'{domain_path}/{domain.instance}.rddl'
    output_file_random_policy=f"{root_folder}/_results/{domain.name}_stats_random_policy.csv"
    output_file_jax_plan=f"{root_folder}/_results/{domain.name}_stats_jax_plan.csv"

    environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path)

    # Random policy
    simulations = run_simulations(environment, domain.state_fluents, batch_size)
    statistics = compute_statistics(simulations, bins)
    record_csv(output_file_random_policy, domain.name, statistics)

    # JaxPlan
    jax_simulations_per_seed = []
    for jax_seed in jax_seeds:
        experiment_params['plan'] = JaxStraightLinePlan()
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds
        experiment_params['policy_hyperparams'] = domain.policy_hyperparams

        env_params = PlannerParameters(**experiment_params)

        simulations, _ = run_jaxplanner(domain.name, environment=environment, planner_parameters=env_params, silent=silent)
        jax_simulations_per_seed.append(simulations)

    statistics = compute_jax_simulation_statistics(jax_simulations_per_seed, domain.state_fluents, bins)
    record_csv(output_file_jax_plan, domain.name, statistics)

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print()
print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()