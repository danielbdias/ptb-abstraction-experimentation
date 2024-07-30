import os

import time
import jax

import pyRDDLGym
from pyRDDLGym.core.grounder import RDDLGrounder
from pyRDDLGym_jax.core.planner import JaxStraightLinePlan

from _domains import domains, jax_seeds, silent, experiment_params
from _utils import run_experiment, save_data, PlannerParameters

root_folder = os.path.dirname(__file__)

print('--------------------------------------------------------------------------------')
print('Experiment Part 2 - Create Warm Start policies')
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

    regular_environment = pyRDDLGym.make(domain=f'{root_folder}/domains/{domain.name}/domain.rddl', instance=f'{root_folder}/domains/{domain.name}/{domain.instance}.rddl')
    
    grounder = RDDLGrounder(regular_environment.model.ast)
    grounded_model = grounder.ground()

    for jax_seed in jax_seeds:
        experiment_params['plan'] = JaxStraightLinePlan()
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds
        experiment_params['policy_hyperparams'] = domain.policy_hyperparams
        experiment_params['ground_fluents_to_freeze'] = domain.ground_fluents_to_freeze

        abstraction_env_params = PlannerParameters(**experiment_params)

        abstraction_env_experiment_summary = run_experiment(f"{domain.name} (abstraction) - Straight line", rddl_model=grounded_model, planner_parameters=abstraction_env_params, silent=silent)

    # save_data(env_experiment_stats, f'{root_folder}/_results/{domain.name}_warmstart_statistics.pickle')

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()