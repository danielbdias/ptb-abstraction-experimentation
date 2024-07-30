import os

import time
import jax

import jax.nn.initializers as initializers

import pyRDDLGym
from pyRDDLGym.core.grounder import RDDLGrounder
from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.parser.expr import Expression

from pyRDDLGym_jax.core.planner import JaxStraightLinePlan

from _domains import domains, jax_seeds, silent, experiment_params
from _utils import run_experiment, save_data, PlannerParameters

def get_grounded_model_with_frozen_fluent(environment, domain):
    grounder = RDDLGrounder(environment.model.ast)
    grounded_model = grounder.ground()

    for fluent_to_freeze in domain.ground_fluents_to_freeze:
        fluent_to_freeze_prime = f"{fluent_to_freeze}{RDDLPlanningModel.NEXT_STATE_SYM}"
        has_cpf = fluent_to_freeze_prime in grounded_model.cpfs

        if has_cpf:
            first, second = grounded_model.cpfs[fluent_to_freeze_prime]
            # force CPF to be " ground_fluent' = ground_fluent "
            grounded_model.cpfs[fluent_to_freeze_prime] = (first, Expression( ('pvar_expr', (fluent_to_freeze, None)) ))
    
    return grounded_model

root_folder = os.path.dirname(__file__)

print('--------------------------------------------------------------------------------')
print('Experiment Part 2 - Create Warm Start policies and Run with Warm Start')
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
    grounded_model = get_grounded_model_with_frozen_fluent(regular_environment, domain)

    env_experiment_stats = []

    for jax_seed in jax_seeds:
        experiment_params['plan'] = JaxStraightLinePlan()
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds
        experiment_params['policy_hyperparams'] = domain.policy_hyperparams
        experiment_params['ground_fluents_to_freeze'] = domain.ground_fluents_to_freeze

        abstraction_env_params = PlannerParameters(**experiment_params)

        abstraction_env_experiment_summary = run_experiment(f"{domain.name} (abstraction) - Straight line", rddl_model=grounded_model, planner_parameters=abstraction_env_params, silent=silent)
        
        initializers_per_action = {}
        for key in abstraction_env_experiment_summary.final_policy_weights.keys():
            initializers_per_action[key] = initializers.constant(abstraction_env_experiment_summary.final_policy_weights[key])

        experiment_params['plan'] = JaxStraightLinePlan(initializer_per_action=initializers_per_action)
        experiment_params['seed'] = jax.random.PRNGKey(jax_seed)
        experiment_params['action_bounds'] = domain.action_bounds

        warm_start_env_params = PlannerParameters(**experiment_params)

        warm_start_env_experiment_summary = run_experiment(f"{domain.name} (warm start) - Straight line", rddl_model=regular_environment.model, planner_parameters=warm_start_env_params, silent=silent)
        env_experiment_stats.append(warm_start_env_experiment_summary)

    save_data(env_experiment_stats, f'{root_folder}/_results/{domain.name}_warmstart_statistics.pickle')

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()