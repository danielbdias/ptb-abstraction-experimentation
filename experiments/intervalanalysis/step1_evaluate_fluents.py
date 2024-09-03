import csv
import os
import time

from _domains import domains

import pyRDDLGym
from pyRDDLGym.core.intervals import RDDLIntervalAnalysis

from pyRDDLGym import RDDLEnv

import numpy as np

root_folder = os.path.dirname(__file__)

def record_time(file_path: str, time: float):
    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Time'])
        writer.writerow([time])

def record_reward_bounds_header(file_path: str):
    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Domain', 'Fluent', 'Reward Lower', 'Reward Upper'])

def record_reward_bounds(file_path: str, domain_name: str, fluent_name: str, reward_lower: float, reward_upper: float):
    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([domain_name, fluent_name, reward_lower, reward_upper])

def build_fluent_values_to_freeze(ground_fluent : str, analysis : RDDLIntervalAnalysis):
    lifted_fluent, object_name = ground_fluent.split('___')

    initial_values = analysis._bound_initial_values()
    
    params = analysis.rddl.variable_params[lifted_fluent]
    shape = analysis.rddl.object_counts(params)

    fluent_values = {lifted_fluent: np.full(shape=shape, fill_value=np.nan)}

    # both bounds are equal, just grab the first one
    ground_fluent_initial_values = initial_values[lifted_fluent][0]

    object_index = analysis.rddl.object_to_index[object_name]

    fluent_values[lifted_fluent][object_index] = ground_fluent_initial_values[object_index]

    return fluent_values

def compute_action_bounds(environment):
    action_bounds = {}

    for action, prange in environment.model.action_ranges.items():
        lower, upper = environment._bounds[action]
        if prange == 'bool':
            lower = np.full(np.shape(lower), fill_value=0, dtype=int)
            upper = np.full(np.shape(upper), fill_value=1, dtype=int)
        action_bounds[action] = (lower, upper)

    return action_bounds

print('--------------------------------------------------------------------------------')
print('Experiment Part 1 - Analysis of Fluent Dynamics')
print('--------------------------------------------------------------------------------')
print()

# possible analysis - per grounded fluent, per lifted fluent
start_time = time.time()

#########################################################################################################
# This script will run interval propagation for each domain and instance, and record statistics
#########################################################################################################

for domain in domains:
    domain_path = f"{root_folder}/domains/{domain.name}"
    domain_file_path = f'{domain_path}/regular/domain.rddl'
    instance_file_path = f'{domain_path}/regular/{domain.instance}.rddl'
    
    output_file_random_policy=f"{root_folder}/_results/intervals_table_random_policy_{domain.name}.csv"
    output_file_analysis_time=f"{root_folder}/_results/execution_time_random_policy_{domain.name}.csv"

    batch_size = domain.experiment_params['batch_size_train']

    environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path, vectorized=True)

    record_reward_bounds_header(output_file_random_policy)

    # Random policy
    start_time_for_analysis = time.time()

    analysis = RDDLIntervalAnalysis(environment.model)
    action_bounds = compute_action_bounds(environment)

    for ground_fluent in domain.ground_fluents_to_freeze:
        # test of fluent bounds 
        fluent_values = build_fluent_values_to_freeze(ground_fluent, analysis)
        
        # evaluate lower and upper bounds on accumulated reward of random policy
        bounds = analysis.bound(action_bounds=action_bounds, per_epoch=True, 
                                fluent_values=fluent_values)
        reward_lower, reward_upper = bounds['reward'] 
        record_reward_bounds(output_file_random_policy, domain.name, ground_fluent, reward_lower, reward_upper)

    elapsed_time_for_analysis = time.time() - start_time_for_analysis

    record_time(output_file_analysis_time, elapsed_time_for_analysis)

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print()
print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()