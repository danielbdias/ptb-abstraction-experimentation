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
        writer.writerow([
            'Domain', 'Fluent', 
            # 'Reward Lower (h=0)', 'Reward Upper (h=0)', 'Reward Diff (h=0)', 'Reward Score (h=0)'
            # 'Reward Lower (h=halfway)', 'Reward Upper (h=halfway)', 'Reward Diff (h=halfway)', 'Reward Score (h=halfway)'
            'Reward Lower (h=end)', 'Reward Upper (h=end)', 'Reward Diff (h=end)', 'Reward Score (h=end)'
        ])

def compute_reward_values(index : int, frozen_reward_lower: float, frozen_reward_upper: float, unfrozen_reward_lower: float, unfrozen_reward_upper: float):
    frozen_reward_lower_for_index, frozen_reward_upper_for_index = frozen_reward_lower[index], frozen_reward_upper[index]
    frozen_reward_diff_for_index = frozen_reward_upper_for_index - frozen_reward_lower_for_index

    unfrozen_reward_lower_for_index, unfrozen_reward_upper_for_index = unfrozen_reward_lower[index], unfrozen_reward_upper[index]
    unfrozen_reward_diff_for_index = unfrozen_reward_upper_for_index - unfrozen_reward_lower_for_index

    score_for_index = frozen_reward_diff_for_index / unfrozen_reward_diff_for_index

    return frozen_reward_lower_for_index, frozen_reward_upper_for_index, frozen_reward_diff_for_index, score_for_index

def record_reward_bounds(file_path: str, domain_name: str, fluent_name: str, frozen_reward_lower: float, frozen_reward_upper: float, unfrozen_reward_lower: float, unfrozen_reward_upper: float):
    # frozen_reward_lower_begin, frozen_reward_upper_begin, reward_diff_begin, score_begin = compute_reward_values(0, frozen_reward_lower, frozen_reward_upper, unfrozen_reward_lower, unfrozen_reward_upper)

    # halfway_index = len(frozen_reward_lower) // 2
    # frozen_reward_lower_halfway, frozen_reward_upper_halfway, reward_diff_halfway, score_halfway = compute_reward_values(halfway_index, frozen_reward_lower, frozen_reward_upper, unfrozen_reward_lower, unfrozen_reward_upper)

    frozen_reward_lower_end, frozen_reward_upper_end, reward_diff_end, score_end = compute_reward_values(-1, frozen_reward_lower, frozen_reward_upper, unfrozen_reward_lower, unfrozen_reward_upper)

    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow([
            domain_name, fluent_name, 
            # frozen_reward_lower_begin, frozen_reward_upper_begin, reward_diff_begin, score_begin,
            # frozen_reward_lower_halfway, frozen_reward_upper_halfway, reward_diff_halfway, score_halfway,
            frozen_reward_lower_end, frozen_reward_upper_end, reward_diff_end, score_end,
        ])

def build_ground_fluent_list(environment : RDDLEnv):
    ground_fluents = []
    
    for lifted_fluent in environment.model.state_fluents:
        params = environment.model.variable_params[lifted_fluent]
        grounded_objects = environment.model.ground_types(params)

        for object_name in grounded_objects:
            if len(object_name) == 0:
                ground_fluents.append(lifted_fluent)
            else:    
                ground_fluents.append(f"{lifted_fluent}___{object_name[0]}")

    return ground_fluents

def build_fluent_values_to_freeze(ground_fluent : str, analysis : RDDLIntervalAnalysis):
    lifted_fluent = ground_fluent
    object_name = ''

    splitted_values = ground_fluent.split('___')

    if len(splitted_values) > 1:
        lifted_fluent, object_name = splitted_values[0], splitted_values[1]

    initial_values = analysis._bound_initial_values()

    # both bounds are equal, just grab the first one
    ground_fluent_initial_values = initial_values[lifted_fluent][0]

    if object_name == '': # that means that there is no object to this fluent, just return the initial value
        return {lifted_fluent: ground_fluent_initial_values}
    
    params = analysis.rddl.variable_params[lifted_fluent]
    shape = analysis.rddl.object_counts(params)
    object_index = analysis.rddl.object_to_index[object_name]

    fluent_values = {lifted_fluent: np.full(shape=shape, fill_value=np.nan)}
    fluent_values[lifted_fluent][object_index] = ground_fluent_initial_values[object_index]

    return fluent_values

def compute_action_bounds(domain, environment):
    if domain.action_bounds_for_interval_analysis is not None:
        return domain.action_bounds_for_interval_analysis

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
    action_bounds = compute_action_bounds(domain, environment)

    # run first without freezing any fluent
    bounds = analysis.bound(action_bounds=action_bounds, per_epoch=True)
    unfrozen_reward_lower, unfrozen_reward_upper = bounds['reward'] # reward per horizon
    record_reward_bounds(output_file_random_policy, domain.name, 'unfrozen', unfrozen_reward_lower, unfrozen_reward_upper, unfrozen_reward_lower, unfrozen_reward_upper)

    ground_fluents = build_ground_fluent_list(environment)

    for ground_fluent in ground_fluents:
        # test of fluent bounds 
        fluent_values = build_fluent_values_to_freeze(ground_fluent, analysis)
        
        # evaluate lower and upper bounds on accumulated reward of random policy
        bounds = analysis.bound(action_bounds=action_bounds, per_epoch=True, 
                                fluent_values=fluent_values)
        frozen_reward_lower, frozen_reward_upper = bounds['reward'] # reward per horizon
        record_reward_bounds(output_file_random_policy, domain.name, ground_fluent, frozen_reward_lower, frozen_reward_upper, unfrozen_reward_lower, unfrozen_reward_upper)

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