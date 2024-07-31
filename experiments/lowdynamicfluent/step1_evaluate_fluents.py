import os
import time

from _domains import domains, experiment_params, bins
from _utils import run_simulations, compute_statistics, record_csv

import pyRDDLGym

root_folder = os.path.dirname(__file__)
batch_size = experiment_params['batch_size_train']

print('--------------------------------------------------------------------------------')
print('Experiment Part 1 - Analysis of Fluent Dynamics')
print('--------------------------------------------------------------------------------')
print()

# possible analysis - per grounded fluent, per lifted fluent
start_time = time.time()

#########################################################################################################
# This script will run simulations for each domain and instance, and record statistics
#########################################################################################################

for domain in domains:
    domain_path = f"{root_folder}/domains/{domain.name}"
    domain_file_path = f'{domain_path}/regular/domain.rddl'
    instance_file_path = f'{domain_path}/regular/{domain.instance}.rddl'
    output_file_random_policy=f"{root_folder}/_results/statistics_table_random_policy_{domain.name}.csv"

    environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path)

    # Random policy
    simulations = run_simulations(environment, domain.state_fluents, batch_size)
    statistics = compute_statistics(simulations, bins)
    record_csv(output_file_random_policy, domain.name, statistics)

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print()
print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()