import csv
import os
import time

from _domains import domains, bins
from _utils import run_simulations, compute_statistics, record_csv

import pyRDDLGym

root_folder = os.path.dirname(__file__)

def record_time(file_path: str, time: float):
    with open(file_path, 'a') as csvfile:
        writer = csv.writer(csvfile, delimiter=';', quotechar='|', quoting=csv.QUOTE_MINIMAL)
        writer.writerow(['Time'])
        writer.writerow([time])

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
    output_file_simulation_time=f"{root_folder}/_results/execution_time_random_policy_{domain.name}.csv"

    batch_size = domain.experiment_params['batch_size_train']

    environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path)

    # Random policy
    start_time_for_simulations = time.time()
    simulations = run_simulations(environment, domain.state_fluents, batch_size)
    statistics = compute_statistics(simulations, bins)
    elapsed_time_for_simulations = time.time() - start_time_for_simulations

    record_csv(output_file_random_policy, domain.name, statistics)
    record_time(output_file_simulation_time, elapsed_time_for_simulations)

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print()
print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()