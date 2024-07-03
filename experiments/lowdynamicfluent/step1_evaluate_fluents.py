import os
import time

from _domains import domains
from _utils import run_simulations

root_folder = os.path.dirname(__file__)

def run_experiment(domain):
    domain_path = f"{root_folder}/domains/{domain.name}/regular"
    domain_file_path = f'{domain_path}/domain.rddl'
    instance_file_path = f'{domain_path}/{domain.instance}.rddl'

    return run_simulations(domain_file_path, instance_file_path, domain.state_fluents, 1)

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

for domain in domains:
    data = run_experiment(domain)
    print(data)

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print()
print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()