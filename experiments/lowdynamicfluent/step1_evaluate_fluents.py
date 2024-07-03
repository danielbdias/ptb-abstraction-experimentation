import os
from dataclasses import dataclass
from typing import Dict, List

import time

import pyRDDLGym
from pyRDDLGym.core.policy import RandomAgent

from _domains import domains, DomainExperiment
from _utils import run_single_simulation

root_folder = os.path.dirname(__file__)

print('--------------------------------------------------------------------------------')
print('Experiment Part 1 - Analysis of Fluent Dynamics')
print('--------------------------------------------------------------------------------')
print()

# possible analysis - per grounded fluent, per lifted fluent

start_time = time.time()

def run_experiment(domain):
    domain_path = f"{root_folder}/domains/{domain.name}/regular"
    domain_file_path = f'{domain_path}/domain.rddl'
    instance_file_path = f'{domain_path}/{domain.instance}.rddl'

    return run_single_simulation(domain_file_path, instance_file_path)

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