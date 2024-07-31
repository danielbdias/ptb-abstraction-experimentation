import os

import time

import pyRDDLGym
from pyRDDLGym import RDDLEnv
from pyRDDLGym.core.debug.decompiler import RDDLDecompiler
from pyRDDLGym.core.grounder import RDDLGrounder
from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.parser.expr import Expression

from _domains import domains, DomainExperiment
from _utils import save_data

def get_grounded_model_with_frozen_fluent(environment: RDDLEnv, domain: DomainExperiment):
    grounder = RDDLGrounder(environment.model.ast)
    grounded_model = grounder.ground()

    for fluent_to_freeze in domain.ground_fluents_to_freeze:
        fluent_to_freeze_prime = f"{fluent_to_freeze}{RDDLPlanningModel.NEXT_STATE_SYM}"
        has_cpf = fluent_to_freeze_prime in grounded_model.cpfs

        if has_cpf:
            first, _ = grounded_model.cpfs[fluent_to_freeze_prime]
            # force CPF to be " ground_fluent' = ground_fluent "
            grounded_model.cpfs[fluent_to_freeze_prime] = (first, Expression( ('pvar_expr', (fluent_to_freeze, None)) ))

    return grounded_model

def write_grounded_model_to_file(grounded_model, domain_file_file_path, grounded_model_file_path):
    decompiler = RDDLDecompiler()
    grounded_domain_file_content = decompiler.decompile_domain(grounded_model)

    with open(domain_file_file_path, 'w') as file:
        file.write(grounded_domain_file_content)
    
    save_data(grounded_model, grounded_model_file_path)

root_folder = os.path.dirname(__file__)

print('--------------------------------------------------------------------------------')
print('Experiment Part 2 - Create Ground RDDL model with ablated fluents')
print('--------------------------------------------------------------------------------')
print()

start_time = time.time()

for domain in domains:
    print('--------------------------------------------------------------------------------')
    print('Domain: ', domain)
    print('--------------------------------------------------------------------------------')
    print()

    domain_path = f"{root_folder}/domains/{domain.name}"
    domain_file_path = f'{domain_path}/regular/domain.rddl'
    instance_file_path = f'{domain_path}/regular/{domain.instance}.rddl'

    regular_environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path)

    grounded_model = get_grounded_model_with_frozen_fluent(regular_environment, domain)
  
    grounded_domain_file_path = f'{domain_path}/ablated/domain_{domain.instance}.rddl'
    grounded_model_file_path = f'{domain_path}/ablated/domain_{domain.instance}.model'
    write_grounded_model_to_file(grounded_model, grounded_domain_file_path, grounded_model_file_path)

end_time = time.time()
elapsed_time = end_time - start_time

print('--------------------------------------------------------------------------------')
print('Elapsed Time: ', elapsed_time)
print('--------------------------------------------------------------------------------')
print()