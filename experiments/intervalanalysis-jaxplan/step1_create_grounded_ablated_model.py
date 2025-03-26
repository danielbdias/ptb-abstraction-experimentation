import os
import time

from typing import Set

import pyRDDLGym
from pyRDDLGym import RDDLEnv
from pyRDDLGym.core.debug.decompiler import RDDLDecompiler
from pyRDDLGym.core.grounder import RDDLGrounder
from pyRDDLGym.core.compiler.model import RDDLPlanningModel
from pyRDDLGym.core.parser.expr import Expression

from _config import experiments, threshold_to_choose_fluents
from _experiment import run_experiment_in_parallel, prepare_parallel_experiment_on_main

from _fileio import file_exists, get_ground_fluents_to_ablate_from_csv, save_pickle_data, save_raw_data

def get_ground_fluents_to_ablate(domain, file_path: str):
    if domain.ground_fluents_to_freeze is not None and len(domain.ground_fluents_to_freeze) < 0:
        return domain.ground_fluents_to_freeze

    return get_ground_fluents_to_ablate_from_csv(file_path)

def get_grounded_model_with_frozen_fluent(environment: RDDLEnv, ground_fluents_to_freeze: Set[str]):
    grounder = RDDLGrounder(environment.model.ast)
    grounded_model = grounder.ground()

    for fluent_to_freeze in ground_fluents_to_freeze:
        fluent_to_freeze_prime = f"{fluent_to_freeze}{RDDLPlanningModel.NEXT_STATE_SYM}"
        has_cpf = fluent_to_freeze_prime in grounded_model.cpfs

        # TODO: (important!) instead of freezing the CPF, we should replace the fluent by its initial state value

        if has_cpf:
            first, _ = grounded_model.cpfs[fluent_to_freeze_prime]
            # force CPF to be " ground_fluent' = ground_fluent "
            grounded_model.cpfs[fluent_to_freeze_prime] = (first, Expression( ('pvar_expr', (fluent_to_freeze, None)) ))

    return grounded_model

def write_grounded_model_to_file(grounded_model, domain_file_path, grounded_model_file_path):
    decompiler = RDDLDecompiler()
    grounded_domain_file_content = decompiler.decompile_domain(grounded_model)

    save_raw_data(grounded_domain_file_content, domain_file_path)    
    save_pickle_data(grounded_model, grounded_model_file_path)

root_folder = os.path.dirname(__file__)

def perform_experiment(domain_instance_experiment, strategy_name, threshold):
    print(f'[{os.getpid()}] Domain: {domain_instance_experiment.domain_name} - Instance: {domain_instance_experiment.instance_name} - Ablation Metric: {strategy_name} - Threshold: {threshold}')
    
    _, domain_file_path, instance_file_path = domain_instance_experiment.get_experiment_paths(root_folder)

    file_common_suffix = f'{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}_{strategy_name}_{threshold}'
    fluents_to_freeze_path = f"{root_folder}/_results/fluents_to_ablate_{file_common_suffix}.csv"
    if not file_exists(fluents_to_freeze_path):
        print(f'File for domain {domain_instance_experiment.domain_name} considering {strategy_name} strategy at threshold {threshold} not found. This means that it was not possible to get valid intervals on interval analysis. Skipping experiment')
        return

    regular_environment = pyRDDLGym.make(domain=domain_file_path, instance=instance_file_path)

    fluents_to_ablate = get_ground_fluents_to_ablate(domain_instance_experiment, fluents_to_freeze_path)
    grounded_model = get_grounded_model_with_frozen_fluent(regular_environment, fluents_to_ablate)

    grounded_domain_file_path = f'{root_folder}/_intermediate/domain_{file_common_suffix}.rddl'
    grounded_model_file_path = f'{root_folder}/_intermediate/domain_{file_common_suffix}.model'
    
    decompiler = RDDLDecompiler()
    grounded_domain_file_content = decompiler.decompile_domain(grounded_model)

    save_raw_data(grounded_domain_file_content, grounded_domain_file_path)    
    save_pickle_data(grounded_model, grounded_model_file_path)

if __name__ == '__main__':
    prepare_parallel_experiment_on_main()

    print('--------------------------------------------------------------------------------')
    print('Abstraction Experiment - Create Ground RDDL model with ablated fluents')
    print('--------------------------------------------------------------------------------')
    print()

    #########################################################################################################
    # Prepare to run in multiple processes
    #########################################################################################################

    start_time = time.time()

    # create combination of parameters that we will use to create ground models
    args_list = []
    
    for domain_instance_experiment in experiments:
        for strategy_name in domain_instance_experiment.bound_strategies.keys():
            for threshold in threshold_to_choose_fluents:
                args_list.append( (domain_instance_experiment, strategy_name, threshold) )

    # Run experiments in parallel
    run_experiment_in_parallel(perform_experiment, args_list)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print()
    print('--------------------------------------------------------------------------------')
    print('Elapsed Time: ', elapsed_time)
    print('--------------------------------------------------------------------------------')
    print()