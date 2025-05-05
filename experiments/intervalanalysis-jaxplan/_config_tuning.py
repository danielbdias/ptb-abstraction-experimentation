from dataclasses import dataclass

#####################################################################################################################################
# This file provides configuration for the experiments directly in python variables,
# as a DSL (domain-specific language)
#####################################################################################################################################

tuning_seed = 42
eval_trials = 5
num_workers = 6
gp_iters = 10

silent = True
run_drp = True

@dataclass(frozen=True)
class DomainInstanceTuningData:
    domain_name: str
    instance_name: str
    drp_template_file: str
    
    def get_experiment_paths(self, root_folder):
        domain_path = f"{root_folder}/domains/{self.domain_name}"
        domain_file_path = f'{domain_path}/domain.rddl'
        instance_file_path = f'{domain_path}/{self.instance_name}.rddl'
        
        return domain_file_path, instance_file_path

def domain_instance_tuning_data(domain_name, instance_name):
    experiment = DomainInstanceTuningData(
        domain_name = domain_name, 
        instance_name = instance_name, 
        drp_template_file = '_template_tuning_drp.toml'
    )
    
    return experiment

experiments = [    
    # Continuous Domains
    # domain_instance_tuning_data('UAV_ippc2023', '3'),
    # domain_instance_tuning_data('MountainCar_ippc2023', '1'),
    # domain_instance_tuning_data('Reservoir_ippc2023', '3'),
    
    # Continuous and Discrete (Mixed) Domains
    domain_instance_tuning_data('MarsRover_ippc2023', '3'),
    domain_instance_tuning_data('HVAC_ippc2023', '3'),
    domain_instance_tuning_data('PowerGen_ippc2023', '3'),
    
    # Discrete Domains
    domain_instance_tuning_data('Wildfire_MDP_ippc2014', '5'),
    domain_instance_tuning_data('SysAdmin_POMDP_ippc2011', '2'),
    domain_instance_tuning_data('TriangleTireworld_MDP_ippc2014', '4'),
]