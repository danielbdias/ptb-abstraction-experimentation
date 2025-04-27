from dataclasses import dataclass

#####################################################################################################################################
# This file provides configuration for the experiments directly in python variables,
# as a DSL (domain-specific language)
#####################################################################################################################################

tuning_seed = 42
eval_trials = 5
num_workers = 6
gp_iters = 20

silent = True

run_drp = True

@dataclass(frozen=True)
class DomainInstanceTuningData:
    domain_name: str
    instance_name: str
    drp_template_file: str

def domain_instance_tuning_data(domain_name, instance_name):
    experiment = DomainInstanceTuningData(
        domain_name = domain_name, 
        instance_name = instance_name, 
        drp_template_file = '_template_tuning_drp.toml'
    )
    
    return experiment

experiments = [    
    # Continuous Domains
    domain_instance_tuning_data('UAV', 'instance4'),
    domain_instance_tuning_data('MountainCar', 'instance1'),
    
    # Continuous and Discrete (Mixed) Domains
    domain_instance_tuning_data('MarsRover', 'inst_6_rovers_7_minerals'),
    
    # Discrete Domains
    domain_instance_tuning_data('Wildfire', 'inst_5x5_grid'),
    domain_instance_tuning_data('SysAdmin', 'instance2'),
    domain_instance_tuning_data('TriangleTireworld', 'instance4'),
]