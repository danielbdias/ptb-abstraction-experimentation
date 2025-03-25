import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import time

from _config import experiments, threshold_to_choose_fluents
from _experiment import run_experiment_in_parallel, prepare_parallel_experiment_on_main
from _fileio import load_pickle_data, file_exists

def load_time(file_path: str):
    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')

        lines = []

        for row in reader:
            lines.append(row[0])

        return float(lines[1])

def plot_convergence_time(plot_path, domain_instance_experiment, evaluation_time, warm_start_computation, warm_start_execution, baseline_execution):
    methods = (
        "Random Policy",
        "Bounded Actions"
    )
    metrics = {
        "Interval Analysis": np.array([0, evaluation_time]),
        "Bounded Action Computation": np.array([0, warm_start_computation]),
        "GurobiPlan Execution": np.array([baseline_execution, warm_start_execution]),
    }
    width = 0.5

    y_axis_height = np.max([baseline_execution, evaluation_time + warm_start_computation + warm_start_execution]) * 1.6

    _, ax = plt.subplots()
    bottom = np.zeros(2)

    for metric_name, metric_value in metrics.items():
        bars = ax.bar(methods, metric_value, width, label=metric_name, bottom=bottom)
        bottom += metric_value
        
        for bar in bars:
            x_value = bar.get_x() + bar.get_width() / 2
            y_value = bar.get_height() + bar.get_y()
            
            bar_height_as_string = f"{bar.get_height():.2f}"
            if bar_height_as_string != "0.00":
                plt.text(x_value, y_value, bar_height_as_string, ha='center', va='bottom')

    ax.set_title(f"Convergence time\n({domain_instance_experiment.domain_name} - {domain_instance_experiment.instance_name})")
    ax.legend(loc="upper right", fontsize=14)

    plt.rcParams.update({'font.size':15})
    plt.rc('font', family='serif')

    plt.ylabel("Time (seconds)")
    plt.ylim(0, y_axis_height)

    plt.savefig(plot_path, format='pdf')

def read_fluent_evaluation_time_csv(file_path: str):
    lines = []

    with open(file_path, 'r') as csvfile:
        reader = csv.reader(csvfile, delimiter=';', quotechar='|')

        for row in reader:
            lines.append(row[0])

    return float(lines[1])
    
def plot_final_reward_per_threshold(plot_path, domain_instance_experiment, random_policy_stats, warm_start_stats_with_thresholds):
    methods = [ 'Random Policy' ]
    rewards = [ random_policy_stats.accumulated_reward ]

    for threshold in warm_start_stats_with_thresholds.keys():
        warm_start_stats = warm_start_stats_with_thresholds[threshold]
        methods.append(f'Warm Start ({threshold * 100}%)')
        rewards.append(warm_start_stats.accumulated_reward)

    plt.bar(methods, rewards)
    
    plt.xticks(rotation=90)
    plt.ylabel('Reward')
    plt.title(f'Accumulated Reward\n({domain_instance_experiment.domain_name} - {domain_instance_experiment.instance_name})')
    plt.tight_layout()

    plt.savefig(plot_path, format='pdf')


###############################

root_folder = os.path.dirname(__file__)
plot_folder = f'{root_folder}/_plots'

def plot_experiments(domain_instance_experiment, strategy_name):    
    print(f'[{os.getpid()}] Domain: {domain_instance_experiment.domain_name} - Instance: {domain_instance_experiment.instance_name} - Ablation Metric: {strategy_name}')
    
    file_common_suffix = f'{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}_{strategy_name}'

    baseline_execution_stats = load_pickle_data(f'{root_folder}/_results/baseline_run_data_{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}.pickle')

    warm_start_stats_with_thresholds = {}

    for threshold in threshold_to_choose_fluents:
        warm_start_execution_experiment_stats_file_path = f'{root_folder}/_results/warmstart_execution_run_data_{file_common_suffix}_{threshold}.pickle'

        if not file_exists(warm_start_execution_experiment_stats_file_path):
            print(f'File for domain {domain_instance_experiment.domain_name} considering {strategy_name} strategy at threshold {threshold} not found. This means that it was not possible to get valid intervals on interval analysis. Skipping experiment')
            return
        
        warm_start_execution_experiment_stats = load_pickle_data(warm_start_execution_experiment_stats_file_path)
        warm_start_stats_with_thresholds[threshold] = warm_start_execution_experiment_stats
    
    ############################################################
    # Final reward per threshold
    ############################################################

    plot_cost_curve_per_iteration_path = f'{plot_folder}/final_reward_per_threshold_{file_common_suffix}.pdf'
    plot_final_reward_per_threshold(plot_cost_curve_per_iteration_path, domain_instance_experiment, baseline_execution_stats, warm_start_stats_with_thresholds)

    ############################################################
    # Solution time
    ############################################################

    evaluation_time = read_fluent_evaluation_time_csv(f'{root_folder}/_results/time_{file_common_suffix}.csv')

    for threshold in threshold_to_choose_fluents:
        bouded_actions_computation_time = load_time(f'{root_folder}/_results/time_final_analysis_{file_common_suffix}_{threshold}.csv')
        
        warm_start_execution = warm_start_execution_experiment_stats.elapsed_time
        baseline_execution = baseline_execution_stats.elapsed_time

        plot_convergence_time_path = f'{plot_folder}/convergence_time_{file_common_suffix}_{threshold}.pdf'
        plot_convergence_time(plot_convergence_time_path, domain_instance_experiment, evaluation_time, bouded_actions_computation_time, warm_start_execution, baseline_execution)


if __name__ == '__main__':
    prepare_parallel_experiment_on_main()
    
    start_time = time.time()
    
    print('--------------------------------------------------------------------------------')
    print('Abstraction Experiment - Generating graphs for PtB with warm start')
    print('--------------------------------------------------------------------------------')
    print()

    # create combination of parameters that we will use to create single plots
    args_list_plots = []
    
    for domain_instance_experiment in experiments:
        for strategy_name in domain_instance_experiment.bound_strategies.keys():
            args_list_plots.append( (domain_instance_experiment, strategy_name,) )

    # run plot generation in parallel
    run_experiment_in_parallel(plot_experiments, args_list_plots)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print()
    print('--------------------------------------------------------------------------------')
    print('Elapsed Time: ', elapsed_time)
    print('--------------------------------------------------------------------------------')
    print()