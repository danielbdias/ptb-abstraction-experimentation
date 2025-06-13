import os
import csv
import numpy as np
import matplotlib.pyplot as plt
import time

from _config_run import get_experiments, run_drp, run_slp, threshold_to_choose_fluents, jax_seeds
from _experiment import run_experiment_in_parallel, prepare_parallel_experiment_on_main
from _fileio import load_pickle_data, file_exists

def plot_convergence_time(plot_path, domain_instance_experiment, planner_type, evaluation_time, warm_start_computation, warm_start_execution, baseline_execution):
    methods = (
        "Random Policy",
        "Warm Start Policy"
    )
    metrics = {
        "Interval Analysis": np.array([0, evaluation_time]),
        "Warm Start Computation": np.array([0, warm_start_computation]),
        "JaxPlan Execution": np.array([baseline_execution, warm_start_execution]),
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

    ax.set_title(f"Convergence time\n({domain_instance_experiment.domain_name} - {domain_instance_experiment.instance_name} - {planner_type})")
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

def get_curves(experiment_summaries, attribute_getter):
    curves = []

    for experiment_summary in experiment_summaries:
        curve = np.array(list(map(attribute_getter, experiment_summary.statistics_history)))
        curves.append(curve)

    return curves

def stat_curves(experiment_summaries, attribute_getter):
    iteration_curves = get_curves(experiment_summaries, lambda item : item.iteration)
    curves = get_curves(experiment_summaries, attribute_getter)

    iteration_curve_max_len = -1
    larger_iteration_curve = None

    # find experiment with more iterations
    for i in range(len(experiment_summaries)):
        iteration_curve = iteration_curves[i]

        if len(iteration_curve) > iteration_curve_max_len:
            iteration_curve_max_len = len(iteration_curve)
            larger_iteration_curve = iteration_curve

    # repeat last value for each curve with less iterations
    resized_curves = []

    for i in range(len(experiment_summaries)):
        curve = curves[i]
        size_diff = iteration_curve_max_len - len(curve)
        if size_diff <= 0:
            resized_curves.append(curve)
        else:
            curve_last_value = curve[-1]
            resized_curve = np.append(curve, np.repeat(curve_last_value, size_diff))
            resized_curves.append(resized_curve)

    # convert "list of np.array" to "np.array of np.array"
    resized_curves = np.stack(resized_curves)

    curves_mean = np.mean(resized_curves, axis=0)
    curves_stddev = np.std(resized_curves, axis=0)

    return larger_iteration_curve, curves_mean, curves_stddev
    
def plot_cost_curve_per_iteration(plot_path, domain_instance_experiment, planner_type, random_policy_stats, warm_start_stats, iter_cutting_point):
    plt.subplots(1, figsize=(8,5))

    statistics = {
        'Random Policy': random_policy_stats,
        'Warm Start': warm_start_stats
    }

    for key in statistics.keys():
        stats = statistics[key]

        iterations, best_return_curves_mean, best_return_curves_stddev = stat_curves(stats, lambda item : item.best_return)
        
        plt.plot(iterations, best_return_curves_mean, '-', label=key)
        plt.fill_between(iterations, (best_return_curves_mean - best_return_curves_stddev), (best_return_curves_mean + best_return_curves_stddev), alpha=0.2)

    plt.title(f'Best Reward per Iteration\n({domain_instance_experiment.domain_name} - {domain_instance_experiment.instance_name} - {planner_type})', fontsize=14, fontweight='bold')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    plt.xlim(0, iter_cutting_point + 500)

    plt.rc('font', family='serif')

    plt.savefig(plot_path, format='pdf')
    
def plot_cost_curve_per_iteration_with_thresholds(plot_path, domain_instance_experiment, planner_type, random_policy_stats, warm_start_stats_with_thresholds, iter_cutting_point):
    plt.subplots(1, figsize=(8,5))

    statistics = {
        'Random Policy': random_policy_stats,
    }

    for threshold in warm_start_stats_with_thresholds.keys():
        warm_start_stats = warm_start_stats_with_thresholds[threshold]
        statistics[f'Warm Start ({threshold * 100:.0f}%)'] = warm_start_stats

    for key in statistics.keys():
        stats = statistics[key]

        iterations, best_return_curves_mean, best_return_curves_stddev = stat_curves(stats, lambda item : item.best_return)
        
        plt.plot(iterations, best_return_curves_mean, '-', label=key)
        plt.fill_between(iterations, (best_return_curves_mean - best_return_curves_stddev), (best_return_curves_mean + best_return_curves_stddev), alpha=0.2)

    plt.title(f'Best Reward per Iteration\n({domain_instance_experiment.domain_name} - {domain_instance_experiment.instance_name} - {planner_type})', fontsize=14, fontweight='bold')
    plt.xlabel("Iterations", fontsize=14)
    plt.ylabel("Reward", fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.tight_layout()

    plt.xlim(0, iter_cutting_point + 500)

    plt.rc('font', family='serif')

    plt.savefig(plot_path, format='pdf')

root_folder = os.path.dirname(__file__)
plot_folder = f'{root_folder}/_plots'

def rebuild_stats_with_seeds(stats_file_partial_path):
    first_seed = jax_seeds[0]
    next_seeds = jax_seeds[1:]
    
    file_path = f'{stats_file_partial_path}_seed_{first_seed}.pickle'
    
    if not file_exists(file_path):
        print(f'File {file_path} not found. This means that it was not possible to get valid intervals on interval analysis. Skipping experiment')
        return None
    
    stats_with_seed = load_pickle_data(file_path)
    
    for seed in next_seeds:
        file_path = f'{stats_file_partial_path}_seed_{seed}.pickle'
        next_stats_with_seed = load_pickle_data(file_path)
        
        stats_with_seed.append(next_stats_with_seed[0])
    
    return stats_with_seed

def plot_experiments(domain_instance_experiment, strategy_name, threshold, planner_type):    
    print(f'[{os.getpid()}] Domain: {domain_instance_experiment.domain_name} - Instance: {domain_instance_experiment.instance_name} - Ablation Metric: {strategy_name} - Threshold: {threshold} - Planner: {planner_type}')
    
    file_common_suffix = f'{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}_{strategy_name}_{threshold}'

    warm_start_execution_experiment_stats_file_path = f'{root_folder}/_results/warmstart_execution_{planner_type}_run_data_{file_common_suffix}'
    warm_start_execution_experiment_stats = rebuild_stats_with_seeds(warm_start_execution_experiment_stats_file_path)

    if warm_start_execution_experiment_stats is None:
        return

    baseline_execution_stats = rebuild_stats_with_seeds(f'{root_folder}/_results/baseline_{planner_type}_run_data_{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}')
        
    ############################################################
    # Convergence value
    ############################################################

    plot_cost_curve_per_iteration_path = f'{plot_folder}/convergence_value_{planner_type}_{file_common_suffix}.pdf'
    plot_cost_curve_per_iteration(plot_cost_curve_per_iteration_path, domain_instance_experiment, planner_type, baseline_execution_stats, warm_start_execution_experiment_stats, domain_instance_experiment.iter_cutting_point)

    ############################################################
    # Convergence time
    ############################################################

    warm_start_creation_experiment_stats = rebuild_stats_with_seeds(f'{root_folder}/_results/warmstart_creation_{planner_type}_run_data_{file_common_suffix}')

    evaluation_time = read_fluent_evaluation_time_csv(f'{root_folder}/_results/time_{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}_{strategy_name}.csv')
    warm_start_computation = np.mean(list(map(lambda item : item.elapsed_time, warm_start_creation_experiment_stats)))
    warm_start_execution = np.mean(list(map(lambda item : item.elapsed_time, warm_start_execution_experiment_stats)))
    baseline_execution = np.mean(list(map(lambda item : item.elapsed_time, baseline_execution_stats)))

    plot_convergence_time_path = f'{plot_folder}/convergence_time_{planner_type}_{file_common_suffix}.pdf'
    plot_convergence_time(plot_convergence_time_path, domain_instance_experiment, planner_type, evaluation_time, warm_start_computation, warm_start_execution, baseline_execution)

def plot_summarizations(domain_instance_experiment, strategy_name, planner_type):    
    print(f'[{os.getpid()}] Domain: {domain_instance_experiment.domain_name} - Instance: {domain_instance_experiment.instance_name} - Ablation Metric: {strategy_name} - Planner: {planner_type}')
    
    baseline_execution_stats = rebuild_stats_with_seeds(f'{root_folder}/_results/baseline_{planner_type}_run_data_{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}')
    
    warm_start_stats_with_thresholds = {}
    
    for threshold in threshold_to_choose_fluents:
        file_common_suffix = f'{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}_{strategy_name}_{threshold}'
        warm_start_execution_experiment_stats_file_path = f'{root_folder}/_results/warmstart_execution_{planner_type}_run_data_{file_common_suffix}'
        warm_start_execution_experiment_stats = rebuild_stats_with_seeds(warm_start_execution_experiment_stats_file_path)
        
        if warm_start_execution_experiment_stats is None:
            return

        warm_start_stats_with_thresholds[threshold] = warm_start_execution_experiment_stats
    
    ############################################################
    # Convergence value
    ############################################################

    plot_cost_curve_per_iteration_path = f'{plot_folder}/summarized_convergence_value_{planner_type}_{strategy_name}_{domain_instance_experiment.domain_name}_{domain_instance_experiment.instance_name}.pdf'
    plot_cost_curve_per_iteration_with_thresholds(plot_cost_curve_per_iteration_path, domain_instance_experiment, planner_type, baseline_execution_stats, warm_start_stats_with_thresholds, domain_instance_experiment.iter_cutting_point)


if __name__ == '__main__':
    prepare_parallel_experiment_on_main()
    
    start_time = time.time()
    
    print('--------------------------------------------------------------------------------')
    print('Abstraction Experiment - Generating graphs for PtB with warm start')
    print('--------------------------------------------------------------------------------')
    print()

    experiments = get_experiments()

    # create combination of parameters that we will use to create single plots
    args_list_plots = []
    
    for domain_instance_experiment in experiments:
        for strategy_name in domain_instance_experiment.bound_strategies.keys():
            for threshold in threshold_to_choose_fluents:
                if run_drp:
                    args_list_plots.append( (domain_instance_experiment, strategy_name, threshold, 'drp') )  
                if run_slp:
                    args_list_plots.append( (domain_instance_experiment, strategy_name, threshold, 'slp') )

    # run plot generation in parallel
    run_experiment_in_parallel(plot_experiments, args_list_plots)
    
    # create combination of parameters that we will use to create summarized plots
    args_list_summarization = []
    
    for domain_instance_experiment in experiments:
        for strategy_name in domain_instance_experiment.bound_strategies.keys():
            if run_drp:
                args_list_summarization.append( (domain_instance_experiment, strategy_name, 'drp') )  
            if run_slp:
                args_list_summarization.append( (domain_instance_experiment, strategy_name, 'slp') )

    # run plot generation in parallel
    run_experiment_in_parallel(plot_summarizations, args_list_summarization)

    end_time = time.time()
    elapsed_time = end_time - start_time

    print()
    print('--------------------------------------------------------------------------------')
    print('Elapsed Time: ', elapsed_time)
    print('--------------------------------------------------------------------------------')
    print()