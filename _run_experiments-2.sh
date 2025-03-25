# fail if any errors
set -ex

#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to run experiments.'

time {
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis-gurobi-2/step0_evaluate_fluents.py
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis-gurobi-2/step1_create_bounds_for_ablated_models.py
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis-gurobi-2/step2_create_baselise_run.py
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis-gurobi-2/step3_create_warm_start_policies_and_run.py
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis-gurobi-2/step4_compute_experiment_statistics.py
}