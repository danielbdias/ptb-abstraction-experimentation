#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to run ablation experiments with SLP.'

time {
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis/step2_create_grounded_ablated_model.py
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis/step3_create_warm_start_policies_and_run.py
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis/step4_compute_experiment_statistics.py
}