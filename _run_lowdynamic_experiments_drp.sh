#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to run experiments with DRP.'

time {
    PYTHONWARNINGS=ignore python ./experiments/lowdynamicfluent-drp/step0_create_baselise_run_drp.py
    PYTHONWARNINGS=ignore python ./experiments/lowdynamicfluent-drp/step1_evaluate_fluents.py
    PYTHONWARNINGS=ignore python ./experiments/lowdynamicfluent-drp/step2_create_grounded_ablated_model.py
    PYTHONWARNINGS=ignore python ./experiments/lowdynamicfluent-drp/step3_create_warm_start_policies_and_run_drp.py
    PYTHONWARNINGS=ignore python ./experiments/lowdynamicfluent-drp/step4_compute_experiment_statistics.py
}