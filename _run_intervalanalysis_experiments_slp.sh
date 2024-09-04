#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to run experiments with SLP.'

time {
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis/step0_create_baselise_run_jaxplan.py
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis/step1_evaluate_fluents.py
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis/step2_create_grounded_ablated_model.py
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis/step3_create_warm_start_policies_and_run_jaxplan.py
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis/step4_compute_experiment_statistics.py
}