#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to run baseline and find fluents to ablate with SLP.'

time {
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis/step0_create_baselise_run.py
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis/step1_evaluate_fluents.py
}