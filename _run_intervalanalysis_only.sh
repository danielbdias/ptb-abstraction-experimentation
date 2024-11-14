#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to find fluents to ablate with SLP.'

time {
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis/step0_evaluate_fluents.py
}