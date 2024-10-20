#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to run hyperparam optimization.'

time {
    PYTHONWARNINGS=ignore python ./experiments/intervalanalysis/_parameter_tuning.py
}