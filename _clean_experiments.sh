#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to clean files.'

time {
    # clean intermediate files
    rm ./experiments/intervalanalysis-gurobi-updated/_intermediate/*.model
    rm ./experiments/intervalanalysis-gurobi-updated/_intermediate/*.rddl
    rm ./experiments/intervalanalysis-gurobi-updated/_intermediate/*.pickle

    # clean plots
    rm ./experiments/intervalanalysis-gurobi-updated/_plots/*.pdf

    # clean results
    rm ./experiments/intervalanalysis-gurobi-updated/_results/*.csv
    rm ./experiments/intervalanalysis-gurobi-updated/_results/*.pickle
}