#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds to clean files.'

time {
    # clean intermediate files
    rm ./experiments/intervalanalysis-jaxplan/_intermediate/*.model
    rm ./experiments/intervalanalysis-jaxplan/_intermediate/*.rddl
    rm ./experiments/intervalanalysis-jaxplan/_intermediate/*.pickle

    # clean plots
    rm ./experiments/intervalanalysis-jaxplan/_plots/*.pdf

    # clean results
    rm ./experiments/intervalanalysis-jaxplan/_results/*.csv
    rm ./experiments/intervalanalysis-jaxplan/_results/*.pickle
}