#Specifying the format for the output of the 'time' command
TIMEFORMAT='Elapsed time is %R seconds.'

time {
    sh ./_run_intervalanalysis_experiments_slp.sh
    sh ./_run_intervalanalysis_experiments_drp.sh
}