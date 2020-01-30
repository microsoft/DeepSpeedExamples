MPI_HOSTS=/job/mpi-hosts
kill_cmd="pssh -P -p 2048 -t 0 -h ${MPI_HOSTS} pkill -9 python"
eval ${kill_cmd}

