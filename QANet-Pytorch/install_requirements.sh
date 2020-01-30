SCRIPT_DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

MPI_HOSTS=/job/mpi-hosts
install_cmd="pssh -P -p 2048 -t 0 -h ${MPI_HOSTS} sudo -H pip install visdom spacy==2.0.11 tqdm"
eval ${install_cmd}

show_cmd="pssh -I -P -p 2048 -t 0 -h ${MPI_HOSTS} sudo -H pip show visdom spacy tqdm"
eval ${show_cmd}
