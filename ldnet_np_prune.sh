FIRST_RUN=1

DATA_ROOT="data/"
NP_DATASET=$DATA_ROOT/np_dataset.pk

CHECKPOINT_ROOT="checkpoint/"
NP_CHECKPOINT=$CHECKPOINT_ROOT/np.th

CHECKPOINT_NAME="p_np0"

green=`tput setaf 2`
reset=`tput sgr0`

if [ $FIRST_RUN == 1 ] && [ ! -e $NP_DATASET ]; then
    echo ${green}=== Downloading Dataset ===${reset}
    mkdir -p DATA_ROOT
    curl http://dmserv4.cs.illinois.edu/np_dataset.pk -o $NP_DATASET
fi

if [ $FIRST_RUN == 1 ] && [ ! -e $NP_CHECKPOINT ]; then
    echo ${green}=== Downloading Checkpoint ===${reset}
    mkdir -p CHECKPOINT_ROOT
    curl http://dmserv4.cs.illinois.edu/np.th -o $NP_CHECKPOINT
fi

echo ${green}=== Pruning NER Model ===${reset}
python prune_sparse_seq.py --cp_root $CHECKPOINT_ROOT --checkpoint_name $CHECKPOINT_NAME --corpus $NP_DATASET --load_seq $NP_CHECKPOINT --seq_lambda0 0.05 --seq_lambda1 2
