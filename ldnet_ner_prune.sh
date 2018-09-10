FIRST_RUN=1

DATA_ROOT="data/"
NER_DATASET=$DATA_ROOT/ner_dataset.pk

CHECKPOINT_ROOT="checkpoint/"
NER_CHECKPOINT=$CHECKPOINT_ROOT/ner.th

CHECKPOINT_NAME="p_ner0"

green=`tput setaf 2`
reset=`tput sgr0`

if [ $FIRST_RUN == 1 ] && [ ! -e $NER_DATASET ]; then
    echo ${green}=== Downloading Dataset ===${reset}
    mkdir -p DATA_ROOT
    curl http://dmserv4.cs.illinois.edu/ner_dataset.pk -o $NER_DATASET
fi

if [ $FIRST_RUN == 1 ] && [ ! -e $NER_CHECKPOINT ]; then
    echo ${green}=== Downloading Checkpoint ===${reset}
    mkdir -p CHECKPOINT_ROOT
    curl http://dmserv4.cs.illinois.edu/ner.th -o $NER_CHECKPOINT
fi

echo ${green}=== Pruning NER Model ===${reset}
python prune_sparse_seq.py --cp_root $CHECKPOINT_ROOT --checkpoint_name $CHECKPOINT_NAME --corpus $NER_DATASET --load_seq $NER_CHECKPOINT --seq_lambda0 0.05 --seq_lambda1 2
