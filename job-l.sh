#!/bin/bash

# ==========================
# SLURM directives
# =========================

#SBATCH --job-name=Testing
#SBATCH --partition="partition-l"
#SBATCH --qos="train"
#SBATCH --nodes=1
#SBATCH --gpus=2
#SBATCH --mem-per-cpu=2G
#SBATCH --cpus-per-task=10
#SBATCH --ntasks-per-node=2
#SBATCH --time="48:00:00"
#SBATCH --output=slurm-jobs/%j.out
#SBATCH --error=slurm-jobs/%j.out

# Move to the directory where the job was submitted
cd /nfs/home/catman/src/gpt-neox

# "data_path": "data/openwebtext2/openwebtext2_text_document",

# Some potentially useful distributed environment variables
export HOSTNAMES=`scontrol show hostnames "$SLURM_JOB_NODELIST"`
export MASTER_ADDR=$(scontrol show hostnames "$SLURM_JOB_NODELIST" | head -n 1)
export MASTER_PORT=$((12000 + SLURM_JOB_ID % 20000))
export COUNT_NODE=`scontrol show hostnames "$SLURM_JOB_NODELIST" | wc -l`

GPUS_PER_NODE=2
mkdir -p hostfiles
# need to add the current slurm jobid to hostfile name so that we don't add to previous hostfile
hostfile=hostfiles/hosts_$SLURM_JOBID
# be extra sure we aren't appending to a previous hostfile
rm $hostfile &> /dev/null
# loop over the node names
for i in `scontrol show hostnames $SLURM_NODELIST`
do
    # add a line to the hostfile
    echo $i slots=$GPUS_PER_NODE >>$hostfile
done

# Tell DeepSpeed where to find our generated hostfile via DLTS_HOSTFILE
export DLTS_HOSTFILE=hostfiles/hosts_$SLURM_JOBID

export LD_LIBRARY_PATH=~catman/miniconda3/lib:$LD_LIBRARY_PATH
export PATH=$HOME/.local/bin:$PATH export CPATH=~catman/miniconda3/include:$CPATH

/bin/python3 deepy.py train.py ./configs/125M-json.yml ./configs/openwebtxt.yml ./configs/gru.yml ./slurm.yml --wandb_run_name="gru"
#/bin/python3 deepy.py train.py ./configs/125M-json.yml ./configs/openwebtxt.yml ./slurm.yml --wandb_run_name="normal"
