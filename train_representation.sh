#!/bin/bash -l
#PBS -l select=128:system=polaris
#PBS -l place=scatter
#PBS -l walltime=3:00:00
#PBS -q prod
#PBS -A datascience
#PBS -l filesystems=home:grand


# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/NEXT_SparseEventID
cd ${WORK_DIR}


# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`

OVERSUBSCRIBE=1
# Turn on MPS:
# nvidia-cuda-mps-control -d

let NRANKS_PER_NODE=4*${OVERSUBSCRIBE}


let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=64
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}

echo "Global batch size: ${GLOBAL_BATCH_SIZE}"

# Set up software deps:
module load conda/2022-09-08
conda activate

# Add-ons from conda:
source /home/cadams/Polaris/polaris_conda_2022-09-08-venv/bin/activate

module load cray-hdf5/1.12.1.3

# Env variables for better scaling:
export NCCL_COLLNET_ENABLE=1
export NCCL_NET_GDR_LEVEL=PHB


run_id=repr_mb${GLOBAL_BATCH_SIZE}-lr1e-3-smallerAug-acc-deeper

CPU_AFFINITY=24-31:16-23:8-15:0-7
export OMP_NUM_THREADS=8


mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind list:${CPU_AFFINITY} \
python bin/exec.py \
--config-name learn_rep \
mode=train \
framework.sparse=True \
framework.oversubscribe=${OVERSUBSCRIBE} \
mode.optimizer.lr_schedule.peak_learning_rate=0.001 \
run.distributed=True \
run.profile=True \
run.id=${run_id} \
run.minibatch_size=${GLOBAL_BATCH_SIZE} \
run.length=500
