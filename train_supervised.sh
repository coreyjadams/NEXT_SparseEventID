#!/bin/bash -l
#PBS -l select=2:system=polaris
#PBS -l place=scatter
#PBS -l walltime=1:00:00
#PBS -q debug
#PBS -A datascience
#PBS -l filesystems=home:grand


# What's the cosmic tagger work directory?
WORK_DIR=/home/cadams/Polaris/NEXT_SparseEventID
cd ${WORK_DIR}


# MPI and OpenMP settings
NNODES=`wc -l < $PBS_NODEFILE`
NRANKS_PER_NODE=4

let NRANKS=${NNODES}*${NRANKS_PER_NODE}

LOCAL_BATCH_SIZE=128
# let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}*${NRANKS}
let GLOBAL_BATCH_SIZE=${LOCAL_BATCH_SIZE}

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

run_id=supervised_ID_mk_mb${GLOBAL_BATCH_SIZE}-1e-3

# mpiexec -n ${NRANKS} -ppn ${NRANKS_PER_NODE} --cpu-bind=numa \
python bin/exec.py \
--config-name supervised_eventID \
mode=train \
framework.sparse=False \
run.distributed=True \
run.id=${run_id} \
encoder.bias=True \
encoder.normalization=batch \
framework.oversubscribe=1 \
run.minibatch_size=${GLOBAL_BATCH_SIZE} \
run.length=150
