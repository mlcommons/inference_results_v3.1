#!/bin/bash
export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"
# calibration
python python/calibration.py \
        --max-batchsize=65536 \
        --model-path=/data/mlperf_data/dlrm_2/model/bf16/dlrm-multihot-pytorch.pt \
        --dataset-path=/data/mlperf_data/dlrm_2/data_npy \
        --use-int8 --calibration

