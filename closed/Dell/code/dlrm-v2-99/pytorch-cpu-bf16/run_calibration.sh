#!/bin/bash
# export LD_PRELOAD="${CONDA_PREFIX}/lib/libtcmalloc.so:${CONDA_PREFIX}/lib/libiomp5.so"
# calibration
# numactl -C 56-111 -m 1 python python/calibration.py --max-batchsize=65536 --model-path=$(pwd)/python/model/dlrm-multihot-pytorch.pt --dataset-path=$(pwd)/dataset/terabyte_input --calibration --use-int8

#quantize
numactl -C 56-111 -m 1 python python/calibration.py --max-batchsize=65536 --model-path=$(pwd)/python/model/dlrm-multihot-pytorch.pt --dataset-path=$(pwd)/dataset/terabyte_input --use-int8

