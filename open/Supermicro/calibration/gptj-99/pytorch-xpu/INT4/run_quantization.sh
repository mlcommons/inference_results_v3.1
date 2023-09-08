#!/bin/bash

set -x

: ${DATA_DIR:=${1:${PWD}/data}}
: ${MODEL_DIR:=${2:${PWD}/model}}

CAL_SAMPLES=(128)
GROUPSIZE=(-1)
COMPRESSION_FACTOR=2

for cal_size in ${CAL_SAMPLES[@]}; do
  for g in ${GROUPSIZE[@]}; do
    echo "Running groups ${g} and samples ${cal_size}"
    numactl -m 0 -C 0-55 python -u gptj.py --model ${MODEL_DIR} \
    --wbits 4 \
    --true-sequential \
    --act-order \
    --groupsize ${g} \
    --save ${MODEL_DIR}/gpt-j-quantized_model_${g}g_${cal_size}samples.pt \
    --calib-data-path ${DATA_DIR}/cnn_dailymail_calibration.json \
    --nsamples ${cal_size} \
    --quant-config-output ${MODEL_DIR}/gpt-j-quantized_model_params.json \
    --compression-factor ${COMPRESSION_FACTOR} \
    --compression-dim "N" \
    --calib-iters ${cal_size} \
    --quantize-lm-head \
    2>&1 | tee log_${g}groups_${cal_size}samples_cf_${COMPRESSION_FACTOR}.log
  done
done

mv ${MODEL_DIR}/gpt-j-quantized_model_${g}g_${cal_size}samples.pt ${MODEL_DIR}/int4_weight.pt

set +x
