#!/bin/bash

export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so
RUN_TYPE="quantize"
CALIB_ITERS=1000
# ========= Run quantization ===========
numactl -m 0 -C 0-55 python -u quantizer.py --${RUN_TYPE} \
	--dtype int4 \
	--state-dict-file "${INT4_CALIBRATION_DIR}/gpt-j-quantized_model_-1g_128samples.pt" \
	--output_dir ${INT4_MODEL_DIR} \
	--cal-data-path ${CALIBRATION_DATA_JSON} \
	--model ${CHECKPOINT_DIR} \
	--calib_iters ${CALIB_ITERS} \
	 2>&1 | tee log_quantize_calibration_samples_${CALIB_ITERS}.log

