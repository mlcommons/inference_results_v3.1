#!/bin/bash

export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

export num_physical_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
num_numa=$(numactl --hardware|grep available|awk -F' ' '{ print $2 }')

python ../../user_config.py
USER_CONF=user.conf

NUM_PROC=$num_numa
CPUS_PER_PROC=$((num_physical_cores/num_numa))
WORKERS_PER_PROC=1
TOTAL_SAMPLE_COUNT=500
BATCH_SIZE=1
TIMESTAMP=$(date +%m-%d-%H-%M)
HOSTNAME=$(hostname)
OUTPUT_DIR=offline-accuracy-output-${HOSTNAME}-batch-${BATCH_SIZE}-procs-${NUM_PROC}-ins-per-proc-${WORKERS_PER_PROC}-${TIMESTAMP}
VALIDATION_DATA_JSON=/home/hengyume/mlperf/closed/Intel/code/gpt-j/neural_engine/data/validation-data/cnn_dailymail_validation.json
LIBRARY_PATH=/home/hengyume/nlptk/intel_extension_for_transformers/backends/neural_engine/graph/build/lib/libGptjPyBind.so
MODEL_PATH=/home/hengyume/model/mlperf-gptj-q4.bin
CHECKPOINT_DIR=/home/hengyume/mlperf/closed/Intel/code/gpt-j/pytorch-cpu/model/gpt-j/checkpoint-final
python runner.py --workload-name gptj \
	--scenario Offline \
	--mode Accuracy \
	--num-proc ${NUM_PROC} \
	--cpus-per-proc ${CPUS_PER_PROC} \
	--dataset-path ${VALIDATION_DATA_JSON} \
    --library-path ${LIBRARY_PATH} \
    --model-path ${MODEL_PATH} \
	--batch-size ${BATCH_SIZE} \
	--mlperf-conf mlperf.conf \
	--user-conf user.conf \
	--pad-inputs \
	--workers-per-proc ${WORKERS_PER_PROC} \
	--total-sample-count ${TOTAL_SAMPLE_COUNT} \
	--output-dir ${OUTPUT_DIR} \
	2>&1 | tee ${OUTPUT_DIR}.log


if [ -e ${OUTPUT_DIR}/mlperf_log_accuracy.json ]; then
	echo " ==================================="
	echo "         Evaluating Accuracy        "
	echo " ==================================="

	python evaluation.py --mlperf-accuracy-file ${OUTPUT_DIR}/mlperf_log_accuracy.json \
		--dataset-file ${VALIDATION_DATA_JSON} \
		--model-name-or-path ${CHECKPOINT_DIR} 2>&1 | tee "accuracy.txt"
fi

