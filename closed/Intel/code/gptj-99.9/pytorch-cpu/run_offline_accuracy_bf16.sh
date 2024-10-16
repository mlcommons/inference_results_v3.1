#!/bin/bash

export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

export num_physical_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
num_numa=$(numactl --hardware|grep available|awk -F' ' '{ print $2 }')
num_sockets=$(lscpu | grep Socket | awk '{print $2}')

export USE_TPP=1

USER_CONF=user_bf16.conf

NUM_PROC=$num_sockets
CPUS_PER_PROC=$((num_physical_cores/num_sockets))
WORKERS_PER_PROC=4
TOTAL_SAMPLE_COUNT=13368
BATCH_SIZE=7
TIMESTAMP=$(date +%m-%d-%H-%M)
HOSTNAME=$(hostname)
OUTPUT_DIR=offline-accuracy-output-${HOSTNAME}-batch-${BATCH_SIZE}-procs-${NUM_PROC}-ins-per-proc-${WORKERS_PER_PROC}-${TIMESTAMP}


python -u runner.py --workload-name gptj \
	--scenario Offline \
	--mode Accuracy \
	--num-proc ${NUM_PROC} \
	--cpus-per-proc ${CPUS_PER_PROC} \
	--model-checkpoint-path ${CHECKPOINT_DIR} \
	--dataset-path ${VALIDATION_DATA_JSON} \
	--batch-size ${BATCH_SIZE} \
	--mlperf-conf mlperf.conf \
	--user-conf ${USER_CONF} \
	--precision bf16 \
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
		--model-name-or-path ${CHECKPOINT_DIR} 2>&1 | tee -a accuracy-offline-${TIMESTAMP}.txt ${OUTPUT_DIR}.log
fi

