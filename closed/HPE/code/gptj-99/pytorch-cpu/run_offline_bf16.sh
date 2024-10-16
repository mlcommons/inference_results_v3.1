#!/bin/bash

export KMP_BLOCKTIME=1
export KMP_AFFINITY=granularity=fine,compact,1,0
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libiomp5.so
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so

export num_physical_cores=`lscpu -b -p=Core,Socket | grep -v '^#' | sort -u | wc -l`
num_numa=$(numactl --hardware|grep available|awk -F' ' '{ print $2 }')

export USE_TPP=1

python ../../user_config.py
USER_CONF=user.conf

NUM_PROC=2 #$num_numa
CPUS_PER_PROC=56 #$((num_physical_cores/num_numa))
WORKERS_PER_PROC=1
TOTAL_SAMPLE_COUNT=13368
BATCH_SIZE=16
TIMESTAMP=$(date +%m-%d-%H-%M)
HOSTNAME=$(hostname)
OUTPUT_DIR=offline-output-${HOSTNAME}-batch-${BATCH_SIZE}-procs-${NUM_PROC}-ins-per-proc-${WORKERS_PER_PROC}-${TIMESTAMP}

python runner.py --workload-name gptj \
	--scenario Offline \
	--mode Performance \
	--num-proc ${NUM_PROC} \
	--cpus-per-proc ${CPUS_PER_PROC} \
	--model-checkpoint-path ${CHECKPOINT_DIR} \
	--dataset-path ${VALIDATION_DATA_JSON} \
	--batch-size ${BATCH_SIZE} \
	--mlperf-conf mlperf.conf \
	--user-conf user.conf \
	--precision bf16 \
	--warmup \
	--workers-per-proc ${WORKERS_PER_PROC} \
	--total-sample-count ${TOTAL_SAMPLE_COUNT} \
	--output-dir ${OUTPUT_DIR} \
	2>&1 | tee ${OUTPUT_DIR}.log

