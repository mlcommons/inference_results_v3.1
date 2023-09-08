#!/bin/bash

date
set -x

: ${WORK_DIR:=${PWD}}
: ${WARMUP_DIR:=${WORK_DIR}/data/cnn_eval_warmup.json}
: ${DATA_DIR:=${WORK_DIR}/data/cnn_eval.json}
: ${MODEL_DIR:=${WORK_DIR}/model}
: ${LOG_DIR:=${WORK_DIR}/logs}
: ${REF_LOG_PATH:=${WORK_DIR}/data/out_prompts_a100_fp16.json}
: ${NLTK_DATA:=${WORK_DIR}/nltk_data}
: ${SCENARIO:=Offline}  # [Offline | Server | SingleStream]
: ${MAX_EXP:=-1}  # [Maximum number of samples for inference]
: ${DEVICE:=xpu}  # [cpu | xpu | cuda]
: ${DTYPE:=float16}  # [float32 | float16 | int4]
: ${ACCURACY:=false}
: ${BS:=1}  # batchsize
: ${BEAM:=4}  # submission request 4
: ${INTER:=1}  # num workers(tiles)
: ${WORLD_SIZE:=8}  # total ranks
: ${START:=0}  # start rank(tile)
: ${WARMUP:=true}
: ${PROFILE:=false}
: ${OPT:=true}  # true to enable optimize_transformers
: ${SORT:=true}  # true to sort samples
: ${VERBOSE:=false}  # true to dump output prompts
: ${DEBUG:=false}
: ${LOG_LEVEL:=10}
: ${ENABLE_SDP_FUSION:=1}  # 1 to enable SDP Fusion
: ${DISABLE_KV_CACHE:=0}  # 1 to disable KV Cache
: ${PICK:=-1}  # pick by sample index from raw dataset
: ${REPEAT:=1}  # times of repeating picked samples
: ${PAD:="left"}  # padding side: left or right
: ${DYNAMIC_BATCHING:="false"}  # true to enable dynamic batching

if [[ ${ACCURACY} == true ]]; then
  OUT_DIR=${LOG_DIR}/${SCENARIO}/Accuracy
else
  OUT_DIR=${LOG_DIR}/${SCENARIO}/Performance
fi

mkdir -p ${OUT_DIR}

export GPTJ_LOG_LEVEL=${LOG_LEVEL}

# TODO: cmp JeMalloc VS TcMalloc
# echo "Use JeMalloc memory allocator"
# export LD_PRELOAD=${CONDA_PREFIX}/lib/libjemalloc.so
# export MALLOC_CONF="oversize_threshold:1,background_thread:true,metadata_thp:auto,dirty_decay_ms:9000000000,muzzy_decay_ms:9000000000"

echo "Use TcMalloc memory allocator"
export LD_PRELOAD=${LD_PRELOAD}:${CONDA_PREFIX}/lib/libtcmalloc.so:${LD_PRELOAD}

echo "Use Intel OpenMP"
export LD_PRELOAD=${CONDA_PREFIX}/lib/libiomp5.so:${LD_PRELOAD}

echo "Set KMP"
export KMP_BLOCKTIME=1
export KMP_SETTINGS=1
export KMP_AFFINITY=granularity=fine,compact,1,0

echo "Set IPEX-XPU runtime env"
export SYCL_PI_LEVEL_ZERO_USE_IMMEDIATE_COMMANDLISTS=2
export ENABLE_SDP_FUSION=${ENABLE_SDP_FUSION}
export DISABLE_KV_CACHE=${DISABLE_KV_CACHE}

echo "Set File Descriptor Limitation"
FD_MAX=`ulimit -n -H`
ulimit -n $((FD_MAX-1))

if [[ ${DTYPE} == "int4" ]]; then
  cp configs/pytorch_model.bin.index_int4.json ${MODEL_DIR}/pytorch_model.bin.index.json
else
  cp configs/pytorch_model.bin.index.json ${MODEL_DIR}
fi

SCRIPT_ARGS=" --scenario ${SCENARIO}"
SCRIPT_ARGS+=" --model_path ${MODEL_DIR}"
SCRIPT_ARGS+=" --dataset_path ${DATA_DIR}"
SCRIPT_ARGS+=" --log_dir ${OUT_DIR}"
SCRIPT_ARGS+=" --device ${DEVICE}"
SCRIPT_ARGS+=" --dtype ${DTYPE}"
SCRIPT_ARGS+=" --batch_size ${BS}"
SCRIPT_ARGS+=" --num_beams ${BEAM}"
SCRIPT_ARGS+=" --num_workers ${INTER}"
SCRIPT_ARGS+=" --world_size ${WORLD_SIZE}"
SCRIPT_ARGS+=" --start_rank ${START}"
SCRIPT_ARGS+=" --max_examples ${MAX_EXP}"
SCRIPT_ARGS+=" --ref_log_path ${REF_LOG_PATH}"
SCRIPT_ARGS+=" --padding_side ${PAD}"
SCRIPT_ARGS+=" --user_conf ${WORK_DIR}/configs/user_${DTYPE}.conf"
SCRIPT_ARGS+=" --repeat ${REPEAT}"
[ ${PICK} != -1 ] && SCRIPT_ARGS+=" --pick_index ${PICK}"
[ ${WARMUP} == true ] && SCRIPT_ARGS+=" --warmup --warmup_path ${WARMUP_DIR}"
[ ${PROFILE} == true ] && SCRIPT_ARGS+=" --profile"
[ ${ACCURACY} == true ] && SCRIPT_ARGS+=" --accuracy"
[ ${OPT} == true ] && SCRIPT_ARGS+=" --optimize_transformers"
[ ${SORT} == true ] && SCRIPT_ARGS+=" --sort"
[ ${VERBOSE} == true ] && SCRIPT_ARGS+=" --verbose"
[ ${DYNAMIC_BATCHING} == true ] && SCRIPT_ARGS+=" --dynamic_batching"

[ ${DEBUG} == "pdb" ] && EXEC_ARGS+=" ipdb3"
[ ${DEBUG} == "gdb" ] && EXEC_ARGS+=" gdb --args python"
[ ${DEBUG} == "lldb" ] && EXEC_ARGS+=" lldb python --"
[ ${DEBUG} == false ] && EXEC_ARGS+=" python -u"

${EXEC_ARGS} main.py ${SCRIPT_ARGS}
date

if [[ ${ACCURACY} == true ]]; then
  SCRIPT_ARGS=" --mlperf_accuracy_file ${OUT_DIR}/mlperf_log_accuracy.json"
  SCRIPT_ARGS+=" --model_path ${MODEL_DIR}"
  SCRIPT_ARGS+=" --dataset_path ${DATA_DIR}"
  SCRIPT_ARGS+=" --sort_results"
  SCRIPT_ARGS+=" --evaluate"
  SCRIPT_ARGS+=" --skip_pick_result"
  [ ${PICK} != -1 ] && SCRIPT_ARGS+=" --pick_index ${PICK}"
  [ ${VERBOSE} == true ] && SCRIPT_ARGS+=" --verbose"

  ${EXEC_ARGS} evaluation.py ${SCRIPT_ARGS} 2>&1 | tee ${OUT_DIR}/accuracy.txt
  date
fi

set +x
