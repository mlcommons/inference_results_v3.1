#!/bin/bash

set -x

: ${CONDA_ENV=${1:-'gptj-infer'}}
: ${WORK_DIR=${2:-${PWD}}}
: ${DATA_DIR:=${WORK_DIR}/data}
: ${MODEL_DIR:=${WORK_DIR}/model}
: ${STAGE=${3:--1}}
: ${USE_INT4=${4:-false}}
: ${INTER=${5:-8}}

if [[ ${STAGE} -le -2 ]]; then
  echo '==> Preparing conda env'
  conda create -y -n ${CONDA_ENV} python=3.9
fi

source activate ${CONDA_ENV}

if [[ ${STAGE} -le -1 ]]; then
  echo '==> Preparing env'
  ./prepare_conda_env.sh
  ./prepare_env.sh ${PWD} ${USE_INT4}
fi

if [[ ${STAGE} -le 0 ]]; then
  echo '==> Downloading model'
  mkdir ${MODEL_DIR}
  # download pre-trained huggingface/model
  # python download_gptj.py
  # download fine-tuned mlperf/model
  wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download --output-document checkpoint.zip
  unzip checkpoint.zip
  mv gpt-j/checkpoint-final/* ${MODEL_DIR}
  rm -rf gpt-j
fi

if [[ ${STAGE} -le 1 ]]; then
  echo '==> Downloading dataset'
  mkdir ${DATA_DIR}
  python download_cnndm.py
fi

if [[ ${STAGE} -le 2 ]]; then
  echo '==> Calibrating model'
  python prepare_calibration.py --calibration-list-file configs/calibration-list.txt --output-dir ${DATA_DIR}
  cd ${WORK_DIR}/../../../calibration/gpt-j/pytorch-xpu/INT4/
  conda env create -f quantization-env.yaml
  git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git
  cp run_quantization.sh cnn_dm_dataset.py GPTQ-for-LLaMa/
  cd GPTQ-for-LLaMa
  git apply ../gptj.patch
  source activate gpt-j-quant-env
  DATA_DIR=${DATA_DIR} MODEL_DIR=${MODEL_DIR} ./run_quantization.sh
  cd ${WORK_DIR}
fi

source activate ${CONDA_ENV}

if [[ ${STAGE} -le 3 ]]; then
  source ./setenv_ipex.sh
  if [[ ${USE_INT4} == true ]]; then
    DATA_TYPE=int4
    BS_OFFLINE=28
    BS_SERVER=17
  else
    DATA_TYPE=float16
    BS_OFFLINE=31
    BS_SERVER=20
  fi

  echo '==> Run GPT-J Offline accuracy'
  INTER=${INTER} BS=${BS_OFFLINE} DTYPE=${DATA_TYPE} SCENARIO=Offline ACCURACY=true ./eval_model.sh
  sleep 5
  echo '==> Run GPT-J Offline benchmark'
  INTER=${INTER} BS=${BS_OFFLINE} DTYPE=${DATA_TYPE} SCENARIO=Offline ./eval_model.sh
  sleep 5
  echo '==> Run GPT-J Server accuracy'
  INTER=${INTER} BS=${BS_SERVER} DTYPE=${DATA_TYPE} SCENARIO=Server ACCURACY=true ./eval_model.sh
  sleep 5
  echo '==> Run GPT-J Server benchmark'
  INTER=${INTER} BS=${BS_SERVER} DTYPE=${DATA_TYPE} SCENARIO=Server ./eval_model.sh
  wait
fi

set +x
