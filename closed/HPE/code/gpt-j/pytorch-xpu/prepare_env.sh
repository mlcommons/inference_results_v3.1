#!/bin/bash

set -x

: ${WORK_DIR=${1:-${PWD}}}
: ${USE_INT4=${2:-false}}

pushd ${WORK_DIR}

sudo apt-get install cmake libblas-dev liblapack-dev numactl unzip

echo '==> Building HuggingFace/Transformers'
if [[ ${USE_INT4} == true ]];then
  git clone https://github.com/huggingface/transformers.git
  pushd transformers
  git checkout v4.29.2
  git apply ${WORK_DIR}/patches/int4-transformers.patch
  python setup.py develop 2>&1 | tee build.log
  popd
else
  pip install transformers==4.29.2
fi

echo '==> Building mlperf-loagen'
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
pushd mlperf_inference/loadgen
CFLAGS="-std=c++14 -O3" python setup.py bdist_wheel
cd ..; pip install --force-reinstall loadgen/dist/`ls -r loadgen/dist/ | head -n1` ; cd -
popd

echo '==> Building IPEX-XPU'
# default installation location {ONEAPI_ROOT} is /opt/intel/oneapi for root account, ${HOME}/intel/oneapi for other accounts.
source ${WORK_DIR}/setenv_ipex.sh
git clone https://github.com/intel-innersource/frameworks.ai.pytorch.ipex-gpu.git
# git clone https://github.com/intel/intel-extension-for-pytorch.git
pushd frameworks.ai.pytorch.ipex-gpu
if [[ ${USE_INT4} == true ]]; then
  git checkout dev/LLM-INT4
else
  git checkout dev/LLM-MLperf
fi
git submodule sync && git submodule update --init --recursive
pip install -r requirements.txt
python setup.py install 2>&1 | tee build.log
popd

set +x
