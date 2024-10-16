#!/bin/bash

CONDA_ENV_NAME=$1

set -x

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

export WORKDIR=${DIR}/${CONDA_ENV_NAME}
if [ -d ${WORKDIR} ]; then
    rm -rf ${WORKDIR}
fi

echo "Working directory is ${WORKDIR}"
mkdir -p ${WORKDIR}

source $(dirname `/usr/bin/which conda`)/activate

conda create -n ${CONDA_ENV_NAME} python=3.9 --yes
conda init bash
conda activate ${CONDA_ENV_NAME}


conda install mkl==2023.2.0 mkl-include==2023.2.0 -y
conda install gperftools==2.10 jemalloc==5.2.1 pybind11==2.10.4 llvm-openmp==16.0.6 -c conda-forge -y
conda install gcc=12.3 gxx=12.3 ninja==1.11.1 -c conda-forge -y
conda install -c conda-forge zlib -y
export CC=`which gcc`
export CXX=`which g++`

pip install setuptools==58.2.0
# ========== Install deps ===========
python -m pip install cmake==3.27.0 cpuid==0.0.11 nltk==3.8.1 evaluate==0.4.0 protobuf==3.20.3 absl-py==1.4.0 rouge-score==0.1.2 tqdm==4.65.0 numpy==1.25.2 cython==3.0.0 sentencepiece==0.1.99 accelerate==0.21.0
pip install optimum


# ========== Install torch ===========
pip3 install --pre torch==2.1.0.dev20230711+cpu torchvision==0.16.0.dev20230711+cpu torchaudio==2.1.0.dev20230711+cpu --index-url https://download.pytorch.org/whl/nightly/cpu

# =========== Install Ipex ==========
cd ${WORKDIR}
git clone https://github.com/intel/intel-extension-for-pytorch ipex-cpu
cd ipex-cpu
git checkout v2.1.0.dev+cpu.llm.mlperf
export IPEX_DIR=${PWD}
git submodule sync
git submodule update --init --recursive

python setup.py clean
python setup.py bdist_wheel 2>&1 | tee ipex-build.log
python -m pip install --force-reinstall dist/*.whl

# =========== Install TPP Pytorch Extension ==========
cd ${WORKDIR}
git clone --branch mlperf_infer_31 https://github.com/libxsmm/tpp-pytorch-extension/ tpp-pytorch-extension
cd tpp-pytorch-extension
git submodule update --init
conda install ninja -y
python setup.py install

# ============ Install transformers =========

pip install transformers==4.28.1

# ============ Install loadgen ==========
cd ${WORKDIR}
git clone https://github.com/mlcommons/inference.git mlperf_inference
cd mlperf_inference
export MLPERF_INFERENCE_ROOT=${PWD}

git submodule update --init --recursive third_party/pybind/

cd loadgen
python -m pip install .

# Copy the mlperf.conf
cp ${MLPERF_INFERENCE_ROOT}/mlperf.conf ${DIR}/

# ======== Build utils =======
cd ${DIR}/utils
python -m pip install .

