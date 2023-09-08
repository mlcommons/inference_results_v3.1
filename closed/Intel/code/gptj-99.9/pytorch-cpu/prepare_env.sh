#!/bin/bash

CONDA_ENV_NAME=$1

set -x

INC_VERSION=a2931eaa4052eec195be3c79a13f7bfa23e54473

DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

export WORKDIR=${DIR}/${CONDA_ENV_NAME}
if [ -d ${WORKDIR} ]; then
    rm -r ${WORKDIR}
fi

echo "Working directory is ${WORKDIR}"
mkdir -p ${WORKDIR}

source ${HOME}/miniconda3/bin/activate

conda create -n ${CONDA_ENV_NAME} python=3.9 --yes
conda init bash
conda activate ${CONDA_ENV_NAME}


conda install mkl==2023.2.0 mkl-include==2023.2.0 -y
conda install gperftools==2.10 jemalloc==5.2.1 pybind11==2.10.4 llvm-openmp==16.0.6 -c conda-forge -y
conda install gcc=12.3 gxx=12.3 ninja==1.11.1 -c conda-forge -y

pip install setuptools==58.2.0
# ========== Install INC deps ===========
python -m pip install cmake==3.27.0 cpuid==0.0.11 nltk==3.8.1 evaluate==0.4.0 protobuf==3.20.3 absl-py==1.4.0 rouge-score==0.1.2 tqdm==4.65.0 numpy==1.25.2 cython==3.0.0 sentencepiece==0.1.99 accelerate==0.21.0 opencv-python-headless==4.5.5.64
pip install git+https://github.com/intel/neural-compressor.git@${INC_VERSION}


# ========== Install torch ===========
pip3 install --pre torch==2.1.0.dev20230711+cpu torchvision==0.16.0.dev20230711+cpu torchaudio==2.1.0.dev20230711+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

# =========== Install TPP Pytorch Extension ==========
cd ${WORKDIR}
git clone --branch mlperf_infer_31 https://github.com/libxsmm/tpp-pytorch-extension/ tpp-pytorch-extension
cd tpp-pytorch-extension
git submodule update --init
conda install -y ninja
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

# ============================
pip install py-libnuma
