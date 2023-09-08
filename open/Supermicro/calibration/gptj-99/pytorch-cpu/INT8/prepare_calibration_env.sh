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


conda install mkl mkl-include -y
conda install gperftools cmake jemalloc==5.2.1 -c conda-forge -y

# ========== Install INC deps ===========
python -m pip install cpuid nltk datasets tqdm numpy cython absl-py tqdm sentencepiece accelerate opencv-python-headless==4.5.5.64
pip install git+https://github.com/intel/neural-compressor.git@${INC_VERSION}


# ========== Install torch ===========
pip3 install --pre torch==2.1.0.dev20230711+cpu torchvision==0.16.0.dev20230711+cpu torchaudio==2.1.0.dev20230711+cpu --index-url https://download.pytorch.org/whl/nightly/cpu
ABI=$(python -c "import torch; print(int(torch._C._GLIBCXX_USE_CXX11_ABI))")

# ========== Build =========
cd ${WORKDIR}
mkdir llvm-project && cd llvm-project
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/cmake-16.0.6.src.tar.xz
wget https://github.com/llvm/llvm-project/releases/download/llvmorg-16.0.6/llvm-16.0.6.src.tar.xz

tar -xf cmake-16.0.6.src.tar.xz && mv cmake-16.0.6.src cmake
tar -xf llvm-16.0.6.src.tar.xz && mv llvm-16.0.6.src llvm
LLVM_SRC=${PWD}
mkdir build && cd build
cmake -G "Unix Makefiles" -DCMAKE_BUILD_TYPE=Release -DCMAKE_CXX_FLAGS="-D_GLIBCXX_USE_CXX11_ABI=${ABI}" -DLLVM_TARGETS_TO_BUILD=X86 -DLLVM_ENABLE_TERMINFO=OFF -DLLVM_INCLUDE_TESTS=OFF -DLLVM_INCLUDE_EXAMPLES=OFF -DLLVM_INCLUDE_BENCHMARKS=OFF -DLLVM_BUILD_LLVM_DYLIB=ON ../llvm/

cmake --build . -j $(nproc)
LLVM_ROOT=${LLVM_SRC}/release
if [ -d ${LLVM_ROOT} ]; then
    rm -rf ${LLVM_ROOT}
fi
cmake -DCMAKE_INSTALL_PREFIX=${LLVM_ROOT} -P cmake_install.cmake
ln -s ${LLVM_ROOT}/bin/llvm-config ${LLVM_ROOT}/bin/llvm-config-13
export PATH=${LLVM_ROOT}/bin:$PATH
export LD_LIBRARY_PATH=${LLVM_ROOT}/lib:$LD_LIBRARY_PATH
export USE_LLVM=${LLVM_ROOT}
export LLVM_DIR=${USE_LLVM}/lib/cmake/llvm

# =========== Install Ipex ==========
cd ${WORKDIR}
git clone --branch llm_feature_branch https://github.com/intel/intel-extension-for-pytorch ipex-cpu
cd ipex-cpu
export IPEX_DIR=${PWD}
git submodule sync
git submodule update --init --recursive
 
export DNNL_GRAPH_BUILD_COMPILER_BACKEND=1
export CXXFLAGS="${CXXFLAGS} -D__STDC_FORMAT_MACROS"
python setup.py clean
python setup.py bdist_wheel 2>&1 | tee ipex-build.log
unset DNNL_GRAPH_BUILD_COMPILER_BACKEND
unset LLVM_DIR
unset USE_LLVM
python -m pip install --force-reinstall dist/*.whl

# ============ Install transformers =========
pip install transformers==4.28.1
