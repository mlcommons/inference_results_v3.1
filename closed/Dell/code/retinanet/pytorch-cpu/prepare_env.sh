#!/usr/bin/env bash
DIR="$( cd "$( dirname "${BASH_SOURCE[0]}" )" > /dev/null && pwd )"

CONDA_ENV_NAME=retinanet-env
export WORKDIR=${DIR}/${CONDA_ENV_NAME}
if [ -d ${WORKDIR} ]; then
	sudo rm -r ${WORKDIR}
fi

echo "Working directory is ${WORKDIR}"
mkdir -p ${WORKDIR}
cd ${WORKDIR}

source ${HOME}/anaconda3/etc/profile.d/conda.sh
conda create -n ${CONDA_ENV_NAME} python=3.9 --yes
conda activate ${CONDA_ENV_NAME}

echo "Installiing dependencies for Retinanet"
python -m pip install Pillow pycocotools==2.0.2
python -m pip install opencv-python
python -m pip install absl-py
python -m pip install fiftyone==0.16.5
python -m pip install wheel==0.38.1
python -m pip install future==0.18.3

#conda install typing_extensions --yes
conda config --add channels intel
conda install -c cctbx202211 setuptools==65.5.1 --yes
conda install cmake intel-openmp --yes
conda install -c intel mkl=2022.0.1 --yes
conda install -c intel mkl-include=2022.0.1 --yes
conda install -c conda-forge llvm-openmp --yes
conda install -c conda-forge jemalloc --yes
conda install numpy==1.23.5 --yes

export CMAKE_PREFIX_PATH=${CONDA_PREFIX:-"$(dirname $(which conda))/../"}

#build pytorch and intel-pytorch-extension
git clone https://github.com/pytorch/pytorch.git pytorch
cd pytorch

git checkout v1.12.0-rc7

git submodule sync
git submodule update --init --recursive
git fetch origin pull/89925/head
git cherry-pick 78cad998e505b667d25ac42f8aaa24409f5031e1
python setup.py install

cd ${WORKDIR}
git clone https://github.com/intel/intel-extension-for-pytorch.git ipex-cpu-dev
cd ipex-cpu-dev

git checkout mlperf/retinanet

git submodule sync
git submodule update --init --recursive

cp ${DIR}/runtime_ignore_dequant_check.patch .
git apply runtime_ignore_dequant_check.patch

python setup.py install

cd ${WORKDIR}

# Install Loadgen
echo "=== Installing loadgen ==="
git clone --recurse-submodules https://github.com/mlcommons/inference.git mlperf_inference
cd mlperf_inference
cp ./mlperf.conf ${DIR}/. 

git submodule update --init --recursive third_party/pybind/
cd loadgen
mkdir build && cd build
cmake ..
make
# Build python lib
cd ..
CFLAGS="-std=c++14" python setup.py install

cd ${WORKDIR}

# Build torchvision
echo "Installiing torch vision"
git clone https://github.com/pytorch/vision
cd vision
git checkout 8e078971b8aebdeb1746fea58851e3754f103053
python setup.py install
cd ${WORKDIR}

# Build OpenCV
git clone https://github.com/opencv/opencv.git
cd opencv
git checkout 4.x
mkdir build && cd build
cmake -DBUILD_TESTS=OFF -DBUILD_PERF_TESTS=OFF -DBUILD_EXAMPLES=OFF -DBUILD_opencv_apps=OFF -DBUILD_LIST=highgui,imgcodecs,imgproc ..
make -j$(nproc)

cd ${WORKDIR}

# Download rapidjson headers
git clone https://github.com/Tencent/rapidjson.git
cd rapidjson
git checkout e4bde977

cd ${WORKDIR}

# Build Gflags
git clone https://github.com/gflags/gflags.git
cd gflags
mkdir build && cd build
cmake ..
make


