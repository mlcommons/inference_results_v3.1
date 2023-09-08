# C++ SUT for DLRM inference

## How to compile
### 1. Install Intel oneAPI compiler
```bash
wget https://registrationcenter-download.intel.com/akdlm/irc_nas/19084/l_HPCKit_p_2023.0.0.25400.sh
sudo bash ./l_HPCKit_p_2023.0.0.25400.sh
```

### 2. create conda environment
```bash
export WORKDIR=$PWD

source ~/anaconda3/bin/activate
conda create -n dlrm python=3.9
conda update -n base -c defaults conda --yes
conda activate dlrm
```

### 3. import oneapi compiler
```bash
source /opt/intel/oneapi/compiler/2022.1.0/env/vars.sh
```

### 4. loadgen
```bash
cd ${WORKDIR}
git clone https://github.com/mlcommons/inference.git
cd inference/loadgen
mkdir build
cd build
CC=icx CXX=icpx cmake .. -DCMAKE_INSTALL_PREFIX=${WORKDIR}/loadgen
make -j && make install
```

### 5. oneDNN
```bash
cd ${WORKDIR}
git clone <ONEDNN_REPO>
cd libraries.performance.math.onednn/
git checkout v2.7
mkdir build
cd build
CC=icx CXX=icpx cmake .. -DCMAKE_INSTALL_PREFIX=${WORKDIR}/spronednn
make -j && make install
```


### 6. C++ SUT
```bash
cd ${WORKDIR}
git clone <WORKLOAD_REPO>
ln -s frameworks.ai.benchmarking.mlperf.develop.inference-datacenter/closed/Intel/code/dlrm-99.9/pytorch-cpu pytorch-cpu
cd pytorch-cpu/src
mkdir build
cd build
export CMAKE_PREFIX_PATH=$CMAKE_PREFIX_PATH:${WORKDIR}/spronednn
CC=icx CXX=icpx cmake .. -DLOADGEN_DIR=${WORKDIR}/loadgen -DONEDNN_DIR=${WORKDIR}/spronednn
make -j
```

### 7. prepare dataset and model
```bash
# download dataset
Create a directory (such as ${WORKDIR}/dataset/terabyte_input) which contain:
	day_fea_count.npz
	terabyte_processed_test.bin

About how to get the dataset, please refer to
	https://github.com/facebookresearch/dlrm

# download model
# Create a directory (such as ${WORKDIR}/dataset/model):
cd ${WORKDIR}/dataset/model
wget https://dlrm.s3-us-west-1.amazonaws.com/models/tb00_40M.pt -O dlrm_terabyte.pytorch
```

### 8. preprocess dataset and model
```bash
cd ${WORKDIR}/pytorch-cpu

export MODEL=<model_dir>	# such as ${WORKDIR}/dataset/model
export DATASET=<dataset_dir>	# such as ${WORKDIR}/dataset/terabyte_input
export DUMP_PATH=<dump_out_dir>

bash dump_model_dataset.sh

cp src/sample_partition.npy ${DUMP_PATH}
```

### 9. run
```bash
cd ${WORKDIR}/pytorch-cpu

# Note: please modify sciprt dataset and model path according to your real path

# Performance mode
bash runcppsut					# offline mode
bash runcppsut performance server		# server mode 
# Accuracy mode
bash runcppsut accuracy	# offline mode
bash runcppsut accuracy server

```

## Docker
### build docker and inference in docker
```bash
# build docker image
git clone <WORKLOAD_REPO>
ln -s frameworks.ai.benchmarking.mlperf.develop.inference-datacenter/closed/Intel/code/dlrm-99.9/pytorch-cpu pytorch-cpu
cd pytorch-cpu/src/docker
bash build_dlrm-99.9_container.sh

# activate container
docker run --privileged --name intel_inference_dlrm -itd --net=host --ipc=host -v /data/dlrm:/data/mlperf_data/dlrm  mlperf_inference_dlrm:3.0
docker exec -it intel_inference_dlrm bash

# preprocess model and dataset
cd ${HOME}/pytorch-cpu

export MODEL=<model_dir>	# such as /data/mlperf_dataset/model
export DATASET=<dataset_dir>	# such as /data/mlperf_dataset/terabyte_input
export DUMP_PATH=<dump_out_dir>

bash dump_model_dataset.sh

cp src/sample_partition.npy ${DUMP_PATH}

# inference in docker
cd ~/pytorch-cpu
# Note: please modify sciprt dataset and model path according to your real path
# Performance mode
bash runcppsut					# offline mode
bash runcppsut performance server		# server mode 
# Accuracy mode
bash runcppsut accuracy	# offline mode
bash runcppsut accuracy server
```

