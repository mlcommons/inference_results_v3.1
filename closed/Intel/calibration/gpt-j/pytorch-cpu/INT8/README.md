## Setup Instructions
### Requires
+ gcc-11
```
   dnf install -y gcc-toolset-11-gcc gcc-toolset-11-gcc-c++
```

### Anaconda and Conda Environment
+ Download and install conda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
```
+ Setup conda environment & dependencies
```
source </path/to/gcc-11/>enable
bash prepare_calibration_env.sh gpt-j-calibration-env
conda activate gpt-j-calibration-env
```

### Download and Prepare Dataset
```
CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data
mkdir -p ${WORKLOAD_DATA}
```

+ Download cnn-dailymail calibration set
```
python download-calibration-dataset.py --calibration-list-file calibration-list.txt --output-dir ${WORKLOAD_DATA}/calibration-data
```

+ Download cnn-dailymail validation set
```
python download-dataset.py --split validation --output-dir ${WORKLOAD_DATA}/validation-data
```

### Download and prepare model
+ Get finetuned checkpoint
```
CHECKPOINT_DIR=${WORKLOAD_DATA}/gpt-j-checkpoint
wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download -O gpt-j-checkpoint.zip
unzip gpt-j-checkpoint.zip
mv gpt-j/checkpoint-final/ ${CHECKPOINT_DIR}
```
+ Generate quantized INT8 model
```
source setup_env.sh
bash run_quantization.sh
```
