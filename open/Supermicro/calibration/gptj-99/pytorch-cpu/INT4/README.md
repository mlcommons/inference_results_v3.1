## Setup Instructions

### Anaconda and Conda Environment
+ Download and install conda
```
wget https://repo.anaconda.com/miniconda/Miniconda3-latest-Linux-x86_64.sh -O miniconda.sh
bash miniconda.sh
```
+ Build conda environment
```
conda env create -f quantization-env.yaml
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

### Download and prepare model
+ Get finetuned checkpoint
```
CHECKPOINT_DIR=${WORKLOAD_DATA}/gpt-j-checkpoint
wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download -O gpt-j-checkpoint.zip
unzip gpt-j-checkpoint.zip
mv gpt-j/checkpoint-final/ ${CHECKPOINT_DIR}
```
+ Run Quantization
```
conda activate gpt-j-calibration-int4-env
source setup_env.sh
bash run_calibration_int4.sh
```

Calibrated weights (state dict) will be saved in `${WORKLOAD_DATA}/quantized-int4-model`

Please go to `$(pwd)/../../../../code/gptj-99/pytorch-cpu` to do quantization of model and benchmark.
