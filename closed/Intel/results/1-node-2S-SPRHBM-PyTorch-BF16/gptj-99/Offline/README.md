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
bash prepare_env.sh gpt-j-env
conda activate gpt-j-env
```

### Download and Prepare Dataset
```
CUR_DIR=$(pwd)
export WORKLOAD_DATA=${CUR_DIR}/data
mkdir -p ${WORKLOAD_DATA}
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

### Run Benchmarks with Bfloat16
```
source setup_env.sh
```

+ Offline (Performance)
```
bash run_offline_bf16.sh
```

+ Offline (Accuracy)
```
bash run_offline_accuracy_bf16.sh
```

+ Server (Performance)
```
bash run_server_bf16.sh
```

+ Server (Accuracy)
```
bash run_server_accuracy_bf16.sh
```

