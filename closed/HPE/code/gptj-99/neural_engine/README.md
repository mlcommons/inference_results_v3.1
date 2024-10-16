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
+ Get neural_engine Repo
```
git clone https://github.com/intel-innersource/frameworks.ai.nlp-toolkit.intel-nlp-toolkit.git
cp -r ./frameworks.ai.nlp-toolkit.intel-nlp-toolkit/intel_extension_for_transformers/backends/neural_engine/graph .
cd graph
mkdir build
cd build
cmake .. -G Ninja
ninja 
cd ..
```
+ Get quantized model
```
python scripts/convert_gptj.py --outtype f32 --outfile  build/ne-f32.bin  ../data/gpt-j-checkpoint/

./build/bin/quant_llama --model_file  build/ne-f32.bin --out_file build/ne-q4_j.bin --bits 4

```
### Run Benchmarks
+ Offline (Performance)
```
bash run_offline.sh
```


+ Offline (Accuracy)
```
bash run_offline_accuracy.sh
```

+ Server (Performance)
```
bash run_server.sh
```

+ Server (Accuracy)
```
bash run_server_accuracy.sh
```


