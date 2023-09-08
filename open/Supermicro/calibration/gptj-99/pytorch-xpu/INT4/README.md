# BKC for MLPerf GPT-J Calibration on PVC

## HW & SW requirements
###
| Compent | Version |
|  -  | -  |
| OS | Ubuntu 9.4.0-1ubuntu1~20.04.1 |
| Driver | hotfix_agama-ci-devel-627.7 |
| GCC | 11.3.0 |
| Intel(R) oneAPI DPC++/C++ Compiler | 2023.1.0 (2023.1.0.20230320) |

## Steps to calibrate GPT-J
### 1. Install anaconda 3.0
```
  wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ~/anaconda3.sh -b -p ~/anaconda3
  export PATH=~/anaconda3/bin:$PATH
```

### 2. Prepare calibration environement
```
  conda env create -f quantization-env.yaml
```

### 3. Prepare calibration Dataset
```
  WORK_DIR=$(pwd)
  DATA_DIR=${WORK_DIR}/data
  mkdir -p ${DATA_DIR}
  python download-calibration-dataset.py --calibration-list-file calibration-list.txt --output-dir ${DATA_DIR}
```

### 4. Prepare fine-tuned Model
```
  MODEL_DIR=${WORK_DIR}/model
  wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download --output-document checkpoint.zip
  unzip checkpoint.zip
  mv gpt-j/checkpoint-final ${MODEL_DIR}
  rm -rf gpt-j
```

### 5. Prepare quantization scripts
```
  git clone https://github.com/qwopqwop200/GPTQ-for-LLaMa.git
  cp run_quantization.sh cnn_dm_dataset.py GPTQ-for-LLaMa/
  cd GPTQ-for-LLaMa
  git apply ../gptj.patch
```

### 6. Quantize Model
```
  source activate gpt-j-quant-env
  DATA_DIR=${DATA_DIR} MODEL_DIR=${MODEL_DIR} run_quantization.sh
```

* Quantized model will be saved in `${MODEL_DIR}`
