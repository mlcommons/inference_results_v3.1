# BKC for MLPerf GPT-J Inference on PVC

## HW & SW requirements
###
| Compent | Version |
|  -  | -  |
| OS | Ubuntu 9.4.0-1ubuntu1~20.04.1 |
| Driver | hotfix_agama-ci-devel-627.7 |
| GCC | 11.3.0 |
| Intel(R) oneAPI DPC++/C++ Compiler | 2023.1.0 (2023.1.0.20230320) |

## Steps to run GPT-J
### 1. Install anaconda 3.0
```
  wget https://repo.continuum.io/archive/Anaconda3-5.0.0-Linux-x86_64.sh -O anaconda3.sh
  chmod +x anaconda3.sh
  ~/anaconda3.sh -b -p ~/anaconda3
  export PATH=~/anaconda3/bin:$PATH
```
### 2. Create conda environement
```
  CONDA_ENV=gptj-infer
  conda create -y -n ${CONDA_ENV} python=3.9
  source activate ${CONDA_ENV}
```

### 3. Execute `run.sh`. The end-to-end process including:
| STAGE(default -1) | STEP |
|  -  | -  |
| -2 | Prepare conda environment |
| -1 | Prepare environment |
| 0 | Download model |
| 1 | Download dataset |
| 2 | Calibrate model |
| 3 | Run Offline/Server accuracy & benchmark |

You can also use the following command to start with your custom conda-env/work-dir/step.
```
  [CONDA_ENV=<env-name>] [WORK_DIR=<pwd>] [STAGE=<stage-num>] [USE_INT4=<true>] bash run.sh
```
* Before benchmarking, please use `bash run_clean.sh` with sudo access to set hardware configurations.
