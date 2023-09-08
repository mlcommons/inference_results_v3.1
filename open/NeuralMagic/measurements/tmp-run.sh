#!/bin/bash

export PYTHONPATH="/home/arjun/CM/repos/local/cache/4b83152968a54108/inference/vision/classification_and_detection/python:/home/arjun/CM/repos/local/cache/4b83152968a54108/inference/tools/submission:${PYTHONPATH}"
export CM_INPUT="/home/arjun/nm/neural_magic"
export CM_MLPERF_INFERENCE_3DUNET_PATH="/home/arjun/CM/repos/local/cache/4b83152968a54108/inference/vision/medical_imaging/3d-unet-kits19"
export CM_MLPERF_INFERENCE_BERT_PATH="/home/arjun/CM/repos/local/cache/4b83152968a54108/inference/language/bert"
export CM_MLPERF_INFERENCE_CLASSIFICATION_AND_DETECTION_PATH="/home/arjun/CM/repos/local/cache/4b83152968a54108/inference/vision/classification_and_detection"
export CM_MLPERF_INFERENCE_CONF_PATH="/home/arjun/CM/repos/local/cache/4b83152968a54108/inference/mlperf.conf"
export CM_MLPERF_INFERENCE_DLRM_PATH="/home/arjun/CM/repos/local/cache/4b83152968a54108/inference/recommendation/dlrm"
export CM_MLPERF_INFERENCE_DLRM_V2_PATH="/home/arjun/CM/repos/local/cache/4b83152968a54108/inference/recommendation/dlrm_v2"
export CM_MLPERF_INFERENCE_GPTJ_PATH="/home/arjun/CM/repos/local/cache/4b83152968a54108/inference/language/gpt-j"
export CM_MLPERF_INFERENCE_RNNT_PATH="/home/arjun/CM/repos/local/cache/4b83152968a54108/inference/speech_recognition/rnnt"
export CM_MLPERF_INFERENCE_SOURCE="/home/arjun/CM/repos/local/cache/4b83152968a54108/inference"
export CM_MLPERF_INFERENCE_VISION_PATH="/home/arjun/CM/repos/local/cache/4b83152968a54108/inference/inference/vision"
export CM_MLPERF_LAST_RELEASE="v3.1"
export CM_MLPERF_SHORT_RUN="no"
export CM_MLPERF_SUBMISSION_CHECKER_EXTRA_ARGS="--skip-extra-files-in-root-check"
export CM_MLPERF_SUBMISSION_DIR="/home/arjun/nm/neural_magic"
export CM_PANDAS_VERSION="1.5.3"
export CM_POST_RUN_CMD="/usr/bin/python3 /home/arjun/CM/repos/local/cache/4b83152968a54108/inference/tools/submission/generate_final_report.py --input summary.csv"
export CM_PYTHONLIB_BOTO3_CACHE_TAGS="version-1.26.74"
export CM_PYTHONLIB_NUMPY_CACHE_TAGS="version-1.24.2"
export CM_PYTHONLIB_OPENCV_PYTHON_CACHE_TAGS="version-4.7.0.68"
export CM_PYTHONLIB_PANDAS_CACHE_TAGS="version-1.5.3"
export CM_PYTHONLIB_REQUESTS_CACHE_TAGS="version-2.25.1"
export CM_PYTHONLIB_TQDM_CACHE_TAGS="version-4.64.1"
export CM_PYTHONLIB_XLSXWRITER_CACHE_TAGS="version-3.0.8"
export CM_PYTHON_BIN="python3"
export CM_PYTHON_BIN_PATH="/usr/bin"
export CM_PYTHON_BIN_WITH_PATH="/usr/bin/python3"
export CM_PYTHON_CACHE_TAGS="version-3.10.6,non-virtual"
export CM_PYTHON_VERSION="3.10.6"
export CM_RUN_CMD="/usr/bin/python3 /home/arjun/CM/repos/local/cache/4b83152968a54108/inference/tools/submission/submission_checker.py --input "/home/arjun/nm/neural_magic" --skip-extra-files-in-root-check"
export CM_TMP_CURRENT_PATH="/home/arjun/nm/neural_magic/open/NeuralMagic/measurements"
export CM_TMP_CURRENT_SCRIPT_PATH="/home/arjun/CM/repos/ctuning@mlcommons-ck/cm-mlops/script/run-mlperf-inference-submission-checker"
export CM_TMP_PIP_VERSION_STRING="==master"
export CM_VERSION="master"
export CM_XLSXWRITER_VERSION="3.0.8"


. "/home/arjun/CM/repos/ctuning@mlcommons-ck/cm-mlops/script/run-mlperf-inference-submission-checker/run.sh"