#!/bin/bash
# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

DATA_DIR=${DATA_DIR:-/work/build/data}

# Make sure the script is executed inside the container
if [ -e /work/code/gptj/tensorrt/download_data.sh ]
then
    echo "Inside container, start downloading..."
    if [ -e build/inference/language/gpt-j/download_cnndm.py ]
    then
        DATASET_CNNDM_PATH=${DATA_DIR}/cnn-daily-mail python3 build/inference/language/gpt-j/download_cnndm.py
    else
        echo "ERROR: Inference repo is out-dated, please run make clone_loadgen to update. Exiting..."
        exit 1
    fi
else
    echo "WARNING: Please enter the MLPerf container (make prebuild) before downloading gptj6b dataset"
    echo "WARNING: gptj6b dataset is NOT downloaded! Exiting..."
    exit 0
fi
