# Habana MLPerf™ inference submission
MLPerf™ is a trademark and service mark of MLCommons Association in the United States and other countries.\
All rights reserved. Unauthorized use is strictly prohibited.

## Setup

### Install firmware, driver, SynapseAI 1.12.98
Please follow the instructions provided in the [Gaudi Installation Guide](https://docs.habana.ai/en/latest/Installation_Guide/index.html) to set up the environment.

### Prepare HabanaLabs MLPerf inference container

```bash
mkdir -p /path/to/Habana
export HABANA_DIR=/path/to/Habana
```

This README is located in [code](./) directory corresponding to Habana's submission. Download the whole [code](./) folder along with all subfolders and copy it under $HABANA_DIR

```bash
docker run --privileged --security-opt seccomp=unconfined \
           --name mlperf-habana -td                \
           -v /dev:/dev                            \
           --device=/dev:/dev                      \
           -v /sys/kernel/debug:/sys/kernel/debug  \
           -v /tmp:/tmp                            \
           -v $HABANA_DIR:/root/Habana/            \
           --cap-add=sys_nice --cap-add=SYS_PTRACE \
           --user root --workdir=/root --net=host  \
           --ulimit memlock=-1:-1 vault.habana.ai/gaudi-docker-mlperf/ver3.1/pytorch-installer-2.0.1:1.12.98-44
```
```bash
docker exec -it mlperf-habana bash
```
### Download checkpoint
```bash
mkdir -p /mnt/weka/data/pytorch/gpt-j/
pushd /mnt/weka/data/pytorch/gpt-j/
wget https://cloud.mlcommons.org/index.php/s/QAZ2oM94MkFtbQx/download  --output-document checkpoint.zip
unzip -q checkpoint.zip && rm checkpoint.zip
popd
```

### Download dataset
```bash
pushd /root/Habana/code/gptj-99.9/gpt-j
python download_cnndm.py
cp data/cnn_eval.json /mnt/weka/data/pytorch/gpt-j/cnn_eval.json
popd
```

##  Reproduce results
### 99 and 99.9 accuracy
For both benchmarks (99 & 99.9) we submitted the same script - no additional improvements for low accuracy (99) were applied and results from the 99.9 benchmark are used for 99 as well.

### Get started
Install the requirements and build latest loadgen.

```bash
cd /root/Habana/code
source gptj-99.9/functions.sh
build_mlperf_inference
```
### Generate results
**To generate full submission results use the following command**
```bash
build_mlperf_inference --output-dir <path_to_output_dir> --submission gptj-99.9-fp8
```
The command produces results from accuracy and performance runs for both Offline and Server scenarios.
Logs can be found under /output_dir/logs/model/, e.g. /results/logs/gptj-99.9-fp8/


**To generate results for Offline and Server scenarios separately use the following commands**
```bash
source functions.sh
build_mlperf_inference --output-dir <path_to_output_dir> --submission gptj-99.9-fp8_Offline
```

```bash
source functions.sh
build_mlperf_inference --output-dir <path_to_output_dir> --submission gptj-99.9-fp8_Server
```
Logs can be found under /output_dir/logs/model/scenario/, e.g. /results/logs/gptj-99.9-fp8/Offline/

**To generate results for accuracy and performance separately add ```--mode``` flag as in one of the following commands**
```bash
source functions.sh
build_mlperf_inference --output-dir <path_to_output_dir> --submission gptj-99.9-fp8_Server --mode acc
```
```bash
source functions.sh
build_mlperf_inference --output-dir <path_to_output_dir> --submission gptj-99.9-fp8_Offline --mode perf
```

Logs can be found under /output_dir/logs/model/scenario/mode/, e.g. /results/logs/gptj-99.9-fp8/Offline/accuracy/

## FP8 flow
As a performance optimization, we are setting heavy-performance ops to operate in fp8-143.

All fp8 ops are working with a fixed fp8 exponent bias = 7 and no scaling is required.

### Environment variables
- ENV var: PT_USE_FP8_143=1\
Effect: Set PT backend fp8 flavor to fp8_143

- ENV var: ENABLE_CALC_DYNAMIC_RANGE=false\
Effect: Disable heavy calculation done in warmup stage, as the quantization info relevant to fp8 tensors is set to default value.

- ENV var: PROPAGATE_FP8_CASTS=true\
Effect: Performance optimization, allowing additional fused operations on TPC and MME

- ENV var: UPDATE_MME_OUTPUT_PRECISION_FILTER="v_proj,matmul_av"\
Effect: Performance optimization, allowing the specified MME layer to output fp8

- ENV var: USE_DEFAULT_QUANT_PARAM=true\
Effect:  Set default quantization info for fp8 operations. The default quantization info is exponentBias = 7, per fp8 tensor.

- ENV var: UPDATE_GRAPH_OUTPUT_MME=false\
Effect: Set MME that produces model output (lm_head) to bf16 precision.
In practice it doesn't affect GPT-J model since the model output isn't produced by MME engine.

- ENV var: SCALES_FILE_PATH=quantization/measurements/per_tensor_scales_gpt_j.json\
Effect: Load per-tensor scales required for fp8 quantization. If not provided, no scaling is applied. 

- ENV var: ENABLE_EXPERIMENTAL_FLAGS=true\
Effect: Enable above flags