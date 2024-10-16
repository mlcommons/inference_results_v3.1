# MLPerf Inference - Language Models - KILT
This implementation runs language models with the KILT backend using the OnnxRT API on an Nvidia GPU.

Currently it supports the following models:
- bert-99

# Setting up your environment

First, install axs.

Install axs in `/local/mnt/workspace/$USER`.
```
git clone --branch master https://github.com/krai/axs ~/axs
```

Add the path to `bashrc`.
```
echo "export PATH='$PATH:$HOME/axs'" >> ~/.bashrc
```

And source it.
```
source ~/.bashrc
```

Upon sucessful installation.
```
user@laptop:~/axs$ axs
DefaultKernel{}
```

| Step | Description | Command |
| --- | --- | --- |
| 1 | Start with a clean work_collection | <pre><code>axs byname work_collection , remove</code></pre> |
| 2a | Import this repo and config into your work_collection using SSH (Private repos) `axs2qaic-dev` | <pre><code>axs byquery git_repo,collection,repo_name=axs2qaic-dev,url=git@github.com:krai/axs2qaic-dev.git</code></pre> |
|   | `axs2kilt-dev`    | <pre><code>axs byquery git_repo,collection,repo_name=axs2kilt-dev,url=git@github.com:krai/axs2kilt-dev.git</code></pre> |
|   | `axs2onnxrt-dev`  | <pre><code>axs byquery git_repo,collection,repo_name=axs2onnxrt-dev,url=git@github.com:krai/axs2onnxrt-dev.git</code></pre> |
| 2b | Alternatively, import this repo and config into your work_collection using HTTPS (Private repos) | <pre><code>axs byquery git_repo,collection,repo_name=axs2qaic-dev</code></pre> |
|   | | <pre><code>axs byquery git_repo,collection,repo_name=axs2kilt-dev</code></pre> |
|   | `axs2onnxrt-dev`     | <pre><code>axs byquery git_repo,collection,repo_name=axs2cpu-dev</code></pre> |
| 3 | Import other necessary repos (`axs2mlperf`) into your work_collection | <pre><code>axs byquery git_repo,collection,repo_name=axs2mlperf</code></pre> |
|   | `kilt-mlperf-dev` | <pre><code>axs byquery git_repo,repo_name=kilt-mlperf-dev,url=git@github.com:krai/kilt-mlperf-dev.git</code></pre> |
| 4 | Set Python version for compatibility | <pre><code>ln -s /usr/bin/python3.9 $HOME/bin/python3</code></pre> |
| 5 | Set Python version in axs | <pre><code>axs byquery shell_tool,can_python</code></pre> |
| 6 | Get and extract onnxrt library | <pre><code>axs byquery extracted,onnxruntime_lib</code></pre> |

# BERT-99 Commands

### Downloading BERT dependencies

| Step | Description | Command |
| --- | --- | --- |
| 1 | Compile protobuf | <pre><code>axs byquery compiled,protobuf</code></pre> |
| 2 | Compile the program binary | <pre><code>axs byquery compiled,kilt_executable,bert,device=onnxrt</code></pre> |
| 3 | Download SQuad dataset, both variants | <pre><code>axs byquery tokenized,squad_v1_1,calibration=no && axs byquery tokenized,squad_v1_1,calibration=yes</code></pre> |
| 4 | Download original base model | <pre><code>axs byquery onnx_conversion_ready,tf_model,model_name=bert_large</code></pre> |
| 5 | Convert original model to input-packed onnx model | <pre><code>axs byquery quant_ready,onnx_model,packed,model_name=bert_large</code></pre> |

### Benchmarking bert-99

| Step | Description | Command |
| --- | --- | --- |
| 1 | Set backend to either CPU or GPU | <pre><code>export BACKEND=<cpu / gpu></code></pre> |
| 2 | Set sut_name according to CPU/GPU | <pre><code>export SUT=<7920t-kilt-onnxruntime_cpu / 7920t-kilt-onnxruntime_gpu></code></pre>
| 2 | Measure Accuracy (Quick Run) | <pre><code>axs byquery sut_name=${SUT},loadgen_output,bert_squad,device=onnxrt,backend_type=${BACKEND},framework=kilt,model_name=bert-99,loadgen_scenario=SingleStream,loadgen_mode=AccuracyOnly , get accuracy_report</code></pre> |
| 3 | Measure Accuracy (Full Run) | <pre><code>axs byquery sut_name=${SUT},loadgen_output,bert_squad,device=onnxrt,backend_type=${BACKEND},framework=kilt,model_name=bert-99.9,loadgen_scenario=SingleStream,loadgen_mode=AccuracyOnly,loadgen_dataset_size=10833,loadgen_buffer_size=10833 , get accuracy_report</code></pre> |
| 4 | Run Performance (Quick Run) | <pre><code>axs byquery sut_name=${SUT},loadgen_output,bert_squad,device=onnxrt,backend_type=${BACKEND},framework=kilt,model_name=bert-99,loadgen_scenario=SingleStream,loadgen_mode=PerformanceOnly,loadgen_target_latency=1000 , parse_summary</code></pre> |
| 5 | Run Performance (Full Run) | <pre><code>axs byquery sut_name=${SUT},loadgen_output,bert_squad,device=onnxrt,backend_type=${BACKEND},framework=kilt,model_name=bert-99,loadgen_scenario=SingleStream,loadgen_mode=PerformanceOnly,loadgen_dataset_size=10833,loadgen_buffer_size=10833,loadgen_target_latency=</measured value/> , parse_summary</code></pre> |

