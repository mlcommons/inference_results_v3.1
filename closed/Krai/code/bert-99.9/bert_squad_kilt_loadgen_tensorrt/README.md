# MLPerf Inference - Language Models - KILT
This implementation runs language models with the KILT backend using TensorRT API on a Nvidia GPU.
The inference is run on serialized engines generated with the NVisia toolset (dateils below).

Currently it supports the following models:
- bert-99
- bert-99.9

## Setting up your environment

| Step | Description | Command |
| --- | --- | --- |
| 1 | Start with a clean work_collection | <pre><code>axs byname work_collection , remove</code></pre> |
| 2a | Import this repo and config into your work_collection using SSH `axs2tensorrt-dev` | <pre><code>axs byquery git_repo,collection,repo_name=axs2tensorrt-dev,url=git@github.com:krai/axs2tensorrt-dev.git</code></pre> |
|   | `axs2kilt-dev`    | <pre><code>axs byquery git_repo,collection,repo_name=axs2kilt-dev,url=git@github.com:krai/axs2kilt-dev.git</code></pre> |
| 2b | Import this repo and config into your work_collection using HTTPS | <pre><code>axs byquery git_repo,collection,repo_name=axs2tensorrt-dev</code></pre> |
|   | | <pre><code>axs byquery git_repo,collection,repo_name=axs2kilt-dev</code></pre> |
| 3 | Import other necessary repos (`axs2mlperf`) into your work_collection | <pre><code>axs byquery git_repo,collection,repo_name=axs2mlperf</code></pre> |
|   | `kilt-mlperf-dev` | <pre><code>axs byquery git_repo,repo_name=kilt-mlperf-dev,url=git@github.com:krai/kilt-mlperf-dev.git</code></pre> |
| 4 | Set Python version for compatibility | <pre><code>ln -s /usr/bin/python3.9 $HOME/bin/python3</code></pre> |
| 5 | Set Python version in axs | <pre><code>axs byquery shell_tool,can_python</code></pre> |

## Downloading BERT dependencies

| Step | Description | Command |
| --- | --- | --- |
| 1 | Compile protobuf | <pre><code>axs byquery compiled,protobuf</code></pre> |
| 2 | Compile the program binary | <pre><code>axs byquery compiled,kilt_executable,bert,device=tensorrt</code></pre> |
| 3 | Set up a docker container for running NVidia submissions (https://github.com/mlcommons/inference_results_v3.0/tree/main/closed/NVIDIA). And use it to generate engines with provided custom configs, that are available at axs2kilt/bert_squad_kilt_loadgen_tensorrt/config/. Before generating engine ensure that the TensorRT version inside the container is the same as shown in the requrements (8.6.1).| |
| 4 | Copy the resulting engines (one for the Offline scenatio, anothe one - for the Server scenario) into your work_collection. | <pre><code>scp ... ~/work_collection/tensorrt_bert_model/</code></pre> |

## Benchmarking BERT-99

| Step | Description | Command |
| --- | --- | --- |
| 1 | Measure Accuracy  | <pre><code>axs byquery sut_name=7920t-kilt-tensorrt,loadgen_output,bert_squad,device=tensorrt,framework=kilt,model_name=bert-99,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly , get accuracy_report</code></pre> |
| 2 | Run Performance (Quick Run) | <pre><code>axs byquery sut_name=7920t-kilt-tensorrt,loadgen_output,bert_squad,device=tensorrt,framework=kilt,model_name=bert-99,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=1</code></pre> |
| 4 | Run Performance (Full Run) | <pre><code>axs byquery sut_name=7920t-kilt-tensorrt,loadgen_output,bert_squad,device=tensorrt,framework=kilt,model_name=bert-99,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_dataset_size=10833,loadgen_buffer_size=10833,loadgen_target_qps=</measured value/></code></pre> |

---

## Benchmarking BERT-99.9

| Step | Description | Command |
| --- | --- | --- |
| 1 | Measure Accuracy | <pre><code>axs byquery sut_name=7920t-kilt-tensorrt,loadgen_output,bert_squad,device=tensorrt,framework=kilt,model_name=bert-99.9,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly , get accuracy_report</code></pre> |
| 2 | Run Performance (Quick Run) | <pre><code>axs byquery sut_name=7920t-kilt-tensorrt,loadgen_output,bert_squad,device=tensorrt,framework=kilt,model_name=bert-99.9,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=1</code></pre> |
| 3 | Run Performance (Full Run) | <pre><code>axs byquery sut_name=7920t-kilt-tensorrt,loadgen_output,bert_squad,device=tensorrt,framework=kilt,model_name=bert-99.9,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_dataset_size=10833,loadgen_buffer_size=10833,loadgen_target_qps=/<measured value/></code></pre> |

To see the results of a performance run, look at the last line in the console output, go to the output directory and open a file named "mlperf_log_summary.txt".
