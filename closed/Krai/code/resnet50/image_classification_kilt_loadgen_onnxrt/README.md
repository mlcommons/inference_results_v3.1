# MLPerf Inference - Image Classification - KILT
This implementation runs image classification models with the KILT backend using the OnnxRT API on an Nvidia GPU.

Currently it supports the following models:
- resnet50

## Setting up your environment
Start with a clean work_collection
```
axs byname work_collection , remove
```

Import these repos into your work_collection
```
axs byquery git_repo,collection,repo_name=axs2kilt
axs byquery git_repo,collection,repo_name=axs2onnxrt
axs byquery git_repo,collection,repo_name=axs2mlperf
axs byquery git_repo,collection,repo_name=axs2config
axs byquery git_repo,repo_name=kilt-mlperf
```

Set Python version for compatibility
```
ln -s /usr/bin/python3.9 $HOME/bin/python3
```

Set Python version in axs 
```
axs byquery shell_tool,can_python
```

Get and extract onnxrt library
```
axs byquery extracted,onnxruntime_lib
```

## Downloading Resnet dependencies

Download full dataset
```
axs byname extractor , extract --archive_path=/datasets/dataset-imagenet-ilsvrc2012-val.tar --tags,=extracted,imagenet --strip_components=1 --dataset_size=50000
```

Compile the program binary
```
axs byquery compiled,kilt_executable,resnet50,device=onnxrt
```

Download the converted onnx model
```
axs byquery onnx_model,converted,model_name=resnet50
```

## Benchmarking resnet50

Set backend to either CPU or GPU
```
export BACKEND=<cpu | gpu >
```

Set sut_name according to CPU/GPU
```
export SUT=<7920t-kilt-onnxruntime_cpu | 7920t-kilt-onnxruntime_gpu>
```

Set scenario
```
export SCENARIO=<SingleStream | MultiStream>
```

Measure Accuracy (Quick Run)
```
axs byquery sut_name=${SUT},loadgen_output,image_classifier,device=onnxrt,backend_type=${BACKEND},loadgen_scenario=${SCENARIO},framework=kilt,model_name=resnet50,loadgen_mode=AccuracyOnly , get accuracy
```

Measure Accuracy (Full Run)
```
axs byquery sut_name=${SUT},loadgen_output,image_classifier,device=onnxrt,backend_type=${BACKEND},loadgen_scenario=${SCENARIO},framework=kilt,model_name=resnet50,loadgen_mode=AccuracyOnly,loadgen_dataset_size=50000,loadgen_buffer_size=1024 , get accuracy
```

Run Performance (Quick Run)
```
axs byquery sut_name=${SUT},loadgen_output,image_classifier,device=onnxrt,backend_type=${BACKEND},loadgen_scenario=${SCENARIO},framework=kilt,model_name=resnet50,loadgen_mode=PerformanceOnly,loadgen_target_latency=1000 , parse_summary
```

Run Performance (Full Run)
```
axs byquery sut_name=${SUT},loadgen_output,image_classifier,device=onnxrt,backend_type=${BACKEND},loadgen_scenario=${SCENARIO},framework=kilt,model_name=resnet50,loadgen_mode=PerformanceOnly,loadgen_dataset_size=50000,loadgen_buffer_size=1024,loadgen_target_latency=<measured value> , parse_summary
```
