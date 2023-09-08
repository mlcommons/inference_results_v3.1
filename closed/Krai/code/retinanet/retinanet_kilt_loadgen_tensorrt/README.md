# MLPerf Inference - Object Detection - KILT
This implementation runs object detection models with the KILT backend using TensorRT API on an Nvidia GPU.
The inference is run on serialized engines generated with the NVidia toolset (details below).

Currently it supports the following models:
- retinanet

## Setting up your environment
Start with a clean work_collection
```
axs byname work_collection , remove
```

Import this repos into your work_collection
```
axs byquery git_repo,collection,repo_name=axs2tensorrt
axs byquery git_repo,collection,repo_name=axs2kilt
axs byquery git_repo,collection,repo_name=axs2tensorrt
axs byquery git_repo,collection,repo_name=axs2kilt
axs byquery git_repo,collection,repo_name=axs2mlperf
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

## Downloading Retinanet dependencies


Compile the program binary
```
axs byquery compiled,kilt_executable,retinanet,device=tensorrt
```

Download `openimages-mlperf.json`
```
axs byquery inference_ready,openimages_annotations,v2_1
```

Download Openimages Datasets Validation
```
axs byquery downloaded,openimages_mlperf,validation+
```


Download Calibration Openimages Datasets
```
axs byquery openimages_mlperf,calibration
```

Preprocess Calibration Datasets 
```
axs byquery preprocessed,dataset_name=openimages,preprocess_method=pillow_torch,index_file=openimages_cal_images_list.txt,calibration+
```

Preprocess Full Openimages Datasets
```
axs byquery preprocessed,dataset_name=openimages,preprocess_method=pillow_torch,first_n=24781,quantized+
```

Download Original Model
```
axs byquery downloaded,onnx_model,model_name=retinanet,no_nms
```

Set up a docker container for running NVidia submissions (https://github.com/mlcommons/inference_results_v3.0/tree/main/closed/NVIDIA). And use it to generate engines with provided custom configs, that are available at axs2kilt/retinanet_kilt_loadgen_tensorrt/config/. Before generating engines ensure that the TensorRT version inside the container is the same as shown in the requrements (8.6.1).

Copy the resulting engines (one for the Offline scenatio, anothe one - for the Server scenario) into your work_collection.
```
scp ... ~/work_collection/tensorrt_bert_model
```

Copy the plugin libnmsoptplugin.so from inside the docker container at a path <pre><code>build/plugins/RNNTOptPlugin/</code></pre> into <pre><code>../kilt-mlperf-dev/plugins/</code></pre> or any other place. Provide this path as additional argument for below commands as follows: <pre><code>,plugins_path=path</code></pre>

## Benchmarking Retinanet


Measure Accuracy  
```
axs byquery sut_name=7920t-kilt-tensorrt,loadgen_output,object_detection,device=tensorrt,framework=kilt,model_name=retinanet,loadgen_scenario=Offline,loadgen_mode=AccuracyOnly , get accuracy
```

Run Performance (Quick Run)
```
axs byquery sut_name=7920t-kilt-tensorrt,loadgen_output,object_detection,device=tensorrt,framework=kilt,model_name=retinanet,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=1 , parse_summary
```

Run Performance (Full Run)
```
axs byquery sut_name=7920t-kilt-tensorrt,loadgen_output,object_detection,device=tensorrt,framework=kilt,model_name=retinanet,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,loadgen_target_qps=<measured value> , parse_summary
```
