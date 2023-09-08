# MLPerf Inference - Image Classification - KILT

To run the experiments you need the following commands

## Benchmarking resnet50 model in Performance mode
```
axs byquery loadgen_output,image_classifier,framework=kilt,loadgen_scenario=Offline,loadgen_mode=PerformanceOnly,model_name=resnet50,loadgen_dataset_size=50000,loadgen_buffer_size=1024,loadgen_target_qps=40000
```

