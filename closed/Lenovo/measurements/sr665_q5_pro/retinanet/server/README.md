# MLPerf Inference - Object Detection - KILT

To run the experiments you need the following commands

## Benchmarking retinanet model in Performance mode
```
axs byquery loadgen_output,object_detection,framework=kilt,loadgen_scenario=Server,loadgen_mode=PerformanceOnly,model_name=retinanet,loadgen_dataset_size=24781,loadgen_buffer_size=64,loadgen_target_qps=1386
```

