# MLPerf Inference - Bert Squad - KILT

To run the experiments you need the following commands

## Benchmarking bert-99 model in Performance mode
```
axs byquery loadgen_output,bert_squad,framework=kilt,loadgen_scenario=Server,loadgen_mode=PerformanceOnly,model_name=bert-99,loadgen_dataset_size=10833,loadgen_buffer_size=10833,loadgen_target_qps=3400
```

