To run this benchmark, first follow the setup steps in `closed/xFusion/README.md`. Then to generate the TensorRT engines and run the harness:

```
make generate_engines RUN_ARGS="--benchmarks=retinanet --scenarios=Offline"
make run_harness RUN_ARGS="--benchmarks=retinanet --scenarios=Offline --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=retinanet --scenarios=Offline --test_mode=PerformanceOnly"
```

For more details, please refer to `closed/xFusion/README.md`.