To run this benchmark, first follow the setup steps in `closed/Nutanix/README.md`. Then to generate the TensorRT engines and run the harness:

```
make generate_engines RUN_ARGS="--benchmarks=resnet50 --scenarios=MultiStream"
make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=MultiStream --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=resnet50 --scenarios=MultiStream --test_mode=PerformanceOnly"
```

For more details, please refer to `closed/Nutanix/README.md`.