To run this benchmark, first follow the setup steps in `closed/Dell/README.md`. Then to generate the TensorRT engines and run the harness:

```
make generate_engines RUN_ARGS="--benchmarks=rnnt --scenarios=Server"
make run_harness RUN_ARGS="--benchmarks=rnnt --scenarios=Server --test_mode=AccuracyOnly"
make run_harness RUN_ARGS="--benchmarks=rnnt --scenarios=Server --test_mode=PerformanceOnly"
```

For more details, please refer to `closed/Dell/README.md`.