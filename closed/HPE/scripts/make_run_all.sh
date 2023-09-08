########## Run all models, all performance tasks #############
cd /work #container's home dir
SUBMITTER="HPE"

### make run (generate engines and run performance together, this rebuilds any existing engines)
#make run SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=resnet50 --scenarios=offline,server"
#make run SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=retinanet --scenarios=offline,server"
#make run SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=3d-unet --scenarios=offline --config_ver=default,high_accuracy"
#make run SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=bert --scenarios=offline,server --config_ver=default,high_accuracy"
#make run SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=gptj --scenarios=offline,server --config_ver=default,high_accuracy"
##make run SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=rnnt --scenarios=offline,server"
##make run SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=dlrm-v2 --scenarios=offline,server --config_ver=default,high_accuracy"


########## Alternative run individual tasks #############

### make generate_engine binaries only (rebuilds any existing engines)
make generate_engines SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=resnet50 --scenarios=offline,server"
make generate_engines SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=retinanet --scenarios=offline,server"
make generate_engines SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=3d-unet --scenarios=offline --config_ver=default,high_accuracy"
make generate_engines SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=bert --scenarios=offline,server --config_ver=default,high_accuracy"
make generate_engines SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=gptj --scenarios=offline,server --config_ver=default,high_accuracy"
#make generate_engines SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=rnnt --scenarios=offline,server"
#make generate_engines SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=dlrm-v2 --scenarios=offline,server --config_ver=default,high_accuracy"

## run_harness (get performance results)
## add the `--fast` option on each line to change from 10min official run to 1min quick checks
make run_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=resnet50 --scenarios=offline,server --test_mode=PerformanceOnly"
make run_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=retinanet --scenarios=offline,server --test_mode=PerformanceOnly"
make run_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=3d-unet --scenarios=offline --config_ver=default,high_accuracy --test_mode=PerformanceOnly"
make run_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=bert --scenarios=offline,server --config_ver=default,high_accuracy --test_mode=PerformanceOnly"
make run_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=gptj --scenarios=offline,server --config_ver=default,high_accuracy --test_mode=PerformanceOnly"
#make run_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=rnnt --scenarios=offline,server --test_mode=PerformanceOnly"
#make run_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=dlrm-v2 --scenarios=offline,server --config_ver=default,high_accuracy --test_mode=PerformanceOnly"

# run_harness (performance with accuracy checks)
make run_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=resnet50 --scenarios=offline,server --test_mode=AccuracyOnly"
make run_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=retinanet --scenarios=offline,server --test_mode=AccuracyOnly"
make run_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=3d-unet --scenarios=offline --config_ver=default,high_accuracy --test_mode=AccuracyOnly"
make run_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=bert --scenarios=offline,server --config_ver=default,high_accuracy --test_mode=AccuracyOnly"
make run_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=gptj --scenarios=offline,server --config_ver=default,high_accuracy --test_mode=AccuracyOnly"
#make run_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=rnnt --scenarios=offline,server --test_mode=AccuracyOnly"
#make run_harness SUBMITTER=$SUBMITTER RUN_ARGS="--benchmarks=dlrm-v2 --scenarios=offline,server --config_ver=default,high_accuracy --test_mode=AccuracyOnly"

