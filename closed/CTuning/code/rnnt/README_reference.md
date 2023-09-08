[ [Back to the common setup](README.md) ]


## Run this benchmark via CM


### Do a test run to detect and record the system performance

```
cmr "generate-run-cmds inference _find-performance _all-scenarios" \
--model=rnnt --implementation=reference --device=cpu --backend=pytorch \
--category=edge --division=open --quiet \
--adr.ml-engine-pytorch.version=1.13.0 --adr.ml-engine-torchvision.version=0.14.1

```
* Use `--device=cuda` to run the inference on Nvidia GPU
* Use `--division=closed` to run all scenarios for the closed division (compliance tests are skipped for `_find-performance` mode)
* Use `--category=datacenter` to run datacenter scenarios

### Do full accuracy and performance runs for all the scenarios

```
cmr "generate-run-cmds inference _submission _all-scenarios" --model=rnnt \
--device=cpu --implementation=reference --backend=pytorch \
--execution-mode=valid --results_dir=$HOME/results_dir \
--category=edge --division=open --quiet \
--adr.ml-engine-pytorch.version=1.13.0 --adr.ml-engine-torchvision.version=0.14.1
```

* Use `--power=yes` for measuring power. It is ignored for accuracy and compliance runs
* `--offline_target_qps`, `--server_target_qps` and  `--singlestream_target_latency` can be used to override the determined performance numbers

### Populate the README files describing your submission

```
cmr "generate-run-cmds inference _populate-readme _all-scenarios" \
--model=retinanet --device=cpu --implementation=reference --backend=pytorch \
--execution-mode=valid --results_dir=$HOME/results_dir \
--category=edge --division=open --quiet \
--adr.ml-engine-pytorch.version=1.13.0 --adr.ml-engine-torchvision.version=0.14.1
```

### Generate and upload MLPerf submission

Follow [this guide](../Submission.md) to generate the submission tree and upload your results.

### Run individual scenarios for testing and optimization

TBD

### Questions? Suggestions?

Don't hesitate to get in touch via [public Discord server](https://discord.gg/JjWNWXKxwT).