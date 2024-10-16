[ [Back to the common setup](README.md) ]


## Run this benchmark via CM


### Do a test run to detect and record the system performance

```
cmr "generate-run-cmds inference _find-performance _all-scenarios" \
--model=3d-unet-99.9 --implementation=reference --device=cpu --backend=onnxruntime \
--category=edge --division=open --quiet
```
* Use `--device=cuda` to run the inference on Nvidia GPU
* Use `--division=closed` to run all scenarios for the closed division (compliance tests are skipped for `_find-performance` mode)
* Use `--category=datacenter` to run datacenter scenarios
* Use `--backend=pytorch` and `backend=tf` to use pytorch and tensorflow backends respectively
* Use `--model=3d-unet-99` to run the low accuracy constraint 3d-unet-99 model. But since we are running the fp32 model, this is redundant and instead, we can reuse the results of 3d-unet-99.9 for 3d-unet-99

### Do full accuracy and performance runs for all the scenarios

```
cmr "generate-run-cmds inference _submission _all-scenarios" --model=3d-unet-99.9 \
--device=cpu --implementation=reference --backend=onnxruntime \
--execution-mode=valid --results_dir=$HOME/inference_3.1_results \
--category=edge --division=open --quiet
```

* Use `--power=yes` for measuring power. It is ignored for accuracy and compliance runs
* Use `--division=closed` to run all scenarios for the closed division including the compliance tests
* `--offline_target_qps`, `--server_target_qps` and `--singlestream_target_latency` can be used to override the determined performance numbers

### Populate the README files describing your submission

```
cmr "generate-run-cmds inference _populate-readme _all-scenarios" \
--model=3d-unet-99.9 --device=cpu --implementation=reference --backend=onnxruntime \
--execution-mode=valid --results_dir=$HOME/inference_3.1_results \
--category=edge --division=open --quiet
```

### Generate actual submission tree

Here, we are copying the performance and accuracy log files (compliance logs also in the case of closed division) from the results directory to the submission tree following the [directory structure required by MLCommons Inference](https://github.com/mlcommons/policies/blob/master/submission_rules.adoc#inference-1). After the submission tree is generated, [accuracy truncate script](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/truncate-mlperf-inference-accuracy-log) is called to truncate accuracy logs and then the [submission checker](https://github.com/mlcommons/ck/tree/master/cm-mlops/script/run-mlperf-inference-submission-checker) is called to validate the generated submission tree.

We should use the master branch of MLCommons inference repo for the submission checker. You can use `--hw_note_extra` option to add your name to the notes.
```
cmr "generate inference submission" --results_dir=$HOME/inference_3.1_results/valid_results \
--submission_dir=$HOME/inference_submission_tree --clean  \
--run-checker --submitter=cTuning --adr.inference-src.version=master \
--hw_notes_extra="Result taken by NAME" --quiet
```


### Run individual scenarios for testing and optimization

TBD

### Questions? Suggestions?

Don't hesitate to get in touch via [public Discord server](https://discord.gg/JjWNWXKxwT).
