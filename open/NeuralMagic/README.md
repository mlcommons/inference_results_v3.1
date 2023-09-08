# Neural Magic's DeepSparse MLPerf Submission

This is the repository of Neural Magic's [DeepSparse](https://github.com/neuralmagic/deepsparse) submission for [MLPerf Inference Benchmark v3.1](https://www.mlperf.org/inference-overview/).

This round's submission features a wide swath of BERT benchmarks faciliated by [Collective Mind (MLCommons CM)](https://github.com/mlcommons/ck/tree/master/cm-mlops) across CPU architectures, model architectures, and optimization methods with models from [SparseZoo](https://sparsezoo.neuralmagic.com/) running on the DeepSparse engine. CM provides a universal interface to any software project and transforms it into a database of reusable automation actions and portable scripts in a transparent and non-intrusive way. Special thanks to [cTuning](https://github.com/mlcommons/ck/blob/master/docs/taskforce.md) for collecting and producing the bulk of the results.

Neural Magic is bringing performant deep learning inference to ARM CPUs! DeepSparse ARM support is available now on the nightly build: `pip install deepsparse-nightly`


## Model Methods

Here are all the models on [SparseZoo that have SQuAD F1 score higher than 90 (>=99% of FP32 F1 90.874).](https://sparsezoo.neuralmagic.com/?useCase=question_answering&datasets=squad&ungrouped=true&sort=Throughput%3Adesc&accuracy=90.012%2C94.62)

You can click on the model to open its detailed card view and view the "RECIPES" tab to see how the model was sparsified. Here is an [example with one of the most performant models](https://sparsezoo.neuralmagic.com/models/mobilebert-squad_wikipedia_bookcorpus-14layer_pruned50.4block_quantized?hardware=deepsparse-c6i.12xlarge&comparison=mobilebert-squad_wikipedia_bookcorpus-base&tab=3): 

![Capture](https://github.com/mgoin/mlperf_inference_submissions_v3.1/assets/3195154/04b1bc5d-c669-45f6-910f-a09a85ebad90)

## Citation info
If you find our models useful, please consider citing our work:
```bibtex
@article{kurtic:2022,
  doi = {10.48550/ARXIV.2203.07259},
  url = {https://arxiv.org/abs/2203.07259},
  author = {Kurtic, Eldar and Campos, Daniel and Nguyen, Tuan and Frantar, Elias and Kurtz, Mark and Fineran, Benjamin and Goin, Michael and Alistarh, Dan},
  title = {The Optimal BERT Surgeon: Scalable and Accurate Second-Order Pruning for Large Language Models},
  publisher = {arXiv},
  year = {2022},
  copyright = {Creative Commons Attribution 4.0 International}
}
```
