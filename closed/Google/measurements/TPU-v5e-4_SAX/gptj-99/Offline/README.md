To run this benchmark, first follow the setup steps in https://github.com/google/saxml/blob/main/README.md
to install Sax, create TPUs, start Sax admin and model server, and then publish the model.

The GPTJ server-side model definition is provided in /code/gptj-99/params/gptj.py. 
A class must be chosen during the publishing step. For example, the base is the GPTJ class.

One can control the SAX behavior by adding attributes that modifies the SAX config.
This has been done for convenience, and added to the same file. 
We used GPTJ4BS32Int8Opt10Wait40MB6 for Server, and GPTJ4BS32Int8Opt10Early for Offline. 

To run mlperf benchmark, run something like
```
blaze-bin/third_party/mlperf/inference/gptj/main \
    --accuracy=<True/False> \
    --scenario=<Server/Offline> \
    --model_path=</path/to/sax/cell/containing/model> \
    --log_path=</path/to/loadgen/output>
```
Please read main.py for other args to adjust.
