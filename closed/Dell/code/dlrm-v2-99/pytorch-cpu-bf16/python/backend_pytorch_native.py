"""
pytoch native backend for dlrm
"""
import os
import torch
import backend
import numpy as np
import sys

from typing import List, Optional, Union, Callable

from dataset import Dataset
from model.dlrm_model import DLRMMLPerf
from torch import nn
from torchrec.datasets.utils import Batch
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor
from torchrec.datasets.random import RandomRecDataset
from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES

import intel_extension_for_pytorch as ipex

# Modules for distributed running
import torch.multiprocessing as mp

def get_backend(backend, dataset, use_gpu, debug):
    if backend == "pytorch-native":
        n_cores = int(os.environ.get("WORLD_SIZE", 1))
        if n_cores > 1:
            if dataset == "debug":
                # 1. Syntetic debug dataset
                backend = BackendPytorchNative(
                    num_embeddings_per_feature = [2 for _ in range(26)],
                    embedding_dim=128,
                    dcn_num_layers=3,
                    dcn_low_rank_dim=512,
                    dense_arch_layer_sizes=[512, 256, 128],
                    over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
                    use_gpu=use_gpu,
                    debug=True
                )
                print("debug dataset")
            elif dataset == "multihot-criteo-sample":
                # 2. Syntetic multihot criteo sample
                backend = BackendPytorchNative(
                    num_embeddings_per_feature = [40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36],
                    embedding_dim=128,
                    dcn_num_layers=3,
                    dcn_low_rank_dim=512,
                    dense_arch_layer_sizes=[512, 256, 128],
                    over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
                    use_gpu=use_gpu,
                    debug=debug
                )
                print("multihot sample")
            elif dataset == "multihot-criteo":
                # 3. Syntetic multihot criteo
                backend = BackendPytorchNative(
                    num_embeddings_per_feature = [40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36],
                    embedding_dim=128,
                    dcn_num_layers=3,
                    dcn_low_rank_dim=512,
                    dense_arch_layer_sizes=[512, 256, 128],
                    over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
                    use_gpu=use_gpu,
                    debug=debug
                )
                print("multihot")
            else:
                raise ValueError("only debug|multihot-criteo-sample|multihot-criteo dataset options are supported")
        else:
            if dataset == "debug":
                # 1. Syntetic debug dataset
                backend = BackendPytorchNative(
                    num_embeddings_per_feature = [2 for _ in range(26)],
                    embedding_dim=128,
                    dcn_num_layers=3,
                    dcn_low_rank_dim=512,
                    dense_arch_layer_sizes=[512, 256, 128],
                    over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
                    use_gpu=use_gpu,
                    debug=True
                )
            elif dataset == "multihot-criteo-sample":
                # 2. Syntetic multihot criteo sample
                backend = BackendPytorchNative(
                    num_embeddings_per_feature = [40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36],
                    embedding_dim=128,
                    dcn_num_layers=3,
                    dcn_low_rank_dim=512,
                    dense_arch_layer_sizes=[512, 256, 128],
                    over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
                    use_gpu=use_gpu,
                    debug=debug
                )
            elif dataset == "multihot-criteo":
                # 3. Syntetic multihot criteo
                backend = BackendPytorchNative(
                    num_embeddings_per_feature = [40000000,39060,17295,7424,20265,3,7122,1543,63,40000000,3067956,405282,10,2209,11938,155,4,976,14,40000000,40000000,40000000,590152,12973,108,36],
                    embedding_dim=128,
                    dcn_num_layers=3,
                    dcn_low_rank_dim=512,
                    dense_arch_layer_sizes=[512, 256, 128],
                    over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
                    use_gpu=use_gpu,
                    debug=debug
                )
            else:
                raise ValueError("only debug|multihot-criteo-sample|multihot-criteo dataset options are supported")

    else:
        raise ValueError("unknown backend: " + backend)
    return backend

def unpack(t: KeyedJaggedTensor) -> (torch.Tensor, List[torch.Tensor], List[torch.Tensor]):
    dense = t.dense_features
    sparse = t.sparse_features
    features = [sparse[f'cat_{i}'] for i in range(26)]
    index = [f.values() for f in features]
    offset = [f.offsets() for f in features]
    return (dense, index, offset)

def convert_int8(max_batchsize: int,
                 calibration: bool,
                 model: torch.nn.Module,
                 int8_configure_dir: str,
                 ds: Dataset):
    from torch.ao.quantization import MinMaxObserver, PerChannelMinMaxObserver, QConfig
    from intel_extension_for_pytorch.quantization import prepare, convert
    qconfig = QConfig(
        activation=MinMaxObserver.with_args(qscheme=torch.per_tensor_symmetric, dtype=torch.qint8),
        weight=PerChannelMinMaxObserver.with_args(dtype=torch.qint8, qscheme=torch.per_channel_symmetric)
    )
    sample_ids = list(range(16))
    batch = ds.test_data.load_batch(sample_ids)
    print('batch done')
    model = prepare(
        model,
        qconfig,
        example_inputs=unpack(batch),
        inplace=True
    )
    print('model', model)
    if calibration:
        assert ds is not None
        count = ds.get_item_count()
        second_half_start_index = int(count // 2 + count % 2)
        batchsize = 256
        num_samples = 128000
        sample_per_batch = num_samples // batchsize
        all_sample_ids = list(range(second_half_start_index, second_half_start_index + num_samples))
        ds.load_query_samples(all_sample_ids)
        for i in range(num_samples // batchsize):
            sample_ids = all_sample_ids[(i * sample_per_batch) : ((i + 1) * sample_per_batch)]
            batch = ds.test_data.load_batch(sample_ids)
            model(*unpack(batch))
        model.save_qconf_summary(qconf_summary=int8_configure_dir)
        print(f"calibration done and save to {int8_configure_dir}")
        return model
    else:
        print("before load qconf")
        model.load_qconf_summary(qconf_summary = int8_configure_dir)
        print("after load qconf")
        convert(model, inplace=True)
        model.eval()
        print("before trace")
        model = torch.jit.trace(model, unpack(batch), check_trace=True)
        model = torch.jit.freeze(model)
        print("after freez")
        model(*unpack(batch))
        model(*unpack(batch))
        return model

class BackendPytorchNative(backend.Backend):
    def __init__(
        self,
        num_embeddings_per_feature,
        embedding_dim=128,
        dcn_num_layers=3,
        dcn_low_rank_dim=512,
        dense_arch_layer_sizes=[512, 256, 128],
        over_arch_layer_sizes=[1024, 1024, 512, 256, 1],
        use_gpu=False,
        debug=False,
    ):
        super(BackendPytorchNative, self).__init__()
        self.i = 0
        self.sess = None
        self.model = None

        self.embedding_dim = embedding_dim
        self.num_embeddings_per_feature = num_embeddings_per_feature
        self.dcn_num_layers = dcn_num_layers
        self.dcn_low_rank_dim = dcn_low_rank_dim
        self.dense_arch_layer_sizes = dense_arch_layer_sizes
        self.over_arch_layer_sizes = over_arch_layer_sizes
        self.debug = debug

        print("Using CPU...")

    def version(self):
        return torch.__version__

    def name(self):
        return "pytorch-native-dlrm"

    def load(self, args, dataset):
        # debug prints
        # print(model_path, inputs, outputs)
        max_batchsize = args.max_batchsize
        model_path = args.model_path
        inputs = args.inputs
        outputs = args.outputs
        int8_configure_dir = args.int8_configure_dir
        calibration = args.calibration
        print(f"Loading model from {model_path}")
        print("Initializing model...")
        model = DLRMMLPerf(
            embedding_dim=self.embedding_dim,
            num_embeddings_pool=self.num_embeddings_per_feature,
            dense_in_features=len(DEFAULT_INT_NAMES),
            dense_arch_layer_sizes=self.dense_arch_layer_sizes,
            over_arch_layer_sizes=self.over_arch_layer_sizes,
            dcn_num_layers=self.dcn_num_layers,
            dcn_low_rank_dim=self.dcn_low_rank_dim
        )
        # path_to_sharded_weights should have 2 subdirectories - batched and sharded
        # If we need to load the weights on different device or world size, we would need to change the process
        # group accordingly. If we would want to load on 8 GPUs, the process group created above should be fine
        # to understand sharding, --print_sharding_plan flag should be used while running dlrm_main.py in
        # torcherec implementation
        if not self.debug:
            print("Loading model weights...")
            ld_model = torch.load(model_path)
            # model_state = self.model.state_dict()
            # param_name = {'model.dense_arch.model._mlp.0._linear.bias': 'dense_arch.model._mlp.0.bias',
            #  'model.dense_arch.model._mlp.0._linear.weight': 'dense_arch.model._mlp.0.weight',
            #  'model.dense_arch.model._mlp.1._linear.bias': 'dense_arch.model._mlp.2.bias',
            #  'model.dense_arch.model._mlp.1._linear.weight': 'dense_arch.model._mlp.2.weight',
            #  'model.dense_arch.model._mlp.2._linear.bias': 'dense_arch.model._mlp.4.bias',
            #  'model.dense_arch.model._mlp.2._linear.weight': 'dense_arch.model._mlp.4.weight',
            #  'model.inter_arch.crossnet.V_kernels.0': 'inter_arch.crossnet.MLPs.V0.weight',
            #  'model.inter_arch.crossnet.V_kernels.1': 'inter_arch.crossnet.MLPs.V1.weight',
            #  'model.inter_arch.crossnet.V_kernels.2': 'inter_arch.crossnet.MLPs.V2.weight',
            #  'model.inter_arch.crossnet.W_kernels.0': 'inter_arch.crossnet.MLPs.W0.weight',
            #  'model.inter_arch.crossnet.W_kernels.1': 'inter_arch.crossnet.MLPs.W1.weight',
            #  'model.inter_arch.crossnet.W_kernels.2': 'inter_arch.crossnet.MLPs.W2.weight',
            #  'model.inter_arch.crossnet.bias.0': 'inter_arch.crossnet.MLPs.W0.bias',
            #  'model.inter_arch.crossnet.bias.1': 'inter_arch.crossnet.MLPs.W1.bias',
            #  'model.inter_arch.crossnet.bias.2': 'inter_arch.crossnet.MLPs.W2.bias',

            #  'model.over_arch.model.0._mlp.0._linear.bias': 'over_arch.model._mlp.0.bias',
            #  'model.over_arch.model.0._mlp.0._linear.weight': 'over_arch.model._mlp.0.weight',
            #  'model.over_arch.model.0._mlp.1._linear.bias': 'over_arch.model._mlp.2.bias',
            #  'model.over_arch.model.0._mlp.1._linear.weight': 'over_arch.model._mlp.2.weight',
            #  'model.over_arch.model.0._mlp.2._linear.bias': 'over_arch.model._mlp.4.bias',
            #  'model.over_arch.model.0._mlp.2._linear.weight': 'over_arch.model._mlp.4.weight',
            #  'model.over_arch.model.0._mlp.3._linear.bias': 'over_arch.model._mlp.6.bias',
            #  'model.over_arch.model.0._mlp.3._linear.weight': 'over_arch.model._mlp.6.weight',
            #  'model.over_arch.model.1.bias': 'over_arch.model._mlp.8.bias',
            #  'model.over_arch.model.1.weight': 'over_arch.model._mlp.8.weight',

            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_0.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.0.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_1.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.1.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_2.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.2.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_3.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.3.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_4.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.4.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_5.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.5.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_6.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.6.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_7.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.7.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_8.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.8.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_9.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.9.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_10.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.10.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_11.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.11.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_12.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.12.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_13.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.13.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_14.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.14.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_15.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.15.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_16.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.16.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_17.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.17.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_18.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.18.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_19.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.19.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_20.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.20.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_21.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.21.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_22.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.22.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_23.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.23.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_24.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.24.weight',
            #  'model.sparse_arch.embedding_bag_collection.embedding_bags.t_cat_25.weight': 'sparse_arch.embedding_bag_collection.embedding_bags.25.weight'}
            # for torrec_name, torch_name in param_name.items():
            #     assert model_state[torch_name].shape == ld_model[torrec_name].shape
            #     model_state[torch_name] = ld_model[torrec_name]
            # torch.save(model_state, '<DIR>')
            model.load_state_dict(ld_model)
            # from torchsnapshot import Snapshot
            # snapshot = Snapshot(path=model_path)
            # snapshot.restore(app_state={"model": self.model})

            ### To understand the keys in snapshot, you can look at following code snippet.
            # d = snapshot.get_manifest()
            # for k, v in d.items():
            #     print(k, v)
        self.model = model
        if args.use_int8:
            self.model = convert_int8(max_batchsize, calibration,
                                      self.model, int8_configure_dir, dataset)
        else:
            self.model = model
        self.model.eval()

        if args.use_bf16:
            self.model = ipex.optimize(self.model, torch.bfloat16, None, inplace=True)
            self.model.sparse_arch.embedding_bag_collection.embedding_bags.bfloat16()

        return self

    def predict(self, batch, ids = None):
        return self.batch_predict(*unpack(batch))

    def batch_predict(self, densex, lSi, lSo):
        with torch.no_grad():
            out = self.model(densex, lSi, lSo)
            out = torch.sigmoid(out)
            out = torch.reshape(out, (-1, ))
            return out
