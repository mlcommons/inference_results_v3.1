# Copyright (c) 2023, NVIDIA CORPORATION. All rights reserved.
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.


from collections import defaultdict
from functools import partial
from nvmitten.nvidia.builder import (TRTBuilder,
                                     MLPerfInferenceEngine,
                                     LegacyBuilder)
from nvmitten.pipeline import Operation
from pathlib import Path
from code.common.systems.system_list import SystemClassifications
if not SystemClassifications.is_soc():
    from torchrec import EmbeddingBagCollection
    from torchrec.datasets.criteo import DEFAULT_CAT_NAMES, DEFAULT_INT_NAMES
    from torchrec.datasets.criteo import INT_FEATURE_COUNT, CAT_FEATURE_COUNT
    from torchrec.models.dlrm import DLRM, DLRM_DCN, DLRMTrain
    from torchrec.modules.embedding_configs import EmbeddingBagConfig
    from torchsnapshot import Snapshot
from typing import List

import numpy as np
import os
import re
import tensorrt as trt
import torch

# Distributed Torch libraries to import DLRMv2's sharded checkpoint
from torch import distributed as torch_distrib
if not SystemClassifications.is_soc():
    from torchrec.distributed.planner import EmbeddingShardingPlanner, Topology
    from torchrec.distributed.model_parallel import DistributedModelParallel, get_default_sharders
    from torchrec.distributed.comm import get_local_size
    from torchrec.distributed.planner.storage_reservations import HeuristicalStorageReservation
    import torchrec.distributed as torchrec_distrib

from code.common.constants import TRT_LOGGER
from code.common.mitten_compat import ArgDiscarder

from code.plugin import load_trt_plugin_by_network
load_trt_plugin_by_network("dlrmv2")

from .criteo import CRITEO_SYNTH_MULTIHOT_N_EMBED_PER_FEATURE, CRITEO_SYNTH_MULTIHOT_SIZES


class DLRMv2_Model:

    def __init__(self,
                 model_path="/home/mlperf_inf_dlrmv2/model/model_weights",
                 num_embeddings_per_feature: int = CRITEO_SYNTH_MULTIHOT_N_EMBED_PER_FEATURE,
                 embedding_dim: int = 128,
                 dcn_num_layers: int = 3,
                 dcn_low_rank_dim: int = 512,
                 dense_arch_layer_sizes: List[int] = (512, 256, 128),
                 over_arch_layer_sizes: List[int] = (1024, 1024, 512, 256, 1),
                 load_ckpt_on_gpu: bool = False):
        self.model_path = model_path
        self.num_embeddings_per_feature = list(num_embeddings_per_feature)
        self.embedding_dim = embedding_dim
        self.dcn_num_layers = dcn_num_layers
        self.dcn_low_rank_dim = dcn_low_rank_dim
        self.dense_arch_layer_sizes = list(dense_arch_layer_sizes)
        self.over_arch_layer_sizes = list(over_arch_layer_sizes)

        if load_ckpt_on_gpu and torch.cuda.is_available():
            self.device = torch.device("cuda:0")
            self.distrib_backend = "nccl"
        else:
            self.device = torch.device("cpu")
            self.distrib_backend = "gloo"

    def load_model(self, return_snapshot: bool = False):
        os.environ["WORLD_SIZE"] = "1"
        os.environ["RANK"] = "0"
        os.environ["MASTER_ADDR"] = "localhost"
        os.environ["MASTER_PORT"] = "29500"

        torch_distrib.init_process_group(backend=self.distrib_backend, rank=0, world_size=1)

        self.embedding_bag_configs = [
            EmbeddingBagConfig(name=f"t_{feature_name}",
                               embedding_dim=self.embedding_dim,
                               num_embeddings=self.num_embeddings_per_feature[feature_idx],
                               feature_names=[feature_name])
            for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
        ]

        self.embedding_bag_collection = EmbeddingBagCollection(tables=self.embedding_bag_configs,
                                                               device=torch.device("meta"))

        dlrm_model = DLRM_DCN(embedding_bag_collection=self.embedding_bag_collection,
                              dense_in_features=len(DEFAULT_INT_NAMES),
                              dense_arch_layer_sizes=self.dense_arch_layer_sizes,
                              over_arch_layer_sizes=self.over_arch_layer_sizes,
                              dcn_num_layers=self.dcn_num_layers,
                              dcn_low_rank_dim=self.dcn_low_rank_dim,
                              dense_device=self.device)
        _model = DLRMTrain(dlrm_model)

        # Distribute the model
        planner = EmbeddingShardingPlanner(
            topology=Topology(
                local_world_size=get_local_size(),
                world_size=torch_distrib.get_world_size(),
                compute_device=self.device.type,
            ),
            storage_reservation=HeuristicalStorageReservation(percentage=0.05),
        )
        plan = planner.collective_plan(
            _model, get_default_sharders(), torch_distrib.GroupMember.WORLD
        )
        model = DistributedModelParallel(
            module=_model,
            device=self.device,
            plan=plan
        )

        # Load weights
        snapshot = Snapshot(path=self.model_path)
        snapshot.restore(app_state={"model": model})
        model.eval()

        if return_snapshot:
            return model, snapshot
        else:
            return model

    def get_embedding_weight(self, model, cat_feature_idx: int):
        assert cat_feature_idx < len(DEFAULT_CAT_NAMES)
        embedding_bag_state = model.module.model.sparse_arch.embedding_bag_collection.state_dict()
        key = f"embedding_bags.t_cat_{cat_feature_idx}.weight"
        out = torch.zeros(embedding_bag_state[key].metadata().size, device=self.device)
        embedding_bag_state[key].gather(0, out=out)
        return out

    def dump_embedding_weights(self, model, save_dir: os.PathLike):
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)

        mega_table_path = save_dir / "mega_table_fp16.npy"
        mega_table = []

        for i in range(len(DEFAULT_CAT_NAMES)):
            save_path = save_dir / f"embed_feature_{i}.weight.pt"
            weight = self.get_embedding_weight(model, i).cpu()
            torch.save(weight, save_path)
            mega_table.append(weight.numpy())

        mega_table = np.vstack(mega_table).reshape(-1).astype(np.float16)
        with open(mega_table_path, 'wb') as mega_table_file:
            np.save(mega_table_file, mega_table)

    def load_embeddings(self, from_dir: os.PathLike):
        embedding_bag_configs = [
            EmbeddingBagConfig(name=f"t_{feature_name}",
                               embedding_dim=self.embedding_dim,
                               num_embeddings=self.num_embeddings_per_feature[feature_idx],
                               feature_names=[feature_name])
            for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES)
        ]

        embedding_bag_collection = EmbeddingBagCollection(tables=embedding_bag_configs,
                                                          device=self.device)

        # torchrec 0.3.2 does not support init_fn as a EmbeddingBagConfig parameter.
        # Manually implement it here.
        for feature_idx, feature_name in enumerate(DEFAULT_CAT_NAMES):
            with open(Path(from_dir) / f"embed_feature_{feature_idx}.weight.pt", 'rb') as f:
                dat = torch.load(f, map_location=self.device)
            with torch.no_grad():
                embedding_bag_collection.embedding_bags[f"t_{feature_name}"].weight.copy_(dat)
        return embedding_bag_collection


class DLRMv2Arch:
    """Loose representation of the DLRMv2 architecture based on TorchRec source:
    https://github.com/pytorch/torchrec/blob/main/torchrec/models/dlrm.py

    The components of the model are as follows:

    1. SparseArch (Embedding table, isolated into DLRMv2_Model)
    2. DenseArch (Bottom MLP)
    3. InteractionDCNArch (DCNv2, or sometimes referred to as interactions network / layer)
    4. OverArch (Top MLP + final linear layer)
    """

    def __init__(self,
                 state_dict,
                 bot_mlp_depth: int = 3,
                 crossnet_depth: int = 3,
                 top_mlp_depth: int = 4):
        self.bot_mlp_depth = bot_mlp_depth
        self.bottom_mlp = self.create_bot_mlp(state_dict)

        self.crossnet_depth = crossnet_depth
        self.crossnet = self.create_crossnet(state_dict)

        self.top_mlp_depth = top_mlp_depth
        self.top_mlp = self.create_top_mlp(state_dict)

        self.final_linear = self.create_final_linear(state_dict)

    def create_bot_mlp(self, state_dict):
        """ Bottom MLP keys
        model.dense_arch.model._mlp.0._linear.bias
        model.dense_arch.model._mlp.0._linear.weight
        model.dense_arch.model._mlp.1._linear.bias
        model.dense_arch.model._mlp.1._linear.weight
        model.dense_arch.model._mlp.2._linear.bias
        model.dense_arch.model._mlp.2._linear.weight
        """
        conf = defaultdict(dict)
        for i in range(self.bot_mlp_depth):
            key_prefix = f"model.dense_arch.model._mlp.{i}._linear."
            conf[i]["weight"] = state_dict[key_prefix + "weight"]
            conf[i]["bias"] = state_dict[key_prefix + "bias"]
        return conf

    def create_crossnet(self, state_dict):
        """ DCNv2 crossnet is based on torchrec.modules.crossnet.LowRankCrossNet:
            - https://pytorch.org/torchrec/torchrec.modules.html#torchrec.modules.crossnet.LowRankCrossNet
            - https://github.com/pytorch/torchrec/blob/42c55844d29343c644521e810597fd67017eac8f/torchrec/modules/crossnet.py#L90

        Keys:
        model.inter_arch.crossnet.V_kernels.0
        model.inter_arch.crossnet.V_kernels.1
        model.inter_arch.crossnet.V_kernels.2
        model.inter_arch.crossnet.W_kernels.0
        model.inter_arch.crossnet.W_kernels.1
        model.inter_arch.crossnet.W_kernels.2
        model.inter_arch.crossnet.bias.0
        model.inter_arch.crossnet.bias.1
        model.inter_arch.crossnet.bias.2
        """
        conf = defaultdict(dict)
        for i in range(self.crossnet_depth):
            V = f"model.inter_arch.crossnet.V_kernels.{i}"
            W = f"model.inter_arch.crossnet.W_kernels.{i}"
            bias = f"model.inter_arch.crossnet.bias.{i}"
            conf[i]['V'] = state_dict[V]
            conf[i]['W'] = state_dict[W]
            conf[i]["bias"] = state_dict[bias]
        return conf

    def create_top_mlp(self, state_dict):
        """ Top MLP keys
        model.over_arch.model.0._mlp.0._linear.bias
        model.over_arch.model.0._mlp.0._linear.weight
        model.over_arch.model.0._mlp.1._linear.bias
        model.over_arch.model.0._mlp.1._linear.weight
        model.over_arch.model.0._mlp.2._linear.bias
        model.over_arch.model.0._mlp.2._linear.weight
        model.over_arch.model.0._mlp.3._linear.bias
        model.over_arch.model.0._mlp.3._linear.weight
        """
        conf = defaultdict(dict)
        for i in range(self.top_mlp_depth):
            key_prefix = f"model.over_arch.model.0._mlp.{i}._linear."
            conf[i]["weight"] = state_dict[key_prefix + "weight"]
            conf[i]["bias"] = state_dict[key_prefix + "bias"]
        return conf

    def create_final_linear(self, state_dict):
        """ Probability reduction linear layer keys
        model.over_arch.model.1.bias
        model.over_arch.model.1.weight
        """
        conf = {
            "weight": state_dict["model.over_arch.model.1.weight"],
            "bias": state_dict["model.over_arch.model.1.bias"],
        }
        return conf


class DLRMv2TRTNetwork:
    def __init__(
        self,
        batch_size: int,
        use_embedding_lookup_plugin: bool = True,
        embedding_weights_on_gpu_part: float = 1.0,
        model_path: str = "/home/mlperf_inf_dlrmv2/model/model_weights",
        mega_table_npy_file: str = '/home/mlperf_inf_dlrmv2/model/embedding_weights/mega_table_fp16.npy',
        reduced_precision_io: bool = True
    ):
        self.batch_size = batch_size
        self.verbose = True
        self.logger = TRT_LOGGER
        self.logger.min_severity = trt.Logger.VERBOSE if self.verbose else trt.Logger.INFO
        self.builder = trt.Builder(self.logger)
        self.network = self.builder.create_network(1 << int(trt.NetworkDefinitionCreationFlag.EXPLICIT_BATCH))

        # DLRMv2EmbeddingLookupPlugin
        self.use_embedding_lookup_plugin = use_embedding_lookup_plugin
        self.embedding_weights_on_gpu_part = embedding_weights_on_gpu_part
        self.reduced_precision_io = reduced_precision_io
        self.mega_table_npy_file = mega_table_npy_file

        # create trt network
        self.load_model(model_path)

    def load_model(self, model_path):
        dlrm_model = DLRMv2_Model(model_path=model_path)
        model = dlrm_model.load_model()

        # Check for megatable
        megatable_path = Path(self.mega_table_npy_file)
        if not megatable_path.exists():
            print("Embedding megatable does not exist. Generating...")
            dlrm_model.dump_embedding_weights(model, megatable_path.parent)
            print("Generated megatable.")
        else:
            print("Found embedding megatable file.")

        # Create TRT Network
        self.arch = DLRMv2Arch(model.state_dict())

        self.embedding_size = dlrm_model.embedding_dim
        self.num_numerical_features = INT_FEATURE_COUNT
        self.num_categorical_features = CAT_FEATURE_COUNT

        # create bottom mlp
        # input from dense features:        [-1, num_features, 1, 1]
        numerical_input = self.network.add_input('numerical_input', trt.DataType.FLOAT, (-1, self.num_numerical_features, 1, 1))
        self.bottom_mlp = self._build_mlp(self.arch.bottom_mlp, numerical_input, self.num_numerical_features, 'bot_mlp')

        # create embedding lookup
        if self.use_embedding_lookup_plugin:

            # flatten dense inputs:         [-1, 128, 1, 1] -> [-1, 128]
            shuffle_dense_op = self.network.add_shuffle(self.bottom_mlp.get_output(0))
            shuffle_dense_op.reshape_dims = (-1, self.embedding_size)
            dense_input = shuffle_dense_op.get_output(0)

            # dense input flattened:        [-1, embedding_size]
            # sparse input from harness:    [-1, total_hotness]
            sparse_input = self.network.add_input('sparse_input', trt.DataType.INT32, (-1, sum(CRITEO_SYNTH_MULTIHOT_SIZES)))

            dlrm_embedding_lookup_plugin = self.get_dlrmv2_embedding_lookup_plugin()
            embedding_op = self.network.add_plugin_v2([dense_input, sparse_input], dlrm_embedding_lookup_plugin)
            embedding_op.get_output(0).name = "dlrmv2_embedding_lookup"

            # input for interaction layer   [-1, 3456]
            interaction_input = embedding_op.get_output(0)

        else:

            # use pytorch for embedding lookups
            # input from bottom mlp:        [-1, embedding_size, 1, 1]
            # input from embedding lookup:  [-1, num_categorical_features, embedding_size]
            embedding_lookup = self.network.add_input('embedding_lookup', trt.DataType.FLOAT, (-1, self.num_categorical_features, self.embedding_size))

            # flatten dense inputs:         [-1, 128, 1, 1] -> [-1, 128]
            shuffle_dense_op = self.network.add_shuffle(self.bottom_mlp.get_output(0))
            shuffle_dense_op.reshape_dims = (-1, self.embedding_size)
            dense_input = shuffle_dense_op.get_output(0)

            # flatten sparse inputs:        [-1, 26, 128] -> [-1, 3328]
            shuffle_sparse_op = self.network.add_shuffle(embedding_lookup)
            shuffle_sparse_op.reshape_dims = (-1, CAT_FEATURE_COUNT * self.embedding_size)
            sparse_input = shuffle_sparse_op.get_output(0)

            # concatenate inputs:           [-1, 3456]
            in_concat_op = self.network.add_concatenation([dense_input, sparse_input])
            in_concat_op.axis = 1

            # input for interaction layer
            interaction_input = in_concat_op.get_output(0)

        # create interaction op
        self.interaction_op = self._build_interaction_op(self.arch.crossnet, interaction_input)

        # create top mlp
        # input from interaction op:         [-1, 3456, 1, 1]
        top_mlp_input = self.interaction_op.get_output(0)
        self.top_mlp = self._build_mlp(self.arch.top_mlp, top_mlp_input, top_mlp_input.shape[1], 'top_mlp')

        # create final linear layer
        # input from top mlp:                [-1, 3456]
        final_linear_input = self.top_mlp.get_output(0)
        self.final_linear = self._build_linear(self.arch.final_linear,
                                               final_linear_input,
                                               final_linear_input.shape[1],
                                               'final_linear',
                                               add_relu=False)

        # create sigmoid output layer
        # input from final_layer:            [-1, 3456]
        sigmoid_input = self.final_linear.get_output(0)
        self.sigmoid_layer = self.network.add_activation(sigmoid_input, trt.ActivationType.SIGMOID)
        self.sigmoid_layer.name = "sigmoid"
        self.sigmoid_layer.get_output(0).name = "sigmoid_output"

        # mark output
        # self.network.mark_output(self.top_mlp.get_output(0))
        self.network.mark_output(self.sigmoid_layer.get_output(0))

    def _build_mlp(self,
                   config,
                   in_tensor,
                   in_channels,
                   name_prefix,
                   use_conv_for_fc=False):
        for index, state in config.items():
            layer = self._build_linear(state,
                                       in_tensor,
                                       in_channels,
                                       f'{name_prefix}_{index}',
                                       use_conv_for_fc=use_conv_for_fc)
            shape = state['weight'].shape[::-1]
            in_channels = shape[-1]
            in_tensor = layer.get_output(0)
        return layer

    def _build_linear(self,
                      state,
                      in_tensor,
                      in_channels,
                      name,
                      add_relu=True,
                      use_conv_for_fc=False):
        weights = state['weight'].numpy()
        bias = state['bias'].numpy()

        shape = weights.shape[::-1]
        out_channels = shape[-1]

        if use_conv_for_fc:
            layer = self.network.add_convolution(in_tensor, out_channels, (1, 1), weights, bias)
        else:
            layer = self.network.add_fully_connected(in_tensor, out_channels, weights, bias)

        layer.name = name
        layer.get_output(0).name = name + ".output"

        if add_relu:
            layer = self.network.add_activation(layer.get_output(0), trt.ActivationType.RELU)
            layer.name = name + ".relu"
            layer.get_output(0).name = name + ".relu.output"
        return layer

    def _build_interaction_op(self, config, x):
        # From LowRankCrossNet docs:
        # https://pytorch.org/torchrec/torchrec.modules.html#torchrec.modules.crossnet.LowRankCrossNet
        # x_next = x_0 * (matmul(W_curr, matmul(V_curr, x_curr)) + bias_curr) + x_curr
        x0 = x

        for _, state in config.items():
            V = state['V'].numpy()
            W = state['W'].numpy()
            b = state['bias'].numpy()

            V_tens = self.network.add_constant(V.shape, V)
            W_tens = self.network.add_constant(W.shape, W)
            b_tens = self.network.add_constant([1, b.shape[0]], b)

            vx = self.network.add_matrix_multiply(x, trt.MatrixOperation.NONE, V_tens.get_output(0), trt.MatrixOperation.TRANSPOSE).get_output(0)
            wvx = self.network.add_matrix_multiply(vx, trt.MatrixOperation.NONE, W_tens.get_output(0), trt.MatrixOperation.TRANSPOSE).get_output(0)
            inner = self.network.add_elementwise(wvx, b_tens.get_output(0), trt.ElementWiseOperation.SUM).get_output(0)
            left_term = self.network.add_elementwise(inner, x0, trt.ElementWiseOperation.PROD).get_output(0)
            x = self.network.add_elementwise(left_term, x, trt.ElementWiseOperation.SUM).get_output(0)

        # unsqueeze output [-1, 3456] -> [-1, 3456, 1, 1]
        unsqueeze = self.network.add_shuffle(x)
        unsqueeze.reshape_dims = (-1, (CAT_FEATURE_COUNT * self.embedding_size) + self.embedding_size, 1, 1)
        return unsqueeze

    def get_dlrmv2_embedding_lookup_plugin(self):
        """Create a plugin layer for the DLRMv2 Embedding Lookup plugin and return it. """

        pluginName = "DLRMv2_EMBEDDING_LOOKUP_TRT"
        embeddingRows = sum(CRITEO_SYNTH_MULTIHOT_N_EMBED_PER_FEATURE)
        tableOffsets = np.concatenate(([0], np.cumsum(CRITEO_SYNTH_MULTIHOT_N_EMBED_PER_FEATURE).astype(np.int32)[:-1])).astype(np.int32)
        tableHotness = np.array(CRITEO_SYNTH_MULTIHOT_SIZES).astype(np.int32)
        totalHotness = sum(CRITEO_SYNTH_MULTIHOT_SIZES)
        reducedPrecisionIO = 1 if self.reduced_precision_io else 0

        plugin = None
        for plugin_creator in trt.get_plugin_registry().plugin_creator_list:
            if plugin_creator.name == pluginName:
                embeddingSize_field = trt.PluginField("embeddingSize", np.array([self.embedding_size], dtype=np.int32), trt.PluginFieldType.INT32)
                embeddingRows_field = trt.PluginField("embeddingRows", np.array([embeddingRows], dtype=np.int32), trt.PluginFieldType.INT32)
                embeddingWeightsOnGpuPart_field = trt.PluginField("embeddingWeightsOnGpuPart", np.array([self.embedding_weights_on_gpu_part], dtype=np.float32), trt.PluginFieldType.FLOAT32)
                tableHotness_field = trt.PluginField("tableHotness", tableHotness, trt.PluginFieldType.INT32)
                tableOffsets_field = trt.PluginField("tableOffsets", tableOffsets, trt.PluginFieldType.INT32)
                batchSize_field = trt.PluginField("batchSize", np.array([self.batch_size], dtype=np.int32), trt.PluginFieldType.INT32)
                embedHotnessTotal_field = trt.PluginField("embedHotnessTotal", np.array([totalHotness], dtype=np.int32), trt.PluginFieldType.INT32)
                embeddingWeightsFilepath_field = trt.PluginField("embeddingWeightsFilepath", np.array(list(self.mega_table_npy_file.encode()), dtype=np.int8), trt.PluginFieldType.CHAR)
                reducedPrecisionIO_field = trt.PluginField("reducedPrecisionIO", np.array([reducedPrecisionIO], dtype=np.int32), trt.PluginFieldType.INT32)

                field_collection = trt.PluginFieldCollection([
                    embeddingSize_field,
                    embeddingRows_field,
                    embeddingWeightsOnGpuPart_field,
                    tableHotness_field,
                    tableOffsets_field,
                    batchSize_field,
                    embedHotnessTotal_field,
                    embeddingWeightsFilepath_field,
                    reducedPrecisionIO_field
                ])
                plugin = plugin_creator.create_plugin(name=pluginName, field_collection=field_collection)

        return plugin


class DLRMv2EngineBuilderOp(TRTBuilder,
                            MLPerfInferenceEngine,
                            Operation,
                            ArgDiscarder):
    @classmethod
    def immediate_dependencies(cls):
        return None

    def __init__(self,
                 workspace_size: int = 4 << 30,
                 # TODO: Legacy value - Remove after refactor is done.
                 config_ver: str = "default",
                 # TODO: This should be a relative path within the ScratchSpace.
                 model_path: str = "/home/mlperf_inf_dlrmv2/model/model_weights",
                 batch_size: int = 8192,
                 use_embedding_lookup_plugin: bool = True,
                 embedding_weights_on_gpu_part: float = 0.5,
                 mega_table_npy_file: str = '/home/mlperf_inf_dlrmv2/model/embedding_weights/mega_table_fp16.npy',
                 reduced_precision_io: bool = True,
                 **kwargs):
        super().__init__(workspace_size=workspace_size, **kwargs)

        self.config_ver = config_ver
        self.model_path = model_path
        self.batch_size = batch_size
        self.use_embedding_lookup_plugin = use_embedding_lookup_plugin
        self.embedding_weights_on_gpu_part = embedding_weights_on_gpu_part
        self.mega_table_npy_file = mega_table_npy_file
        self.reduced_precision_io = reduced_precision_io

    def create_network(self, builder: trt.Builder):
        dlrm_network = DLRMv2TRTNetwork(batch_size=self.batch_size,
                                        use_embedding_lookup_plugin=self.use_embedding_lookup_plugin,
                                        embedding_weights_on_gpu_part=self.embedding_weights_on_gpu_part,
                                        model_path=self.model_path,
                                        mega_table_npy_file=self.mega_table_npy_file,
                                        reduced_precision_io=self.reduced_precision_io)
        return dlrm_network.network

    def run(self, scratch_space, dependency_outputs):
        builder_config = self.create_builder_config(self.builder)

        # Needed for ConvMulAdd fusion from Myelin
        builder_config.builder_optimization_level = 4

        network = self.create_network(self.builder)
        engine_dir = self.engine_dir(scratch_space)
        engine_name = self.engine_name("gpu",
                                       self.batch_size,
                                       self.precision,
                                       tag=self.config_ver)
        engine_fpath = engine_dir / engine_name
        self.build_engine(network, builder_config, self.batch_size, engine_fpath)


class DLRMv2(LegacyBuilder):
    """Temporary spoofing class to wrap around Mitten to adhere to the old API.
    """

    def __init__(self, args):
        super().__init__(DLRMv2EngineBuilderOp(**args))
