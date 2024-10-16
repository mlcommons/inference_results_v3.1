# Copyright (c) 2023, NVIDIA CORPORATION.  All rights reserved.
#
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

from code.common.fix_sys_path import ScopedRestrictedImport
with ScopedRestrictedImport():
    import numpy as np
import onnx
import tensorrt as trt
import json

from code.bert.tensorrt.builder_utils import add_gelu, mark


def bert_encoder_layer_fp16_var_seqlen(cfg, max_seqlen, weights_dict, network, input_tensor, cu_seqlens, layer, mask):
    """Builds one encoder layer in FP16 with var seqlen.
    Sets the dynamic ranges extracted from the qat checkpoint."""

    plg_registry = trt.get_plugin_registry()
    qkv_plg_creator = plg_registry.get_plugin_creator("CustomQKVToContextPluginDynamic", "2", "")
    pc_skln = plg_registry.get_plugin_creator("CustomSkipLayerNormPluginDynamic", "2", "")
    dtype = trt.float16
    N = cfg.N
    H = cfg.H
    prefix = 'l{}_'.format(layer)

    # FC QKV
    Wqkv = np.zeros((3, cfg.hidden_size, cfg.hidden_size), np.float32)
    Bqkv = np.zeros((3, cfg.hidden_size), np.float32)
    Wqkv[0, :, :] = weights_dict[prefix + 'attention_self_query_kernel']
    Wqkv[1, :, :] = weights_dict[prefix + 'attention_self_key_kernel']
    Wqkv[2, :, :] = weights_dict[prefix + 'attention_self_value_kernel']
    Bqkv[0, :] = weights_dict[prefix + 'attention_self_query_bias']
    Bqkv[1, :] = weights_dict[prefix + 'attention_self_key_bias']
    Bqkv[2, :] = weights_dict[prefix + 'attention_self_value_bias']

    Wqkv = np.ascontiguousarray(Wqkv.reshape((3, N, H, N, H)).transpose((1, 0, 2, 3, 4)))
    Bqkv = np.ascontiguousarray(Bqkv.reshape((3, N, H)).transpose((1, 0, 2)))

    fc_qkv = network.add_fully_connected(input_tensor, cfg.qkv_size, Wqkv, Bqkv)
    fc_qkv.name = prefix + 'fc_qkv'
    fc_qkv_out = fc_qkv.get_output(0)
    fc_qkv_out.name = prefix + 'attention_self_qkv_mult'
    # QKV2CTX
    pf_type = trt.PluginField("type_id", np.array([int(dtype)], np.int32), trt.PluginFieldType.INT32)
    pf_hidden_size = trt.PluginField("hidden_size", np.array([cfg.hidden_size], np.int32), trt.PluginFieldType.INT32)
    pf_num_heads = trt.PluginField("num_heads", np.array([cfg.N], np.int32), trt.PluginFieldType.INT32)
    pf_has_mask = trt.PluginField("has_mask", np.array([1], np.int32), trt.PluginFieldType.INT32)
    pf_var_seqlen = trt.PluginField("var_seqlen", np.array([int(1)], np.int32), trt.PluginFieldType.FLOAT32)

    pfc = trt.PluginFieldCollection([pf_hidden_size, pf_num_heads, pf_has_mask, pf_type, pf_var_seqlen])
    qkv2ctx_plug = qkv_plg_creator.create_plugin("qkv2ctx", pfc)

    qkv2ctx_layer = network.add_plugin_v2([fc_qkv_out, mask, cu_seqlens, max_seqlen], qkv2ctx_plug)
    qkv2ctx_layer.name = prefix + 'qkv_to_ctx'
    qkv2ctx_out = qkv2ctx_layer.get_output(0)
    # FC AOUT
    Waout = weights_dict[prefix + 'attention_output_dense_kernel']
    Baout = weights_dict[prefix + 'attention_output_dense_bias']
    fc_aout = network.add_fully_connected(qkv2ctx_out, cfg.hidden_size, Waout, Baout)
    fc_aout.precision = dtype
    fc_aout.name = prefix + 'fc_aout'
    fc_aout_out = fc_aout.get_output(0)
    fc_aout_out.dtype = dtype
    # Skip-Layernorm 1
    pf_ld = trt.PluginField("ld", np.array([cfg.hidden_size], np.int32), trt.PluginFieldType.INT32)
    pf_type = trt.PluginField("type_id", np.array([int(dtype)], np.int32), trt.PluginFieldType.INT32)
    pf_beta = trt.PluginField("beta", weights_dict[prefix + 'attention_output_layernorm_beta'], trt.PluginFieldType.FLOAT32)
    pf_gamma = trt.PluginField("gamma", weights_dict[prefix + 'attention_output_layernorm_gamma'], trt.PluginFieldType.FLOAT32)
    pf_bias = trt.PluginField("bias", Baout, trt.PluginFieldType.FLOAT32)
    fields = [pf_ld, pf_beta, pf_gamma, pf_type]
    pfc = trt.PluginFieldCollection(fields)
    skipln_plug = pc_skln.create_plugin("skipln", pfc)

    fc_aout_out.dtype = dtype
    skipln_inputs = [fc_aout_out, input_tensor]
    skln1 = network.add_plugin_v2(skipln_inputs, skipln_plug)
    skln1.name = prefix + 'skln_1'
    skln1_out = skln1.get_output(0)
    skln1_out.dtype = dtype
    # FC MID
    Wmid = weights_dict[prefix + 'intermediate_dense_kernel']
    Bmid = weights_dict[prefix + 'intermediate_dense_bias']
    fc_mid = network.add_fully_connected(skln1_out, cfg.mid_size, Wmid, Bmid)
    fc_mid.name = prefix + 'fc_mid'
    fc_mid_out = fc_mid.get_output(0)
    # GELU
    gelu_layer = add_gelu(network, fc_mid_out)
    gelu_layer.name = prefix + 'gelu'
    gelu_out = gelu_layer.get_output(0)
    # FC OUT
    Wout = weights_dict[prefix + 'output_dense_kernel']
    Bout = weights_dict[prefix + 'output_dense_bias']
    fc_out = network.add_fully_connected(gelu_out, cfg.hidden_size, Wout, Bout)
    fc_out.name = prefix + 'fc_out'
    fc_out.precision = dtype
    fc_out_out = fc_out.get_output(0)
    fc_out_out.dtype = dtype
    # Skip-Layernorm 2
    pf_beta = trt.PluginField("beta", weights_dict[prefix + 'output_layernorm_beta'], trt.PluginFieldType.FLOAT32)
    pf_gamma = trt.PluginField("gamma", weights_dict[prefix + 'output_layernorm_gamma'], trt.PluginFieldType.FLOAT32)
    pf_bias = trt.PluginField("bias", Bout, trt.PluginFieldType.FLOAT32)
    fields = [pf_ld, pf_beta, pf_gamma, pf_type]
    pfc = trt.PluginFieldCollection(fields)
    skipln_plug = pc_skln.create_plugin("skipln", pfc)
    skln1_out.dtype = dtype
    skipln_inputs = [fc_out_out, skln1_out]
    skln2 = network.add_plugin_v2(skipln_inputs, skipln_plug)
    skln2.name = prefix + 'skln_2'
    skln2_out = skln2.get_output(0)

    return skln2_out


def bert_squad_fp16_var_seqlen(network, weights_dict, cfg, input_shape, cu_seqlens_shape):
    """Build BERT network in FP16 mode with var seqlen."""

    # instantiate all the plugins
    plg_registry = trt.get_plugin_registry()

    pc_emb = plg_registry.get_plugin_creator("CustomEmbLayerNormPluginDynamic", "2", "")

    wbeta = trt.PluginField("bert_embeddings_layernorm_beta", weights_dict["bert_embeddings_layernorm_beta"], trt.PluginFieldType.FLOAT32)
    wgamma = trt.PluginField("bert_embeddings_layernorm_gamma", weights_dict["bert_embeddings_layernorm_gamma"], trt.PluginFieldType.FLOAT32)
    wwordemb = trt.PluginField("bert_embeddings_word_embeddings", weights_dict["bert_embeddings_word_embeddings"], trt.PluginFieldType.FLOAT32)
    wtokemb = trt.PluginField("bert_embeddings_token_type_embeddings", weights_dict["bert_embeddings_token_type_embeddings"], trt.PluginFieldType.FLOAT32)
    wposemb = trt.PluginField("bert_embeddings_position_embeddings", weights_dict["bert_embeddings_position_embeddings"], trt.PluginFieldType.FLOAT32)

    output_fp16 = trt.PluginField("output_fp16", np.array([int(trt.float16)]).astype(np.int32), trt.PluginFieldType.INT32)

    pfc = trt.PluginFieldCollection([wbeta, wgamma, wwordemb, wtokemb, wposemb, output_fp16])
    embln_plugin = pc_emb.create_plugin("embeddings", pfc)

    dtype = trt.float16

    input_ids = network.add_input(name="input_ids", dtype=trt.int32, shape=input_shape)
    segment_ids = network.add_input(name="segment_ids", dtype=trt.int32, shape=input_shape)

    cu_seqlens = network.add_input(name="cu_seqlens", dtype=trt.int32, shape=cu_seqlens_shape)

    # dummy input used to indicate maximum sequence length to plugins
    max_seqlen = network.add_input(name="max_seqlen", dtype=trt.int32, shape=(-1,))

    inputs = [input_ids, segment_ids, cu_seqlens, max_seqlen]
    emb_layer = network.add_plugin_v2(inputs, embln_plugin)
    emb_layer.name = 'embln'

    embeddings = emb_layer.get_output(0)

    mask = emb_layer.get_output(1)
    embeddings.dtype = dtype

    layer = 0
    for layer in range(cfg.L):
        embeddings = bert_encoder_layer_fp16_var_seqlen(cfg, max_seqlen, weights_dict, network, embeddings, cu_seqlens, layer, mask)

    Wsquad = weights_dict['cls_squad_output_weights']
    Bsquad = weights_dict['cls_squad_output_bias']

    squad_output = network.add_fully_connected(embeddings, 2, Wsquad, Bsquad)
    squad_output.name = 'squad_logits'
    logits = squad_output.get_output(0)

    mark(network, logits, trt.float16)
