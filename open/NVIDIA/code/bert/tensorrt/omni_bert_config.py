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

import logging
from typing import List, Dict, Any, Union, Optional
import os
import copy
from transformers import PretrainedConfig, BertConfig


class BertOmniConfigBase:
    def to_dict(self) -> Dict[str, Any]:
        return self.to_dict_copy_recursive(self)

    @staticmethod
    def to_dict_copy_recursive(d: Any) -> Union[Dict[str, Any], Any]:
        if isinstance(d, BertOmniConfigBase):
            return BertOmniConfigBase.to_dict_copy_recursive(d.__dict__)
        if isinstance(d, dict):
            new_dict = {}
            for key, val in d.items():
                new_dict[key] = BertOmniConfigBase.to_dict_copy_recursive(val)                

            if new_dict.get("torch_dtype", None) is not None and not isinstance(new_dict["torch_dtype"], str):
                new_dict["torch_dtype"] = str(new_dict["torch_dtype"]).split(".")[1]
            return new_dict
        if isinstance(d, list):
            new_lst = []
            for item in d:
                new_lst.append(BertOmniConfigBase.to_dict_copy_recursive(item))
            return new_lst
        return copy.deepcopy(d)

class BertOmniEmbeddingsConfig(BertOmniConfigBase):
    def __init__(
        self, 
        vocab_size = 30522,
        hidden_size = 768,
        pad_token_id = 0,
        max_position_embeddings = 512,
        type_vocab_size = 2,
        layer_norm_eps = 1e-12,
        hidden_dropout_prob = 0.1, 
        position_embedding_type = "absolute"
    ):
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        self.pad_token_id = pad_token_id
        self.max_position_embeddings = max_position_embeddings 
        self.type_vocab_size = type_vocab_size
        self.layer_norm_eps = layer_norm_eps 
        self.hidden_dropout_prob = hidden_dropout_prob  
        self.position_embedding_type = position_embedding_type
 

class BertOmniIntermediateConfig(BertOmniConfigBase):
    def __init__(
        self, 
        hidden_size=768,
        intermediate_size=3072,
        hidden_act="gelu",
    ):
        self.hidden_size = hidden_size
        self.intermediate_size = intermediate_size
        self.hidden_act = hidden_act 


class BertOmniOutputConfig(BertOmniConfigBase):
    def __init__(
        self, 
        intermediate_size=3072,
        hidden_size=768,
        layer_norm_eps = 1e-12,
        hidden_dropout_prob=0.1,
    ):
        self.intermediate_size = intermediate_size
        self.hidden_size = hidden_size 
        self.layer_norm_eps = layer_norm_eps  
        self.hidden_dropout_prob = hidden_dropout_prob   


class BertOmniSelfAttentionConfig(BertOmniConfigBase):
    def __init__(
        self, 
        num_attention_heads=12,
        input_hidden_size=768,
        output_hidden_size=768,
        attention_probs_dropout_prob=0.1,
        max_position_embeddings=512,
        is_decoder=False
    ):
        self.num_attention_heads = num_attention_heads 
        self.input_hidden_size = input_hidden_size
        self.output_hidden_size = output_hidden_size
        self.attention_probs_dropout_prob = attention_probs_dropout_prob
        self.max_position_embeddings = max_position_embeddings 
        self.is_decoder = is_decoder


class BertOmniSelfOutputConfig(BertOmniConfigBase):
    def __init__(
        self, 
        input_hidden_size=768,
        output_hidden_size=768,
        layer_norm_eps=1e-12,
        hidden_dropout_prob=0.1,
    ):
        self.input_hidden_size = input_hidden_size
        self.output_hidden_size = output_hidden_size
        self.layer_norm_eps = layer_norm_eps
        self.hidden_dropout_prob = hidden_dropout_prob


class BertOmniAttentionConfig(BertOmniConfigBase):
    def __init__(
        self, 
        self_attention: Union[Dict[str, Any], BertOmniSelfAttentionConfig]={},
        self_output: Union[Dict[str, Any], BertOmniSelfOutputConfig]={},
    ):
        self.self_attention = self_attention if isinstance(self_attention, BertOmniSelfAttentionConfig) else BertOmniSelfAttentionConfig(**self_attention)
        self.self_output = self_output if isinstance(self_output, BertOmniSelfOutputConfig) else BertOmniSelfOutputConfig(**self_output)

class BertOmniLayerConfig(BertOmniConfigBase):
    def __init__(
        self,
        chunk_size_feed_forward=0,
        is_decoder=False,
        add_cross_attention=False,
        attention: Union[Dict[str, Any], BertOmniAttentionConfig]={},
        crossattention: Optional[Union[Dict[str, Any], BertOmniAttentionConfig]]=None,
        intermediate: Union[Dict[str, Any], BertOmniIntermediateConfig]={},
        output: Union[Dict[str, Any], BertOmniOutputConfig]={},
    ):
        self.chunk_size_feed_forward = chunk_size_feed_forward
        self.is_decoder = is_decoder
        self.add_cross_attention = add_cross_attention
        self.attention = attention if isinstance(attention, BertOmniAttentionConfig) else BertOmniAttentionConfig(**attention)
        if crossattention is not None:
            self.crossattention = crossattention if isinstance(crossattention, BertOmniAttentionConfig) else BertOmniAttentionConfig(**crossattention)
        self.intermediate = intermediate  if isinstance(intermediate, BertOmniIntermediateConfig) else BertOmniIntermediateConfig(**intermediate)
        self.output = output if isinstance(output, BertOmniOutputConfig) else BertOmniOutputConfig(**output)

class BertOmniPoolerConfig(BertOmniConfigBase):
    def __init__(
        self, 
        hidden_size=768,
    ):
        self.hidden_size = hidden_size


class BertOmniEncoderConfig(BertOmniConfigBase):
    def __init__(
        self, 
        layers: Union[List[Dict[str, Any]], List[BertOmniLayerConfig]]=[],
        add_cross_attention=False,
    ):
        self.layers = [layer if isinstance(layer, BertOmniLayerConfig) else BertOmniLayerConfig(**layer) for layer in layers]
        self.add_cross_attention = add_cross_attention


class BertOmniModelConfig(PretrainedConfig, BertOmniConfigBase):

    model_type = "bert"

    def __init__(
        self,
        embeddings: Union[Dict[str, Any], BertOmniEmbeddingsConfig]={},
        encoder: Union[Dict[str, Any], BertOmniEncoderConfig]={},
        pooler: Optional[Union[Dict[str, Any], BertOmniPoolerConfig]]=None,
        **kwargs
    ):
        super().__init__(**kwargs)

        self.embeddings = embeddings if isinstance(embeddings, BertOmniEmbeddingsConfig) else BertOmniEmbeddingsConfig(**embeddings)
        self.encoder = encoder if isinstance(encoder, BertOmniEncoderConfig) else BertOmniEncoderConfig(**encoder)
        if pooler is not None:
            self.pooler = pooler if isinstance(pooler, BertOmniPoolerConfig) else BertOmniPoolerConfig(**pooler)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        return BertOmniConfigBase.to_dict_copy_recursive(d)


class BertOmniQAModelConfig(PretrainedConfig, BertOmniConfigBase):
    r"""
    This is the configuration class to store the configuration of a omnimized Bert Question Answering Model.
    """

    model_type = "bert"

    @classmethod
    def from_pretrained(cls, pretrained_model_name_or_path: Union[str, os.PathLike], **kwargs) -> PretrainedConfig:
        config_dict, kwargs = PretrainedConfig.get_config_dict(pretrained_model_name_or_path, **kwargs)
        if "model_type" in config_dict and hasattr(cls, "model_type") and config_dict["model_type"] != cls.model_type:
            logger.warning(
                f"You are using a model of type {config_dict['model_type']} to instantiate a model of type "
                f"{cls.model_type} for class {cls}. This is not supported for all configurations of models and can yield errors."
            )
        return cls.from_dict(config_dict, **kwargs)

    @classmethod
    def from_json_file(cls, json_file: Union[str, os.PathLike]) -> PretrainedConfig:
        config_dict = cls._dict_from_json_file(json_file)
        return cls.from_dict(config_dict)

    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any], **kwargs) -> PretrainedConfig:
        if "config_class" in config_dict and config_dict["config_class"] == BertOmniQAModelConfig.__name__:
            return super(BertOmniQAModelConfig, cls).from_dict(config_dict, **kwargs)
        else:
            config = BertConfig.from_dict(config_dict, **kwargs)
            config.hidden_size_per_head = config.hidden_size // config.num_attention_heads 
            assert(config.hidden_size_per_head * config.num_attention_heads == config.hidden_size)
            config.qkv_size = 3 * config.hidden_size

            return config

    def __init__(
        self,
        bert: Union[Dict[str, Any], BertOmniModelConfig]={},
        **kwargs
    ):
        super().__init__( **kwargs)
        self.config_class = BertOmniQAModelConfig.__name__
        self.bert = bert if isinstance(bert, BertOmniModelConfig) else BertOmniModelConfig(**bert)

    def to_dict(self) -> Dict[str, Any]:
        d = super().to_dict()
        return BertOmniConfigBase.to_dict_copy_recursive(d)
