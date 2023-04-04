import torch
from torch import nn
from torchrec.sparse.jagged_tensor import KeyedJaggedTensor, JaggedTensor
from torchrec.datasets.utils import Batch
from torchrec.modules.embedding_modules import EmbeddingBagCollection

from merlin.schema import Schema
from merlin.models.torch.transforms.aggregation import ConcatFeatures


class ToKeyedJaggedTensor(nn.Module):
    def __init__(self, schema: Schema):
        super().__init__()
        self.schema = schema        
    
    def forward(self, inputs) -> KeyedJaggedTensor:
        jagged_dict = {}
        
        for col in self.schema:
            if col.is_ragged:
                jagged_dict[col.name] = JaggedTensor(
                    inputs[col.name + "__values"], offsets=inputs[col.name + "__offsets"]
                )
            else:
                if col.name not in inputs:
                    continue
                feature = inputs[col.name]
                jagged_dict[col.name] = JaggedTensor(
                    feature, lengths=torch.ones_like(feature, dtype=torch.int)
                )
            
        output = KeyedJaggedTensor.from_jt_dict(jagged_dict)
        
        return output


class ToTorchRecBatch(nn.Module):
    def __init__(self, schema: Schema, dense_concat=ConcatFeatures()):
        super().__init__()
        self.schema = schema
        self.dense_concat = dense_concat
        
    def forward(self, inputs) -> Batch:
        dense_dict = {}
        jagged_dict = {}
        
        for col in self.schema:
            if col.is_ragged or col.is_list:
                values = inputs.get(col.name + "__values", None)
                offsets = inputs.get(col.name + "__offsets", None)
                
                if not values or not offsets:
                    continue
                
                jagged_dict[col.name] = JaggedTensor(values, offsets=offsets)
            else:
                if col.name not in inputs:
                    continue
                dense_dict[col.name] = inputs[col.name]
                
        dense = self.dense_concat(dense_dict)
        key_jagged = KeyedJaggedTensor.from_jt_dict(jagged_dict)
                
        return Batch(dense_features=dense, sparse_features=key_jagged, targets=None)        
        
        
# class Embeddings()