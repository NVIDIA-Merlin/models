import pandas as pd
import numpy as np
import torch
from torch import nn

from merlin.models.torch.predict import (
    module_encode,
    encode_output
)
from merlin.dataloader.torch import Loader
from merlin.table import TensorTable


class TensorOutputModel(nn.Module):
    def forward(self, x):
        return x * 2


class DictOutputModel(nn.Module):
    def forward(self, x):
        return {'testing': x * 2}
    
    
class Test_batch_predict:
    def test_something(self, music_streaming_data):
        loader = Loader(music_streaming_data, batch_size=128)
        
        # For batch-predict we have the following steps:
        # 1. Call map_partitions on loader.dataset
        # 2. For each partition, create a new instance of Loader with the partition
        
        input_table = TensorTable(loader.peek()[0])
        output_table = TensorTable(loader.peek()[1])
        
        b = None
        


class Test_module_encode:
    def test_with_tensor_output(self):
        module = TensorOutputModel()
        inputs = torch.tensor([1, 2, 3, 4])
        output = module_encode(module, inputs)
        
        np.testing.assert_array_equal(output, np.array([2, 4, 6, 8]))

    def test_with_dict_output(self):
        module = DictOutputModel()
        inputs = torch.tensor([1, 2, 3, 4])
        output = module_encode(module, inputs)
        
        if hasattr(output, "to_pandas"):
            output = output.to_pandas()
        
        expected_output = pd.DataFrame({'testing': [2, 4, 6, 8]})
        pd.testing.assert_frame_equal(output, expected_output)


class Test_encode_output:
    def test_1d_tensor(self):
        tensor = torch.tensor([1, 2, 3, 4])
        expected_output = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(encode_output(tensor), expected_output)

    def test_2d_tensor_single_column(self):
        tensor = torch.tensor([[1], [2], [3], [4]])
        expected_output = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(encode_output(tensor), expected_output)

    def test_2d_tensor_multi_columns(self):
        tensor = torch.tensor([[1, 2], [3, 4], [5, 6], [7, 8]])
        expected_output = np.array([[1, 2], [3, 4], [5, 6], [7, 8]])
        np.testing.assert_array_equal(encode_output(tensor), expected_output)

    def test_3d_tensor(self):
        tensor = torch.tensor([[[1], [2]], [[3], [4]]])
        expected_output = np.array([[[1], [2]], [[3], [4]]])
        np.testing.assert_array_equal(encode_output(tensor), expected_output)

    def test_float_tensor(self):
        tensor = torch.tensor([1.0, 2.0, 3.0, 4.0])
        expected_output = np.array([1.0, 2.0, 3.0, 4.0])
        np.testing.assert_array_equal(encode_output(tensor), expected_output)

    def test_device_tensor(self):
        tensor = torch.tensor([1, 2, 3, 4], device='cuda')
        expected_output = np.array([1, 2, 3, 4])
        np.testing.assert_array_equal(encode_output(tensor), expected_output)