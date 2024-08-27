# BSD 2-Clause License
#
# Copyright (c) 2021-2024, Hewlett Packard Enterprise
# All rights reserved.
#
# Redistribution and use in source and binary forms, with or without
# modification, are permitted provided that the following conditions are met:
#
# 1. Redistributions of source code must retain the above copyright notice, this
#    list of conditions and the following disclaimer.
#
# 2. Redistributions in binary form must reproduce the above copyright notice,
#    this list of conditions and the following disclaimer in the documentation
#    and/or other materials provided with the distribution.
#
# THIS SOFTWARE IS PROVIDED BY THE COPYRIGHT HOLDERS AND CONTRIBUTORS "AS IS"
# AND ANY EXPRESS OR IMPLIED WARRANTIES, INCLUDING, BUT NOT LIMITED TO, THE
# IMPLIED WARRANTIES OF MERCHANTABILITY AND FITNESS FOR A PARTICULAR PURPOSE ARE
# DISCLAIMED. IN NO EVENT SHALL THE COPYRIGHT HOLDER OR CONTRIBUTORS BE LIABLE
# FOR ANY DIRECT, INDIRECT, INCIDENTAL, SPECIAL, EXEMPLARY, OR CONSEQUENTIAL
# DAMAGES (INCLUDING, BUT NOT LIMITED TO, PROCUREMENT OF SUBSTITUTE GOODS OR
# SERVICES; LOSS OF USE, DATA, OR PROFITS; OR BUSINESS INTERRUPTION) HOWEVER
# CAUSED AND ON ANY THEORY OF LIABILITY, WHETHER IN CONTRACT, STRICT LIABILITY,
# OR TORT (INCLUDING NEGLIGENCE OR OTHERWISE) ARISING IN ANY WAY OUT OF THE USE
# OF THIS SOFTWARE, EVEN IF ADVISED OF THE POSSIBILITY OF SUCH DAMAGE.

import io
import sys

import pytest
import torch
from torch import nn
from torch.nn import functional as F
import numpy as np

from dragon.data.ddict.ddict import DDict

from smartsim._core.mli.infrastructure.storage.featurestore import FeatureStoreKey
from smartsim._core.mli.infrastructure.storage.dragonfeaturestore import DragonFeatureStore
from smartsim._core.mli.infrastructure.worker.torch_worker import TorchWorker
from smartsim._core.mli.infrastructure.worker.worker import (
    ExecuteResult,
    FetchInputResult,
    FetchModelResult,
    InferenceRequest,
    LoadModelResult,
    TransformInputResult,
)
from smartsim._core.mli.message_handler import MessageHandler
from smartsim.log import get_logger

from memory_profiler import profile

logger = get_logger(__name__)
# The tests in this file belong to the group_a group
pytestmark = pytest.mark.group_a


# # simple MNIST in PyTorch
# class Net(nn.Module):
#     def __init__(self):
#         super(Net, self).__init__()
#         self.conv1 = nn.Conv2d(1, 32, 3, 1)
#         self.conv2 = nn.Conv2d(32, 64, 3, 1)
#         self.dropout1 = nn.Dropout(0.25)
#         self.dropout2 = nn.Dropout(0.5)
#         self.fc1 = nn.Linear(9216, 128)
#         self.fc2 = nn.Linear(128, 10)

#     def forward(self, x):
#         x = self.conv1(x)
#         x = F.relu(x)
#         x = self.conv2(x)
#         x = F.relu(x)
#         x = F.max_pool2d(x, 2)
#         x = self.dropout1(x)
#         x = torch.flatten(x, 1)
#         x = self.fc1(x)
#         x = F.relu(x)
#         x = self.dropout2(x)
#         x = self.fc2(x)
#         output = F.log_softmax(x, dim=1)
#         return output


# torch_device = {"cpu": "cpu", "gpu": "cuda"}


# def get_batch() -> torch.Tensor:
#     return torch.rand(20, 1, 28, 28)

# @profile
# def create_torch_model():
#     n = Net()
#     example_forward_input = get_batch()
#     module = torch.jit.trace(n, example_forward_input)
#     model_buffer = io.BytesIO()
#     torch.jit.save(module, model_buffer)
#     return model_buffer.getvalue()

# @profile
# def get_request() -> InferenceRequest:

#     tensors = [get_batch() for _ in range(2)]
#     tensor_numpy = [tensor.numpy() for tensor in tensors]
#     serialized_tensors_descriptors = [
#         MessageHandler.build_tensor_descriptor("c", "float32", list(tensor.shape))
#         for tensor in tensors
#     ]

#     return InferenceRequest(
#         model_key=FeatureStoreKey(key="model", descriptor="xyz"),
#         callback=None,
#         raw_inputs=tensor_numpy,
#         input_keys=None,
#         input_meta=serialized_tensors_descriptors,
#         output_keys=None,
#         raw_model=create_torch_model(),
#         batch_size=0,
#     )


# sample_request: InferenceRequest = get_request()
# worker = TorchWorker()


# def test_profile_worker_load_model():
#     ...


# def test_profile_worker_fetch_model():
#     ...

# @profile(precision=5)
# def test_profile_getitem_dragonfeaturestore():
#     mgr_per_node = 1
#     num_nodes = 2
#     mem_per_node = 1024**3
#     total_mem = num_nodes * mem_per_node

#     storage = DDict(
#         managers_per_node=mgr_per_node,
#         n_nodes=num_nodes,
#         total_mem=total_mem,
#     )

#     dfs = DragonFeatureStore(storage)

#     item = np.random.rand(1024,1024,3)

#     dfs["key"] = item

#     the_item = dfs["key"]

#     assert np.array_equal(item, the_item)


# @profile(precision=5)
# def test_profile_ddict():
#     mgr_per_node = 1
#     num_nodes = 2
#     mem_per_node = 1024**3
#     total_mem = num_nodes * mem_per_node

#     storage = DDict(
#         managers_per_node=mgr_per_node,
#         n_nodes=num_nodes,
#         total_mem=total_mem,
#     )

#     item = np.random.rand(1024,1024,3).tobytes()

#     storage["key"] = item

#     the_item = storage["key"]

#     assert item == the_item
#     assert type(item) == bytes

@profile(precision=5)
def test_profile_toch_jit_load_load_model():
    model_bytes = b'a'*1024*1024
    buffer = io.BytesIO(initial_bytes=model_bytes)
    with torch.no_grad():
        model = torch.jit.load(buffer, map_location="cpu")
        model.eval()

    assert False