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

import os
import base64
import dragon
import multiprocessing as mp
import pickle
import pytest
import unittest

import dragon.fli as fli
from dragon.channels import Channel
import dragon.globalservices.api_setup as serv

from smartsim._core.mli.infrastructure import MemoryFeatureStore, DragonFeatureStore, DragonDict
from smartsim._core.mli.workermanager import EnvironmentConfigLoader

# mem_fs = MemoryFeatureStore()
# mem_fs._storage["key"] = b"value_bytes"

# mem_fs_2 = MemoryFeatureStore()
# mem_fs_2._storage["another key"] = b"value_bytes"

# serv.connect_to_infrastructure()
# mp.set_start_method("dragon")

# chan = Channel.make_process_local()
# _fli = fli.FLInterface(main_ch=chan)
# _fli.write_bytes(b"message")

# @pytest.fixture(scope="module")
# def prep_env():
#     serv.connect_to_infrastructure()

# @pytest.mark.parametrize(
#     "fs, queue",
#     [
#         (mem_fs, _fli),
#         (mem_fs_2, _fli),
#         (None, None),
#     ]
# # )
# class TestEnvLoader(unittest.TestCase):
#     def test_env_loader():
#         expected_value = b"value_bytes"
#         expected_key = "key"
#         dragon_dict = DragonDict()
#         dragon_dict[expected_key] = expected_value
#         fs = DragonFeatureStore(dragon_dict)
#         fs[expected_key] = expected_value
#         chan = Channel.make_process_local()
#         queue = fli.FLInterface(main_ch=chan)
#         sender = queue.sendh(use_main_as_stream_channel=True)
#         sender.send_bytes(b"bytessss")
#         os.environ["SSFeatureStore"] = base64.b64encode(pickle.dumps(fs)).decode('utf-8')
#         os.environ["SSQueue"] = base64.b64encode(pickle.dumps(queue)).decode('utf-8')
#         config = EnvironmentConfigLoader()
#         config_store = config.get_feature_store()
#         assert config_store[expected_key] == expected_value
#         config_queue = config.get_queue()
#         assert config_queue.__class__ == queue.__class__

def test_env_loader():
    expected_value = b"value_bytes"
    expected_key = "key"
    dragon_dict = DragonDict()
    dragon_dict[expected_key] = expected_value
    fs = DragonFeatureStore(dragon_dict)
    chan = Channel.make_process_local()
    queue = fli.FLInterface(main_ch=chan)
    sender = queue.sendh(use_main_as_stream_channel=True)
    sender.send_bytes(b"bytessss")
    os.environ["SSFeatureStore"] = base64.b64encode(pickle.dumps(fs)).decode('utf-8')
    os.environ["SSQueue"] = base64.b64encode(pickle.dumps(queue)).decode('utf-8')
    config = EnvironmentConfigLoader()
    config_store = config.get_feature_store()
    assert config_store[expected_key] == expected_value
    config_queue = config.get_queue()
    assert config_queue.__class__ == queue.__class__

    print("Test complete")

# def test_env_loader(fs, queue):
#     expected_value = b"value"
#     expected_key = "key"
#     os.environ["SSFeatureStore"] = base64.b64encode(pickle.dumps(fs)).decode('utf-8')
#     os.environ["SSQueue"] = base64.b64encode(pickle.dumps(queue)).decode('utf-8')
#     config = EnvironmentConfigLoader()
#     config_store = config.get_feature_store()
#     assert config_store._storage[expected_key] == expected_value
#     config_queue = config.get_queue()
#     assert config_queue.__class__ == queue.__class__

#     print("Test complete")


# if __name__ == "__main__":
#     mp.set_start_method("dragon")
#     serv.connect_to_infrastructure()
#     # chan = Channel.make_process_local()
    
#     # _fli = fli.FLInterface(main_ch=chan)
#     # chan2 = Channel.make_process_local()
#     # sender = _fli.sendh(chan2)

#     # sender.send_bytes(b"bytessss")
#     # mem_fs = MemoryFeatureStore()
#     # mem_fs._storage["key"] = b"value_bytes"

#     dragon_dict = DragonDict()
#     dragon_dict["key"] = b"value"
#     fs = DragonFeatureStore(dragon_dict)
#     chan = Channel.make_process_local()
#     queue = fli.FLInterface(main_ch=chan)
#     sender = queue.sendh(use_main_as_stream_channel=True)
#     sender.send_bytes(b"bytessss")


#     test_env_loader(fs, queue)

    

# if __name__ == "__main__":
#     mp.set_start_method('dragon')
#     unittest.main()
if __name__ == "__main__":
    mp.set_start_method('dragon')
    serv.connect_to_infrastructure()
    pytest.main()
    # test_env_loader()