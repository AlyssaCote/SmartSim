import pathlib
import typing as t

import pytest
import torch

import smartsim.error as sse
from smartsim._core.mli import workermanager as mli
from smartsim._core.mli.message_handler import MessageHandler
from smartsim._core.utils import installed_redisai_backends

# The tests in this file belong to the group_b group
pytestmark = pytest.mark.group_b

# retrieved from pytest fixtures
is_dragon = pytest.test_launcher == "dragon"
torch_available = "torch" in installed_redisai_backends()


@pytest.fixture
def persist_torch_model(test_dir: str) -> pathlib.Path:
    test_path = pathlib.Path(test_dir)
    model_path = test_path / "basic.pt"

    model = torch.nn.Linear(2, 1)
    torch.save(model, model_path)

    return model_path


# todo: move deserialization tests into suite for worker manager where serialization occurs


# @pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
# def test_deserialize_direct_request(persist_torch_model: pathlib.Path) -> None:
#     """Verify that a direct requestis deserialized properly"""
#     worker = mli.IntegratedTorchWorker
#     # feature_store = mli.MemoryFeatureStore()

#     model_bytes = persist_torch_model.read_bytes()
#     input_tensor = torch.randn(2)

#     expected_device = "cpu"
#     expected_callback_channel = b"faux_channel_descriptor_bytes"
#     callback_channel = mli.DragonCommChannel.find(expected_callback_channel)

#     message_tensor_input = MessageHandler.build_tensor(
#         input_tensor, "c", "float32", [2]
#     )

#     request = MessageHandler.build_request(
#         reply_channel=callback_channel.descriptor,
#         model=model_bytes,
#         device=expected_device,
#         inputs=[message_tensor_input],
#         outputs=[],
#         custom_attributes=None,
#     )

#     msg_bytes = MessageHandler.serialize_request(request)

#     inference_request = worker.deserialize(msg_bytes)
#     assert inference_request.device == expected_device
#     assert inference_request.callback._descriptor == expected_callback_channel


# @pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
# def test_deserialize_indirect_request(persist_torch_model: pathlib.Path) -> None:
#     """Verify that an indirect request is deserialized correctly"""
#     worker = mli.IntegratedTorchWorker
#     # feature_store = mli.MemoryFeatureStore()

#     model_key = "persisted-model"
#     # model_bytes = persist_torch_model.read_bytes()
#     # feature_store[model_key] = model_bytes

#     input_key = f"demo-input"
#     # input_tensor = torch.randn(2)
#     # feature_store[input_key] = input_tensor

#     expected_device = "cpu"
#     expected_callback_channel = b"faux_channel_descriptor_bytes"
#     callback_channel = mli.DragonCommChannel.find(expected_callback_channel)

#     output_key = f"demo-output"

#     message_tensor_output_key = MessageHandler.build_tensor_key(output_key)
#     message_tensor_input_key = MessageHandler.build_tensor_key(input_key)
#     message_model_key = MessageHandler.build_model_key(model_key)

#     request = MessageHandler.build_request(
#         reply_channel=callback_channel.descriptor,
#         model=message_model_key,
#         device=expected_device,
#         inputs=[message_tensor_input_key],
#         outputs=[message_tensor_output_key],
#         custom_attributes=None,
#     )

#     msg_bytes = MessageHandler.serialize_request(request)

#     inference_request = worker.deserialize(msg_bytes)
#     assert inference_request.device == expected_device
#     assert inference_request.callback._descriptor == expected_callback_channel


# @pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
# def test_deserialize_mixed_mode_indirect_inputs(
#     persist_torch_model: pathlib.Path,
# ) -> None:
#     """Verify that a mixed mode (combining direct and indirect inputs, models, outputs)
#     with indirect inputs is deserialized correctly"""
#     worker = mli.IntegratedTorchWorker
#     # feature_store = mli.MemoryFeatureStore()

#     # model_key = "persisted-model"
#     model_bytes = persist_torch_model.read_bytes()
#     # feature_store[model_key] = model_bytes

#     input_key = f"demo-input"
#     # input_tensor = torch.randn(2)
#     # feature_store[input_key] = input_tensor

#     expected_device = "cpu"
#     expected_callback_channel = b"faux_channel_descriptor_bytes"
#     callback_channel = mli.DragonCommChannel.find(expected_callback_channel)

#     output_key = f"demo-output"

#     message_tensor_output_key = MessageHandler.build_tensor_key(output_key)
#     message_tensor_input_key = MessageHandler.build_tensor_key(input_key)
#     # message_model_key = MessageHandler.build_model_key(model_key)

#     request = MessageHandler.build_request(
#         reply_channel=callback_channel.descriptor,
#         model=model_bytes,
#         device=expected_device,
#         inputs=[message_tensor_input_key],
#         # outputs=[message_tensor_output_key],
#         outputs=[],
#         custom_attributes=None,
#     )

#     msg_bytes = MessageHandler.serialize_request(request)

#     inference_request = worker.deserialize(msg_bytes)
#     assert inference_request.device == expected_device
#     assert inference_request.callback._descriptor == expected_callback_channel


# @pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
# def test_deserialize_mixed_mode_indirect_outputs(
#     persist_torch_model: pathlib.Path,
# ) -> None:
#     """Verify that a mixed mode (combining direct and indirect inputs, models, outputs)
#     with indirect outputs is deserialized correctly"""
#     worker = mli.IntegratedTorchWorker
#     # feature_store = mli.MemoryFeatureStore()

#     # model_key = "persisted-model"
#     model_bytes = persist_torch_model.read_bytes()
#     # feature_store[model_key] = model_bytes

#     input_key = f"demo-input"
#     input_tensor = torch.randn(2)
#     # feature_store[input_key] = input_tensor

#     expected_device = "cpu"
#     expected_callback_channel = b"faux_channel_descriptor_bytes"
#     callback_channel = mli.DragonCommChannel.find(expected_callback_channel)

#     output_key = f"demo-output"

#     message_tensor_output_key = MessageHandler.build_tensor_key(output_key)
#     # message_tensor_input_key = MessageHandler.build_tensor_key(input_key)
#     # message_model_key = MessageHandler.build_model_key(model_key)
#     message_tensor_input = MessageHandler.build_tensor(
#         input_tensor, "c", "float32", [2]
#     )

#     request = MessageHandler.build_request(
#         reply_channel=callback_channel.descriptor,
#         model=model_bytes,
#         device=expected_device,
#         inputs=[message_tensor_input],
#         # outputs=[message_tensor_output_key],
#         outputs=[message_tensor_output_key],
#         custom_attributes=None,
#     )

#     msg_bytes = MessageHandler.serialize_request(request)

#     inference_request = worker.deserialize(msg_bytes)
#     assert inference_request.device == expected_device
#     assert inference_request.callback._descriptor == expected_callback_channel


# @pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
# def test_deserialize_mixed_mode_indirect_model(
#     persist_torch_model: pathlib.Path,
# ) -> None:
#     """Verify that a mixed mode (combining direct and indirect inputs, models, outputs)
#     with indirect outputs is deserialized correctly"""
#     worker = mli.IntegratedTorchWorker
#     # feature_store = mli.MemoryFeatureStore()

#     model_key = "persisted-model"
#     # model_bytes = persist_torch_model.read_bytes()
#     # feature_store[model_key] = model_bytes

#     # input_key = f"demo-input"
#     input_tensor = torch.randn(2)
#     # feature_store[input_key] = input_tensor

#     expected_device = "cpu"
#     expected_callback_channel = b"faux_channel_descriptor_bytes"
#     callback_channel = mli.DragonCommChannel.find(expected_callback_channel)

#     output_key = f"demo-output"

#     # message_tensor_output_key = MessageHandler.build_tensor_key(output_key)
#     # message_tensor_input_key = MessageHandler.build_tensor_key(input_key)
#     message_model_key = MessageHandler.build_model_key(model_key)
#     message_tensor_input = MessageHandler.build_tensor(
#         input_tensor, "c", "float32", [2]
#     )

#     request = MessageHandler.build_request(
#         reply_channel=callback_channel.descriptor,
#         model=message_model_key,
#         device=expected_device,
#         inputs=[message_tensor_input],
#         # outputs=[message_tensor_output_key],
#         outputs=[],
#         custom_attributes=None,
#     )

#     msg_bytes = MessageHandler.serialize_request(request)

#     inference_request = worker.deserialize(msg_bytes)
#     assert inference_request.device == expected_device
#     assert inference_request.callback._descriptor == expected_callback_channel


# @pytest.mark.skipif(not torch_available, reason="Torch backend is not installed")
# def test_serialize(test_dir: str, persist_torch_model: pathlib.Path) -> None:
#     """Verify that the worker correctly executes reply serialization"""
#     worker = mli.IntegratedTorchWorker

#     reply = mli.InferenceReply()
#     reply.output_keys = ["foo", "bar"]

#     # use the worker implementation of reply serialization to get bytes for
#     # use on the callback channel
#     reply_bytes = worker.serialize_reply(reply)
#     assert reply_bytes is not None

#     # deserialize to verity the mapping in the worker.serialize_reply was correct
#     actual_reply = MessageHandler.deserialize_response(reply_bytes)

#     actual_tensor_keys = [tk.key for tk in actual_reply.result.keys]
#     assert set(actual_tensor_keys) == set(reply.output_keys)
#     assert actual_reply.status == 200
#     assert actual_reply.statusMessage == "success"