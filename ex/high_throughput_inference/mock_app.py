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

# isort: off
import dragon
from dragon import fli
from dragon.channels import Channel
import dragon.channels
from dragon.data.ddict.ddict import DDict
from dragon.globalservices.api_setup import connect_to_infrastructure
from dragon.utils import b64decode, b64encode

# isort: on

import argparse
import io
import numpy
import os
import time
import torch
import cloudpickle as cp
import pickle

from mpi4py import MPI
from smartsim._core.mli.message_handler import MessageHandler
from smartsim.log import get_logger
from smartsim._core.utils.timings import PerfTimer

torch.set_num_interop_threads(16)
torch.set_num_threads(1)

logger = get_logger("App")

CHECK_RESULTS_AND_MAKE_ALL_SLOWER = False

class ProtoClient:
    def __init__(self, timing_on: bool):
        comm = MPI.COMM_WORLD
        rank = comm.Get_rank()
        connect_to_infrastructure()
        ddict_str = os.environ["SS_DRG_DDICT"]
        self._ddict = DDict.attach(ddict_str)
        to_worker_fli_str = None
        while to_worker_fli_str is None:
            try:
                to_worker_fli_str = self._ddict["to_worker_fli"]
                self._to_worker_fli = fli.FLInterface.attach(to_worker_fli_str)
            except KeyError:
                time.sleep(1)
        self._from_worker_ch = Channel.make_process_local()
        self._from_worker_ch_serialized = self._from_worker_ch.serialize()
        self._to_worker_ch = Channel.make_process_local()

        self._perf_timer: PerfTimer = PerfTimer(debug=False, timing_on=timing_on, prefix=f"a{rank}_")

    def run_model(self, model: bytes | str, batch: torch.Tensor):
        tensors = [batch.numpy()]
        self._perf_timer.start_timings("batch_size", batch.shape[0])
        built_tensor_desc = MessageHandler.build_tensor_descriptor(
            "c", "float32", list(batch.shape))
        self._perf_timer.measure_time("build_tensor_descriptor")
        if isinstance(model, str):
            model_arg = MessageHandler.build_model_key(model)
        else:
            model_arg = MessageHandler.build_model(model, "resnet-50", "1.0")
        request = MessageHandler.build_request(
            reply_channel=self._from_worker_ch_serialized,
            model= model_arg,
            inputs=[built_tensor_desc],
            outputs=[],
            output_descriptors=[],
            custom_attributes=None,
        )
        self._perf_timer.measure_time("build_request")
        request_bytes = MessageHandler.serialize_request(request)
        self._perf_timer.measure_time("serialize_request")
        # tensor_bytes = [bytes(tensor.data) for tensor in tensors]
        # tensor_bytes = [tensor.reshape(-1).view(numpy.uint8).data for tensor in tensors]
        # self._perf_timer.measure_time("serialize_tensor")
        to_sendh = self._to_worker_fli.sendh(timeout=None, stream_channel=self._to_worker_ch)
        to_sendh.send_bytes(request_bytes)
        # self._perf_timer.measure_time("send request bytes")
        cp.dump(tensors, file=fli.PickleWriteAdapter(sendh=to_sendh), protocol=pickle.HIGHEST_PROTOCOL)
        # self._perf_timer.measure_time("pickle adapter send tensors")
        # with self._to_worker_fli.sendh(timeout=None, stream_channel=self._to_worker_ch) as to_sendh:
        #     to_sendh.send_bytes(request_bytes)
        #     for tb in tensor_bytes:
        #         to_sendh.send_bytes(tb) #TODO NOT FAST ENOUGH!!!
                # to_sendh.send_bytes(bytes(t.data))

        self._perf_timer.measure_time("send")
        with self._from_worker_ch.recvh(timeout=None) as from_recvh:
            resp = from_recvh.recv_bytes(timeout=None)
            self._perf_timer.measure_time("receive_response")
            response = MessageHandler.deserialize_response(resp)
            self._perf_timer.measure_time("deserialize_response")
            # list of data blobs? recv depending on the len(response.result.descriptors)?
            data_blob: bytes = from_recvh.recv_bytes(timeout=None)
            self._perf_timer.measure_time("receive_tensor")
            result = torch.from_numpy(
                numpy.frombuffer(
                    data_blob,
                    dtype=str(response.result.descriptors[0].dataType),
                )
            )
            self._perf_timer.measure_time("deserialize_tensor")

        self._perf_timer.end_timings()
        return result

    def set_model(self, key: str, model: bytes):
        self._ddict[key] = model

    def print_timings(self, to_file: bool):
        self._perf_timer.print_timings(to_file)


class ResNetWrapper():
    def __init__(self, name: str, model: str):
        self._model = torch.jit.load(model)
        self._name = name
        buffer = io.BytesIO()
        scripted = torch.jit.trace(self._model, self.get_batch())
        torch.jit.save(scripted, buffer)
        self._serialized_model = buffer.getvalue()

    def get_batch(self, batch_size: int=32):
        return torch.randn((batch_size, 3, 224, 224), dtype=torch.float32)

    @property
    def model(self):
        return self._serialized_model

    @property
    def name(self):
        return self._name

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Mock application")
    parser.add_argument("--device", default="cpu")
    parser.add_argument("--log_max_batchsize", default=8, type=int)
    args = parser.parse_args()

    resnet = ResNetWrapper("resnet50", f"resnet50.{args.device.upper()}.pt")

    client = ProtoClient(timing_on=True)
    client.set_model(resnet.name, resnet.model)

    pt_model = torch.jit.load(io.BytesIO(initial_bytes=(resnet.model))).to("cuda:0")

    TOTAL_ITERATIONS = 100

    for log2_bsize in range(args.log_max_batchsize+1):
        b_size: int = 2**log2_bsize
        logger.info(f"Batch size: {b_size}")
        for iteration_number in range(TOTAL_ITERATIONS + int(b_size==1)):
            logger.info(f"Iteration: {iteration_number}")
            batch = resnet.get_batch(b_size)
            remote_result = client.run_model(resnet.name, batch)
            logger.info(client._perf_timer.get_last("total_time"))
            if CHECK_RESULTS_AND_MAKE_ALL_SLOWER:
                local_res = pt_model(batch.to("cuda:0"))
                err_norm = torch.linalg.vector_norm(torch.flatten(remote_result).to("cuda:0")-torch.flatten(local_res), ord=1).cpu()
                res_norm = torch.linalg.vector_norm(remote_result, ord=1).item()
                local_res_norm = torch.linalg.vector_norm(local_res, ord=1).item()
                logger.info(f"Avg norm of error {err_norm.item()/b_size} compared to result norm of {res_norm/b_size}:{local_res_norm/b_size}")
                torch.cuda.synchronize()

    client.print_timings(to_file=True)