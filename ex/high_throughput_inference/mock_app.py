# isort: off
import dragon
from dragon import fli
from dragon.channels import Channel
import dragon.channels
from dragon.data.ddict.ddict import DDict
from dragon.globalservices.api_setup import connect_to_infrastructure
from dragon.utils import b64decode

# isort: on

import argparse
import io
import numpy
import os
import time
import torch
import numbers

from collections import OrderedDict
from smartsim._core.mli.message_handler import MessageHandler

class ProtoClient:
    def __init__(self, timing_on: bool):
        connect_to_infrastructure()
        ddict_str = os.environ["SS_DRG_DDICT"]
        self._ddict = DDict.attach(ddict_str)
        to_worker_fli_str = None
        while to_worker_fli_str is None:
            try:
                to_worker_fli_str = self._ddict["to_worker_fli"]
                self._to_worker_fli = fli.FLInterface.attach(b64decode(to_worker_fli_str))
            except KeyError:
                time.sleep(1)
        self._from_worker_ch = Channel.make_process_local()
        self._from_worker_ch_serialized = self._from_worker_ch.serialize()
        self._to_worker_ch = Channel.make_process_local()

        self._start = None
        self._interm = None
        self._timings: OrderedDict[str, list[numbers.Number]] = OrderedDict()
        self._timing_on = timing_on

    def _add_label_to_timings(self, label: str):
        if label not in self._timings:
            self._timings[label] = []

    @staticmethod
    def _format_number(number: numbers.Number):
        return f"{number:0.4e}"

    def start_timings(self, batch_size: int):
        if self._timing_on:
            self._add_label_to_timings("batch_size")
            self._timings["batch_size"].append(batch_size)
            self._start = time.perf_counter()
            self._interm = time.perf_counter()

    def end_timings(self):
        if self._timing_on:
            self._add_label_to_timings("total_time")
            self._timings["total_time"].append(self._format_number(time.perf_counter()-self._start))

    def measure_time(self, label: str):
        if self._timing_on:
            self._add_label_to_timings(label)
            self._timings[label].append(self._format_number(time.perf_counter()-self._interm))
            self._interm = time.perf_counter()

    def print_timings(self, to_file: bool = False):
        print(" ".join(self._timings.keys()))
        value_array = numpy.array([value for  value in self._timings.values()], dtype=float)
        value_array = numpy.transpose(value_array)
        for i in range(value_array.shape[0]):
            print(" ".join(self._format_number(value) for value in value_array[i]))
        if to_file:
            numpy.save("timings.npy", value_array)
            numpy.savetxt("timings.txt", value_array)


    def run_model(self, model: bytes, batch: torch.Tensor):
        self.start_timings(batch.shape[0])
        built_tensor = MessageHandler.build_tensor(
            batch.numpy(), "c", "float32", list(batch.shape))
        self.measure_time("build_tensor")
        request = MessageHandler.build_request(
            reply_channel=self._from_worker_ch_serialized,
            model=model,
            inputs=[built_tensor],
            outputs=[],
            output_descriptors=[],
            custom_attributes=None,
        )
        self.measure_time("build_request")
        request_bytes = MessageHandler.serialize_request(request)
        self.measure_time("serialize_request")
        with self._to_worker_fli.sendh(timeout=None, stream_channel=self._to_worker_ch) as to_sendh:
            to_sendh.send_bytes(request_bytes)

        self.measure_time("send")
        with self._from_worker_ch.recvh(timeout=None) as from_recvh:
            resp = from_recvh.recv_bytes(timeout=None)
            self.measure_time("receive")
            response = MessageHandler.deserialize_response(resp)
            self.measure_time("deserialize_response")
            result = torch.from_numpy(
                numpy.frombuffer(
                    response.result.data[0].blob,
                    dtype=str(response.result.data[0].tensorDescriptor.dataType),
                )
            )
            self.measure_time("deserialize_tensor")

        self.end_timings()
        return result


class ResNetWrapper():
    def __init__(self, model: str):
        self._model = torch.jit.load(model)
        buffer = io.BytesIO()
        scripted = torch.jit.trace(self._model, self.get_batch())
        torch.jit.save(scripted, buffer)
        self._serialized_model = buffer.getvalue()

    def get_batch(self, batch_size: int=32):
        return torch.randn((batch_size, 3, 224, 224), dtype=torch.float32)

    @property
    def model(self):
        return self._serialized_model

if __name__ == "__main__":

    parser = argparse.ArgumentParser("Mock application")
    parser.add_argument("--device", default="cpu")
    args = parser.parse_args()

    resnet = ResNetWrapper(f"resnet50.{args.device.upper()}.pt")

    client = ProtoClient(timing_on=True)

    total_iterations = 10

    for batch_size in [1, 8, 32, 64, 128]:
        for iteration_number in range(total_iterations + int(batch_size==1)):
            client.run_model(resnet.model, resnet.get_batch(batch_size))

    client.print_timings(to_file=True)