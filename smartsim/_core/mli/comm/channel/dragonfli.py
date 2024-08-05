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
from dragon import fli
import dragon.channels as dch

# isort: on

import sys
import typing as t

import cloudpickle as cp

import smartsim._core.mli.comm.channel.channel as cch
from smartsim.log import get_logger

logger = get_logger(__name__)


class DragonFLIChannel(cch.CommChannelBase):
    """Passes messages by writing to a Dragon FLI Channel"""

    def __init__(self, fli_desc: bytes, sender_supplied: bool = True) -> None:
        """Initialize the DragonFLIChannel instance"""
        super().__init__(fli_desc)
        # todo: do we need memory pool information to construct the channel correctly?
        self._fli: "fli" = fli.FLInterface.attach(fli_desc)
        self._channel: t.Optional["dch"] = (
            dch.Channel.make_process_local() if sender_supplied else None
        )

    def send(self, value: bytes) -> None:
        """Send a message through the underlying communication channel
        :param value: The value to send"""
        with self._fli.sendh(timeout=None, stream_channel=self._channel) as sendh:
            sendh.send_bytes(value)

    def recv(self) -> t.Tuple[t.Any, t.Any]:  # fix this
        """Receive a message through the underlying communication channel
        :returns: the received message"""
        recvh = self._fli.recvh(timeout=None)
        request_bytes = None
        received_tensors = []
        try:
            request_bytes = recvh.recv_bytes(timeout=None)
            received_tensors = cp.load(file=fli.PickleReadAdapter(recvh=recvh))
            return request_bytes, received_tensors
        except Exception:
            raise ValueError("No request data found")
