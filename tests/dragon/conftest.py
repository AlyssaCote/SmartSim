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

from __future__ import annotations

import os
import pathlib
import socket
import subprocess
import sys
import typing as t

import pytest

dragon = pytest.importorskip("dragon")

# isort: off
import dragon.data.ddict.ddict as dragon_ddict
import dragon.infrastructure.policy as dragon_policy
import dragon.infrastructure.process_desc as dragon_process_desc
import dragon.native.process as dragon_process

from dragon.fli import FLInterface

# isort: on

from smartsim._core.mli.comm.channel.dragon_fli import DragonFLIChannel
from smartsim._core.mli.comm.channel.dragon_util import create_local
from smartsim._core.mli.infrastructure.storage import dragon_util
from smartsim._core.mli.infrastructure.storage.backbone_feature_store import (
    BackboneFeatureStore,
)
from smartsim.log import get_logger

logger = get_logger(__name__)


@pytest.fixture(scope="module")
def the_storage() -> dragon_ddict.DDict:
    """Fixture to instantiate a dragon distributed dictionary."""
    return dragon_util.create_ddict(1, 2, 32 * 1024**2)


@pytest.fixture(scope="module")
def the_worker_channel() -> DragonFLIChannel:
    """Fixture to create a valid descriptor for a worker channel
    that can be attached to."""
    channel_ = create_local()
    fli_ = FLInterface(main_ch=channel_, manager_ch=None)
    comm_channel = DragonFLIChannel(fli_)
    return comm_channel


@pytest.fixture(scope="module")
def the_backbone(
    the_storage: t.Any, the_worker_channel: DragonFLIChannel
) -> BackboneFeatureStore:
    """Fixture to create a distributed dragon dictionary and wrap it
    in a BackboneFeatureStore.

    :param the_storage: The dragon storage engine to use
    :param the_worker_channel: Pre-configured worker channel
    """

    backbone = BackboneFeatureStore(the_storage, allow_reserved_writes=True)
    backbone[BackboneFeatureStore.MLI_WORKER_QUEUE] = the_worker_channel.descriptor

    return backbone


@pytest.fixture(scope="module")
def backbone_descriptor(the_backbone: BackboneFeatureStore) -> str:
    # create a shared backbone featurestore
    return the_backbone.descriptor


def function_as_dragon_proc(
    entrypoint_fn: t.Callable[[t.Any], None],
    args: t.List[t.Any],
    cpu_affinity: t.List[int],
    gpu_affinity: t.List[int],
) -> dragon_process.Process:
    """Execute a function as an independent dragon process.

    :param entrypoint_fn: The function to execute
    :param args: The arguments for the entrypoint function
    :param cpu_affinity: The cpu affinity for the process
    :param gpu_affinity: The gpu affinity for the process
    :returns: The dragon process handle
    """
    options = dragon_process_desc.ProcessOptions(make_inf_channels=True)
    local_policy = dragon_policy.Policy(
        placement=dragon_policy.Policy.Placement.HOST_NAME,
        host_name=socket.gethostname(),
        cpu_affinity=cpu_affinity,
        gpu_affinity=gpu_affinity,
    )
    return dragon_process.Process(
        target=entrypoint_fn,
        args=args,
        cwd=os.getcwd(),
        policy=local_policy,
        options=options,
        stderr=dragon_process.Popen.STDOUT,
        stdout=dragon_process.Popen.STDOUT,
    )