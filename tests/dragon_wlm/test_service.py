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

import typing as t

import dragon.channels as dch
import dragon.data.ddict.ddict as dd
import dragon.fli as fli
import dragon.globalservices.pool as dragon_gs_pool
import dragon.managed_memory as dm
import pytest

from smartsim._core.entrypoints.service import DragonShutdownResource, Service
from smartsim._core.mli.comm.channel.dragon_util import create_local
from smartsim._core.mli.infrastructure.storage.dragon_util import create_ddict

# channel
drg_channel = create_local()
drg_channel_descriptor = drg_channel.serialize()

# fli
drg_fli_channel = create_local()
drg_fli = fli.FLInterface(main_ch=drg_fli_channel, manager_ch=None)
drg_fli_descriptor = drg_fli.serialize()

# pool
drg_pool = dm.MemoryPool.attach(dragon_gs_pool.create(2 * 1024**3).sdesc)
drg_pool_descriptor = drg_pool.serialize()

# ddict
drg_ddict = create_ddict(2, 1, 512 * 1024**2)
drg_ddict_descriptor = drg_ddict.serialize()


class MockService(Service):
    def _can_shutdown(self) -> bool:
        return self.trigger_shutdown

    def _on_iteration(self):
        return super()._on_iteration()


@pytest.mark.parametrize(
    "resource_descriptor, attacher",
    [
        (drg_channel_descriptor, dch.Channel.attach),
        (drg_fli_descriptor, fli.FLInterface.attach),
    ],
)
def test_track_resources_duplicates(
    resource_descriptor: bytes, attacher: t.Callable[[str], t.Any]
) -> None:
    """Verify that duplicates cannot be added to the set of tracked resources."""
    service = MockService()
    assert service.dragon_resources == set()

    # this should be added in
    service.track_resource(DragonShutdownResource(resource_descriptor, attacher))
    assert len(service.dragon_resources) == 1

    # this duplicate should not be added in
    service.track_resource(DragonShutdownResource(resource_descriptor, attacher))
    assert len(service.dragon_resources) == 1


def test_track_resources_no_duplicates() -> None:
    """Verify that unique resources can be added to the set of tracked resources."""
    service = MockService()
    assert service.dragon_resources == set()

    service.track_resource(
        DragonShutdownResource(drg_channel_descriptor, dch.Channel.attach)
    )
    assert len(service.dragon_resources) == 1

    service.track_resource(
        DragonShutdownResource(drg_fli_descriptor, fli.FLInterface.attach)
    )
    assert len(service.dragon_resources) == 2


@pytest.mark.parametrize(
    "resources",
    [
        [
            DragonShutdownResource(drg_channel_descriptor, dch.Channel.attach),
            DragonShutdownResource(drg_fli_descriptor, fli.FLInterface.attach),
            DragonShutdownResource(drg_pool_descriptor, dm.MemoryPool.attach),
            DragonShutdownResource(drg_ddict_descriptor, dd.DDict.attach),
        ],
    ],
)
def test_on_shutdown(resources: t.List[DragonShutdownResource]) -> None:
    """Verify that resources are properly shutdown."""
    service = MockService()
    for resource in resources:
        service.track_resource(resource)
    assert len(service.dragon_resources) == len(resources)
    service._on_shutdown()
    assert len(service.dragon_resources) == 0
