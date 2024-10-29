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

import signal
import pytest

from smartsim._core.entrypoints.service import Service, SIGNALS

class MockService(Service):
    def _on_iteration(self):
        ...
    
    def _can_shutdown(self):
        ...

@pytest.mark.parametrize(
    "sig",
    [
        pytest.param(signal.SIGINT, id="SIGINT"),
        pytest.param(signal.SIGTERM, id="SIGTERM"),
        pytest.param(signal.SIGQUIT, id="SIGQUIT"),
        pytest.param(signal.SIGABRT, id="SIGABRT"),
    ],
)
def test_handle_signal(sig: signal.Signals):
    """Verify that the handle_signal method sets the trigger_shutdown flag."""
    service = MockService()
    
    assert service.trigger_shutdown == False
    service.handle_signal(sig)
    assert service.trigger_shutdown == True

def test_register_signal_handlers():
    """Verify that the register_signal_handlers method registers the signal handlers."""
    service = MockService()

    for sig in SIGNALS:
        assert signal.getsignal(sig) == service.handle_signal