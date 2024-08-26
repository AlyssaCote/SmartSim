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
from abc import ABC, abstractmethod

from pydantic import BaseModel, Field

# from memory_profiler import profile

from smartsim.log import get_logger

logger = get_logger(__name__)


class FeatureStoreKey(BaseModel):
    """A key,descriptor pair enabling retrieval of an item from a feature store"""

    key: str = Field(min_length=1)
    """The unique key of an item in a feature store"""
    descriptor: str = Field(min_length=1)
    """The unique identifier of the feature store containing the key"""


class FeatureStore(ABC):
    """Abstract base class providing the common interface for retrieving
    values from a feature store implementation"""

    @abstractmethod
    # @profile
    def __getitem__(self, key: str) -> t.Union[str, bytes]:
        """Retrieve an item using key

        :param key: Unique key of an item to retrieve from the feature store"""

    @abstractmethod
    # @profile
    def __setitem__(self, key: str, value: t.Union[str, bytes]) -> None:
        """Assign a value using key

        :param key: Unique key of an item to set in the feature store
        :param value: Value to persist in the feature store"""

    @abstractmethod
    # @profile
    def __contains__(self, key: str) -> bool:
        """Membership operator to test for a key existing within the feature store.

        :param key: Unique key of an item to retrieve from the feature store
        :returns: `True` if the key is found, `False` otherwise"""

    @property
    @abstractmethod
    # @profile
    def descriptor(self) -> str:
        """Unique identifier enabling a client to connect to the feature store

        :returns: A descriptor encoded as a string"""
