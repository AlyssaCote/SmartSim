"""This is an automatically generated stub for `request_attributes.capnp`."""
# mypy: ignore-errors

from __future__ import annotations

from contextlib import contextmanager
from io import BufferedWriter
from typing import Iterator, Literal

TorchTensorType = Literal["nested", "sparse", "tensor"]
TFTensorType = Literal["ragged", "sparse", "variable", "constant"]

class TorchRequestAttributes:
    tensorType: TorchTensorType
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[TorchRequestAttributesReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> TorchRequestAttributesReader: ...
    @staticmethod
    def new_message() -> TorchRequestAttributesBuilder: ...
    def to_dict(self) -> dict: ...

class TorchRequestAttributesReader(TorchRequestAttributes):
    def as_builder(self) -> TorchRequestAttributesBuilder: ...

class TorchRequestAttributesBuilder(TorchRequestAttributes):
    @staticmethod
    def from_dict(dictionary: dict) -> TorchRequestAttributesBuilder: ...
    def copy(self) -> TorchRequestAttributesBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> TorchRequestAttributesReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...

class TensorFlowRequestAttributes:
    name: str
    tensorType: TFTensorType
    @staticmethod
    @contextmanager
    def from_bytes(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> Iterator[TensorFlowRequestAttributesReader]: ...
    @staticmethod
    def from_bytes_packed(
        data: bytes,
        traversal_limit_in_words: int | None = ...,
        nesting_limit: int | None = ...,
    ) -> TensorFlowRequestAttributesReader: ...
    @staticmethod
    def new_message() -> TensorFlowRequestAttributesBuilder: ...
    def to_dict(self) -> dict: ...

class TensorFlowRequestAttributesReader(TensorFlowRequestAttributes):
    def as_builder(self) -> TensorFlowRequestAttributesBuilder: ...

class TensorFlowRequestAttributesBuilder(TensorFlowRequestAttributes):
    @staticmethod
    def from_dict(dictionary: dict) -> TensorFlowRequestAttributesBuilder: ...
    def copy(self) -> TensorFlowRequestAttributesBuilder: ...
    def to_bytes(self) -> bytes: ...
    def to_bytes_packed(self) -> bytes: ...
    def to_segments(self) -> list[bytes]: ...
    def as_reader(self) -> TensorFlowRequestAttributesReader: ...
    @staticmethod
    def write(file: BufferedWriter) -> None: ...
    @staticmethod
    def write_packed(file: BufferedWriter) -> None: ...
