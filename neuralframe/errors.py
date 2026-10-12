"""Error taxonomy shared by every NeuralFrame component.

Every failure that can reach a user (CLI, API, GUI, job report) is a :class:`NeuralFrameError` carrying a
stable :class:`ErrorCode`, so front ends can react programmatically and logs stay greppable.
"""

from __future__ import annotations

from collections.abc import Mapping
from enum import StrEnum
from typing import Any


class ErrorCode(StrEnum):
    INVALID_INPUT = "INVALID_INPUT"
    PROBE_FAILED = "PROBE_FAILED"
    DECODE_FAILED = "DECODE_FAILED"
    ENGINE_BUILD_FAILED = "ENGINE_BUILD_FAILED"
    OOM_UNRECOVERABLE = "OOM_UNRECOVERABLE"
    ENCODE_FAILED = "ENCODE_FAILED"
    MUX_FAILED = "MUX_FAILED"
    SYNC_CHECK_FAILED = "SYNC_CHECK_FAILED"
    MODEL_UNAVAILABLE = "MODEL_UNAVAILABLE"
    DOWNLOAD_FAILED = "DOWNLOAD_FAILED"
    BACKEND_UNAVAILABLE = "BACKEND_UNAVAILABLE"
    INFERENCE_FAILED = "INFERENCE_FAILED"
    RESUME_MISMATCH = "RESUME_MISMATCH"
    CONFIG_INVALID = "CONFIG_INVALID"
    TOOL_MISSING = "TOOL_MISSING"
    CANCELLED = "CANCELLED"
    INTERNAL = "INTERNAL"


class NeuralFrameError(Exception):
    """Base exception with a machine-readable code, optional details and a user-facing hint."""

    def __init__(
        self,
        code: ErrorCode,
        message: str,
        *,
        details: Mapping[str, Any] | None = None,
        hint: str | None = None,
    ) -> None:
        super().__init__(message)
        self.code = code
        self.message = message
        self.details: dict[str, Any] = dict(details or {})
        self.hint = hint

    def __str__(self) -> str:
        text = f"[{self.code}] {self.message}"
        if self.hint:
            text += f" (hint: {self.hint})"
        return text

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {"code": str(self.code), "message": self.message}
        if self.details:
            data["details"] = self.details
        if self.hint:
            data["hint"] = self.hint
        return data


class JobCancelled(NeuralFrameError):
    """Raised inside pipeline stages when the job's cancel token fires."""

    def __init__(self, reason: str = "cancelled by user") -> None:
        super().__init__(ErrorCode.CANCELLED, reason)
