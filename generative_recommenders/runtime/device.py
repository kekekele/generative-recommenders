# pyre-strict

from typing import Literal

import torch


AcceleratorType = Literal["auto", "cuda", "npu", "cpu"]


def _npu_available() -> bool:
    has_npu = hasattr(torch, "npu")
    return bool(has_npu and torch.npu.is_available())


def detect_accelerator(preferred: AcceleratorType = "auto") -> str:
    if preferred != "auto":
        if preferred == "cuda":
            if not torch.cuda.is_available():
                raise RuntimeError("Requested CUDA accelerator but CUDA is unavailable")
            return "cuda"
        if preferred == "npu":
            if not _npu_available():
                raise RuntimeError("Requested NPU accelerator but NPU is unavailable")
            return "npu"
        return "cpu"

    if torch.cuda.is_available():
        return "cuda"
    if _npu_available():
        return "npu"
    return "cpu"


def get_device_count(accelerator: str) -> int:
    if accelerator == "cuda":
        return torch.cuda.device_count()
    if accelerator == "npu":
        return torch.npu.device_count()
    return 1


def get_device_for_rank(rank: int, accelerator: str) -> torch.device:
    if accelerator in ("cuda", "npu"):
        return torch.device(f"{accelerator}:{rank}")
    return torch.device("cpu")


def set_current_device(device: torch.device) -> None:
    if device.type == "cuda":
        torch.cuda.set_device(device)
    elif device.type == "npu":
        torch.npu.set_device(device)


def dist_backend_for_accelerator(accelerator: str) -> str:
    if accelerator == "cuda":
        return "nccl"
    if accelerator == "npu":
        return "hccl"
    return "gloo"


def autocast_device_type(device: torch.device) -> str:
    # torch.autocast currently expects a valid device type string.
    if device.type in ("cuda", "cpu", "xpu", "npu"):
        return device.type
    return "cpu"


def can_use_bf16(device: torch.device) -> bool:
    if device.type == "cuda":
        return bool(hasattr(torch.cuda, "is_bf16_supported") and torch.cuda.is_bf16_supported())
    if device.type == "npu":
        return True
    return False
