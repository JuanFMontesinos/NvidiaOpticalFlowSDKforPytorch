"""
Input/Output utilities for optical flow data.
Supports reading and writing optical flow in various formats.
"""

import struct
from pathlib import Path
from typing import Union

import numpy as np
import torch
from numpy.typing import NDArray
from torch import Tensor

FlowType = Union[NDArray, Tensor]


def read_flo(filepath: Union[str, Path]) -> NDArray:  # noqa
    """
    Read optical flow from .flo file (Middlebury format).

    Args:
        filepath: Path to the .flo file

    Returns:
        Optical flow as numpy array of shape (H, W, 2) with dtype float32

    Raises:
        ValueError: If file format is invalid
    """
    filepath = Path(filepath)

    with open(filepath, "rb") as f:
        # Check magic number
        magic = struct.unpack("f", f.read(4))[0]
        if magic != 202021.25:
            raise ValueError(f"Invalid .flo file format. Magic number: {magic}")

        # Read dimensions
        width = struct.unpack("i", f.read(4))[0]
        height = struct.unpack("i", f.read(4))[0]

        # Read flow data
        data = np.fromfile(f, np.float32)

    # Reshape to (height, width, 2)
    flow = data.reshape((height, width, 2))
    return flow


def write_flo(filepath: Union[str, Path], flow: FlowType) -> None:
    """
    Write optical flow to .flo file (Middlebury format).

    Args:
        filepath: Path to save the .flo file
        flow: Optical flow as numpy array of shape (H, W, 2)

    Raises:
        ValueError: If flow array shape is invalid
    """
    filepath = Path(filepath)
    if isinstance(flow, torch.Tensor):
        flow = flow.detach().cpu().numpy()

    if flow.ndim != 3 or flow.shape[2] != 2:
        raise ValueError(f"Flow must have shape (H, W, 2), got {flow.shape}")

    height, width, _ = flow.shape

    with open(filepath, "wb") as f:
        # Write magic number
        f.write(struct.pack("f", 202021.25))

        # Write dimensions
        f.write(struct.pack("i", width))
        f.write(struct.pack("i", height))

        # Write flow data
        flow.astype(np.float32).tofile(f)
