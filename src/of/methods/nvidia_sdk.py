import torch
import numpy as np
from typing import Literal
from .nvof_torch import TorchNVOpticalFlow as _C_TorchNVOpticalFlow


class TorchNVOpticalFlow(_C_TorchNVOpticalFlow):
    @classmethod
    def from_tensor(
        cls,
        input: torch.Tensor,
        preset: Literal["slow", "medium", "fast"] = "fast",
        grid_size: Literal[1, 2, 4] = 1,
    ):
        """
        Factory method to create engine from a tensor.

        Args:
            input: Tensor of shape (H, W, C) to infer dimensions from
            preset: Performance preset ("slow", "medium", "fast")
            grid_size: Grid size for optical flow (1, 2, or 4)
            bidirectional: Whether to compute bidirectional flow # TODO

        Returns:
            TorchNVOpticalFlow instance
        """
        h = input.size(0)
        w = input.size(1)
        gpu = input.get_device()
        return cls(w, h, gpu, preset, grid_size, False)

    @torch.no_grad()
    def compute_flow(
        self,
        input: torch.Tensor,
        reference: torch.Tensor,
        upsample: bool = True,
    ) -> torch.Tensor:
        """
        Compute optical flow between input and reference frames.

        Args:
            input: Input tensor of shape (H, W, 3) or (H, W, 4), uint8, CUDA
            reference: Reference tensor of shape (H, W, 3) or (H, W, 4), uint8, CUDA
            upsample: Whether to upsample the output to full resolution

        Returns:
            Flow tensor of shape (H, W, 2), int16
        """
        # NVOpticalFlow requires ABGR uint8
        if isinstance(input, np.ndarray):
            input = torch.from_numpy(input)
        if isinstance(reference, np.ndarray):
            reference = torch.from_numpy(reference)

        if input.shape[-1] == 3:
            alpha_input = input.sum(dim=-1, keepdim=True).div(3).clamp(0, 255).byte()
            input = torch.cat([alpha_input, input], dim=-1).to(f"cuda:{self.gpu_id()}")

        if reference.shape[-1] == 3:
            alpha_reference = reference.sum(dim=-1, keepdim=True).div(3).clamp(0, 255).byte()
            reference = torch.cat([alpha_reference, reference], dim=-1).to(f"cuda:{self.gpu_id()}")

        # Checks performed in C++
        return super().compute_flow(input, reference, upsample=upsample).float() / 32.0
