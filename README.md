# Torch Optical Flow

PyTorch bindings for NVIDIA Optical Flow SDK, providing hardware-accelerated optical flow computation with PyTorch
end-to-end integration in Nvidia and Python.

Please read more about the NVIDIA Optical Flow SDK here: [https://developer.nvidia.com/optical-flow-sdk](https://developer.nvidia.com/optical-flow-sdk)

# What's this repo about?
- Hardware-accelerated optical flow using a special processor in Nvidia GPUs. No gradients are computed, this is for inference only.
- Frame interpolation and ROI, or other additional content in the SDK is not supported.  
- Configurable speed (slow, medium, fast) vs and grid size (1, 2, 4)
- Support for various ABGR8 format, namely, RGB images.
- End-to-end GPU processing with PyTorch.
- Biderectional optical flow computation (forward and backward) in a single call. This is supported by the SDK, but not exposed in the wrappers they provide.

The package comes with basic functionality for optical flow:
- .flo reader and writer
- Optical Flow common metrics
- Visualization utilities


# Requirements
### System Requirements
- NVIDIA GPU with Optical Flow SDK support (Turing, Ampere or Ada)
- Tested on linux (Ubuntu), the SDK is compatible with windows too. Read optical_flow_sdk>Read_Me.pdf for windows instructions.

### Software Requirements
- CUDA toolkit >=10.2
- Linux drivers "nvidia-smi" >=528.85
- GCC >= 5.1
- CMake >= 3.14
- When you pip install torch, it comes with its own CUDA binaries. Get the same or higher CUDA toolkit version as your PyTorch installation.

# Installation
## Precompiled binaries (experimental)
  ```bash
  uv add torch-nvidia-of-sdk
  # or
  uv add torch-nvidia-of-sdk[full] # To have headless opencv for visualization examples
  ```
  If you still use pip
  ```bash
  pip install torch-nvidia-of-sdk
  # or
  pip install torch-nvidia-of-sdk[full] # To have headless opencv for
  ```

  
## Build from Source (Recommended if precompiled binaries do not work)
This repository uses uv. 
A oneshot comand to build, install and test the package would be:

```bash
rm -rf build _skbuild .venv  && CC=gcc CXX=g++ uv sync --extra full --reinstall-package torch-nvidia-of-sdk && uv run --extra full examples/minimal_example.py
```
`--reinstall-package` forces `uv` to re-compile the package. Clearing caches is not really needed but I'm paranoid.
`--extra full` is analogous to pip extras `pip install torch-nvidia-of-sdk[full]`. It just adds headless opencv for visualization
## Compiling your own wheel
`CC=gcc CXX=g++ uv build --wheel --package torch-nvidia-of-sdk` will build a wheel in `dist/` that you can install with pip.

# Quick Start

Try the minimal example to get started quickly:

```bash
# Run the minimal example (uses sample frames from assets/)
uv run  --extra full examples/minimal_example.py
```

This will:
1. Load two sample frames from the `assets/` directory
2. Compute optical flow using NVOF
3. Generate visualizations and save results to `output/`

See [`examples/README.md`](examples/README.md) for more examples and tutorials.

# Basic Usage

```python
import torch
import numpy as np
from of import TorchNVOpticalFlow
from of.io import read_flo, write_flo
from of.visualization import flow_to_color

# Load your images (RGB format, uint8)
img1 = torch.from_numpy(np.array(...)).cuda()  # Shape: (H, W, 3)
img2 = torch.from_numpy(np.array(...)).cuda()

# Initialize optical flow engine
flow_engine = TorchNVOpticalFlow(
    width=img1.shape[1],
    height=img1.shape[0],
    gpu_id=0,
    preset="medium",  # "slow", "medium", or "fast"
    grid_size=1,      # 1, 2, or 4
)

# Compute optical flow
flow = flow_engine.compute_flow(img1, img2, upsample=True)

# Flow is a (H, W, 2) tensor where flow[..., 0] is x-displacement, flow[..., 1] is y-displacement
print(f"Flow shape: {flow.shape}")

# Visualize flow as RGB image
flow_rgb = flow_to_color(flow.cpu().numpy())

# Save flow to .flo file
write_flo("output_flow.flo", flow)
```

# API Reference

## Core Class: `TorchNVOpticalFlow`

### Constructor

```python
TorchNVOpticalFlow(
    width: int,
    height: int,
    gpu_id: int = 0,
    preset: str = "medium",
    grid_size: int = 1,
    bidirectional: bool = False
)
```

**Parameters:**
- `width`: Width of input images in pixels
- `height`: Height of input images in pixels
- `gpu_id`: CUDA device ID (default: 0)
- `preset`: Speed/quality preset. Options:
  - `"slow"`: Highest quality, slowest
  - `"medium"`: Balanced (recommended)
  - `"fast"`: Fastest, lower quality
- `grid_size`: Output grid size. Options: 1, 2, or 4
  - 1: Full resolution output (default)
  - 2/4: Downsampled output (faster, use with `upsample=True` to restore resolution)
- `bidirectional`: Enable bidirectional flow computation (forward and backward)

### Methods

#### `compute_flow(input, reference, upsample=True)`

Compute forward optical flow between two frames.

**Parameters:**
- `input`: First frame as CUDA tensor of shape `(H, W, 4)`, dtype `uint8`, RGBA format
- `reference`: Second frame as CUDA tensor of shape `(H, W, 4)`, dtype `uint8`, RGBA format
- `upsample`: If True and grid_size > 1, upsample flow to full resolution (default: True)

**Returns:**
- `torch.Tensor`: Optical flow of shape `(H, W, 2)`, dtype `float32`
  - `flow[..., 0]`: Horizontal displacement (x)
  - `flow[..., 1]`: Vertical displacement (y)

**Example:**
```python
flow = flow_engine.compute_flow(img1_rgba, img2_rgba, upsample=True)
```

#### `compute_flow_bidirectional(input, reference, upsample=True)`

Compute both forward and backward optical flow.

**Parameters:**
- `input`: First frame as CUDA tensor of shape `(H, W, 4)`, dtype `uint8`, RGBA format
- `reference`: Second frame as CUDA tensor of shape `(H, W, 4)`, dtype `uint8`, RGBA format
- `upsample`: If True and grid_size > 1, upsample flows to full resolution (default: True)

**Returns:**
- `Tuple[torch.Tensor, torch.Tensor]`: Forward and backward flows, each of shape `(H, W, 2)`

**Example:**
```python
forward_flow, backward_flow = flow_engine.compute_flow_bidirectional(
    img1_rgba, img2_rgba, upsample=True
)
```

#### `output_shape()`

Get the output shape for the current configuration.

**Returns:**
- `List[int]`: Output shape as `[height, width, 2]`

---

## I/O Utilities (`of.io`)

### `read_flo(filepath)`

Read optical flow from `.flo` file (Middlebury format).

**Parameters:**
- `filepath`: Path to `.flo` file (str or Path)

**Returns:**
- `np.ndarray`: Flow array of shape `(H, W, 2)`, dtype `float32`

### `write_flo(filepath, flow)`

Write optical flow to `.flo` file (Middlebury format).

**Parameters:**
- `filepath`: Output file path (str or Path)
- `flow`: Flow array of shape `(H, W, 2)` (numpy array or torch tensor)
  
# Examples

This repository includes several examples in the `examples/` directory:

See [`examples/README.md`](examples/README.md) for detailed documentation and usage instructions.