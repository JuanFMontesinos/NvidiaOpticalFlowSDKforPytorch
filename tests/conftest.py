"""Pytest configuration and shared fixtures."""

import pytest
from pathlib import Path
import numpy as np
import torch
import imageio.v3 as imageio


@pytest.fixture(scope="session")
def assets_dir():
    """Return the path to the assets directory."""
    return Path(__file__).parent.parent / "assets"


@pytest.fixture(scope="session")
def frame1_path(assets_dir):
    """Return the path to the first test frame."""
    return assets_dir / "frame_0001.png"


@pytest.fixture(scope="session")
def frame2_path(assets_dir):
    """Return the path to the second test frame."""
    return assets_dir / "frame_0002.png"


@pytest.fixture(scope="session")
def gt_flow_path(assets_dir):
    """Return the path to the ground truth flow file."""
    return assets_dir / "frame_0001.flo"


@pytest.fixture(scope="session")
def frame1_image(frame1_path):
    """Load the first test frame as a numpy array."""
    return imageio.imread(frame1_path)


@pytest.fixture(scope="session")
def frame2_image(frame2_path):
    """Load the second test frame as a numpy array."""
    return imageio.imread(frame2_path)


@pytest.fixture(scope="session")
def gt_flow(gt_flow_path):
    """Load the ground truth flow if available."""
    if gt_flow_path.exists():
        from of.io import read_flo
        return read_flo(gt_flow_path)
    return None


@pytest.fixture(scope="session")
def frame1_tensor(frame1_image):
    """Convert the first frame to a torch tensor."""
    if frame1_image.dtype == np.uint8:
        return torch.from_numpy(frame1_image)
    else:
        return torch.from_numpy((frame1_image * 255).astype(np.uint8))


@pytest.fixture(scope="session")
def frame2_tensor(frame2_image):
    """Convert the second frame to a torch tensor."""
    if frame2_image.dtype == np.uint8:
        return torch.from_numpy(frame2_image)
    else:
        return torch.from_numpy((frame2_image * 255).astype(np.uint8))


@pytest.fixture(scope="function")
def temp_output_dir(tmp_path):
    """Create a temporary directory for test outputs."""
    output_dir = tmp_path / "output"
    output_dir.mkdir(exist_ok=True)
    return output_dir


@pytest.fixture
def cuda_available():
    """Check if CUDA is available."""
    return torch.cuda.is_available()


@pytest.fixture
def skip_if_no_cuda(cuda_available):
    """Skip test if CUDA is not available."""
    if not cuda_available:
        pytest.skip("CUDA is not available")
