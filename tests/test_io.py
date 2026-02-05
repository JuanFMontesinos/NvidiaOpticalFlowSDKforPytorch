"""Tests for input/output utilities."""

import pytest
import numpy as np
import torch

from of.io import read_flo, write_flo


class TestFloIO:
    """Test reading and writing .flo files."""

    def test_read_flo_valid(self, gt_flow_path):
        """Test reading a valid .flo file."""
        if not gt_flow_path.exists():
            pytest.skip("Ground truth flow file not available")

        flow = read_flo(gt_flow_path)

        # Check shape and dtype
        assert flow.ndim == 3, "Flow should be 3D array"
        assert flow.shape[2] == 2, "Flow should have 2 channels (u, v)"
        assert flow.dtype == np.float32, "Flow should be float32"
        assert flow.shape[0] > 0 and flow.shape[1] > 0, "Flow should have non-zero dimensions"

    def test_read_flo_invalid_magic(self, temp_output_dir):
        """Test reading a file with invalid magic number."""
        bad_file = temp_output_dir / "bad.flo"

        # Write file with wrong magic number
        with open(bad_file, "wb") as f:
            import struct
            f.write(struct.pack("f", 123.456))  # Wrong magic
            f.write(struct.pack("i", 10))  # Width
            f.write(struct.pack("i", 10))  # Height

        with pytest.raises(ValueError, match="Invalid .flo file format"):
            read_flo(bad_file)

    def test_write_flo_numpy(self, temp_output_dir):
        """Test writing .flo file from numpy array."""
        # Create a simple flow field
        height, width = 100, 150
        flow = np.random.randn(height, width, 2).astype(np.float32)

        output_path = temp_output_dir / "test.flo"
        write_flo(output_path, flow)

        # Verify the file was created
        assert output_path.exists(), "Flow file should be created"

        # Read it back and verify
        flow_read = read_flo(output_path)
        np.testing.assert_array_almost_equal(flow, flow_read, decimal=5)

    def test_write_flo_torch_tensor(self, temp_output_dir):
        """Test writing .flo file from torch tensor."""
        height, width = 80, 120
        flow = torch.randn(height, width, 2)

        output_path = temp_output_dir / "test_torch.flo"
        write_flo(output_path, flow)

        assert output_path.exists()

        # Read it back
        flow_read = read_flo(output_path)
        np.testing.assert_array_almost_equal(
            flow.cpu().numpy(), flow_read, decimal=5
        )

    def test_write_flo_invalid_shape(self, temp_output_dir):
        """Test that writing flow with invalid shape raises error."""
        # Wrong shape (missing channel dimension)
        flow = np.random.randn(100, 150)

        output_path = temp_output_dir / "invalid.flo"
        with pytest.raises(ValueError, match="Flow must have shape"):
            write_flo(output_path, flow)

    def test_read_write_roundtrip(self, gt_flow_path, temp_output_dir):
        """Test that reading and writing preserves data."""
        if not gt_flow_path.exists():
            pytest.skip("Ground truth flow file not available")

        # Read original
        flow_original = read_flo(gt_flow_path)

        # Write and read back
        output_path = temp_output_dir / "roundtrip.flo"
        write_flo(output_path, flow_original)
        flow_roundtrip = read_flo(output_path)

        # Should be identical
        np.testing.assert_array_equal(flow_original, flow_roundtrip)

    def test_read_flo_accepts_path_object(self, gt_flow_path):
        """Test that read_flo accepts Path objects."""
        if not gt_flow_path.exists():
            pytest.skip("Ground truth flow file not available")

        flow = read_flo(gt_flow_path)
        assert isinstance(flow, np.ndarray)

    def test_read_flo_accepts_string(self, gt_flow_path):
        """Test that read_flo accepts string paths."""
        if not gt_flow_path.exists():
            pytest.skip("Ground truth flow file not available")

        flow = read_flo(str(gt_flow_path))
        assert isinstance(flow, np.ndarray)
