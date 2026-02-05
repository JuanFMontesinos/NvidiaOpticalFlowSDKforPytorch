"""Tests for NVIDIA Optical Flow SDK integration."""

import pytest
import torch

# Only try to import if CUDA is available
try:
    from of import TorchNVOpticalFlow
    NVOF_AVAILABLE = torch.cuda.is_available()
except (ImportError, RuntimeError):
    NVOF_AVAILABLE = False


@pytest.mark.skipif(not NVOF_AVAILABLE, reason="CUDA or NVOF not available")
class TestTorchNVOpticalFlow:
    """Test NVIDIA Optical Flow SDK functionality."""

    def test_from_tensor_factory(self, frame1_tensor, skip_if_no_cuda):
        """Test creating TorchNVOpticalFlow from tensor."""
        # Move to CUDA
        frame1_cuda = frame1_tensor.cuda()

        engine = TorchNVOpticalFlow.from_tensor(
            frame1_cuda,
            preset="fast",
            grid_size=1
        )

        assert engine is not None
        assert engine.gpu_id() >= 0

    def test_compute_flow_basic(self, frame1_tensor, frame2_tensor, skip_if_no_cuda):
        """Test basic optical flow computation."""
        # Move to CUDA and ensure uint8
        frame1_cuda = frame1_tensor.cuda().byte()
        frame2_cuda = frame2_tensor.cuda().byte()

        # Create engine
        engine = TorchNVOpticalFlow.from_tensor(
            frame1_cuda,
            preset="fast",
            grid_size=1
        )

        # Compute flow
        flow = engine.compute_flow(frame1_cuda, frame2_cuda, upsample=True)

        # Check output
        assert isinstance(flow, torch.Tensor)
        assert flow.shape[2] == 2, "Flow should have 2 channels (u, v)"
        assert flow.shape[0] == frame1_cuda.shape[0], "Height should match"
        assert flow.shape[1] == frame1_cuda.shape[1], "Width should match"
        assert flow.dtype == torch.float32, "Flow should be float32"

    def test_compute_flow_different_presets(self, frame1_tensor, frame2_tensor, skip_if_no_cuda):
        """Test flow computation with different presets."""
        frame1_cuda = frame1_tensor.cuda().byte()
        frame2_cuda = frame2_tensor.cuda().byte()

        presets = ["fast", "medium", "slow"]
        flows = []

        for preset in presets:
            engine = TorchNVOpticalFlow.from_tensor(
                frame1_cuda,
                preset=preset,
                grid_size=1
            )
            flow = engine.compute_flow(frame1_cuda, frame2_cuda, upsample=True)
            flows.append(flow)

        # All flows should be valid
        for flow in flows:
            assert flow.shape[2] == 2
            assert not torch.isnan(flow).any(), "Flow should not contain NaN"

    def test_compute_flow_different_grid_sizes(self, frame1_tensor, frame2_tensor, skip_if_no_cuda):
        """Test flow computation with different grid sizes."""
        frame1_cuda = frame1_tensor.cuda().byte()
        frame2_cuda = frame2_tensor.cuda().byte()

        grid_sizes = [1, 2, 4]

        for grid_size in grid_sizes:
            engine = TorchNVOpticalFlow.from_tensor(
                frame1_cuda,
                preset="fast",
                grid_size=grid_size
            )
            flow = engine.compute_flow(frame1_cuda, frame2_cuda, upsample=True)

            assert flow.shape[2] == 2
            assert not torch.isnan(flow).any()

    def test_compute_flow_no_upsample(self, frame1_tensor, frame2_tensor, skip_if_no_cuda):
        """Test flow computation without upsampling."""
        frame1_cuda = frame1_tensor.cuda().byte()
        frame2_cuda = frame2_tensor.cuda().byte()

        engine = TorchNVOpticalFlow.from_tensor(
            frame1_cuda,
            preset="fast",
            grid_size=4
        )

        flow = engine.compute_flow(frame1_cuda, frame2_cuda, upsample=False)

        # Without upsampling, flow should be smaller
        # Grid size 4 means flow is 1/4 resolution
        assert flow.shape[0] <= frame1_cuda.shape[0]
        assert flow.shape[1] <= frame1_cuda.shape[1]

    def test_compute_flow_rgb_input(self, frame1_tensor, frame2_tensor, skip_if_no_cuda):
        """Test flow computation with RGB input (will be converted to ABGR)."""
        # Ensure RGB format (H, W, 3)
        if frame1_tensor.shape[2] == 4:
            frame1_rgb = frame1_tensor[:, :, :3]
            frame2_rgb = frame2_tensor[:, :, :3]
        else:
            frame1_rgb = frame1_tensor
            frame2_rgb = frame2_tensor

        frame1_cuda = frame1_rgb.cuda().byte()
        frame2_cuda = frame2_rgb.cuda().byte()

        engine = TorchNVOpticalFlow.from_tensor(
            frame1_cuda,
            preset="fast",
            grid_size=1
        )

        # Should automatically add alpha channel
        flow = engine.compute_flow(frame1_cuda, frame2_cuda, upsample=True)

        assert flow.shape[2] == 2

    def test_compute_flow_numpy_input(self, frame1_image, frame2_image, skip_if_no_cuda):
        """Test flow computation with numpy input."""
        # Create a small test case
        h, w = min(frame1_image.shape[0], 256), min(frame1_image.shape[1], 256)
        frame1_small = frame1_image[:h, :w]
        frame2_small = frame2_image[:h, :w]

        # Ensure RGB
        if frame1_small.shape[2] == 4:
            frame1_small = frame1_small[:, :, :3]
            frame2_small = frame2_small[:, :, :3]

        # Convert to tensor for engine creation
        frame1_tensor = torch.from_numpy(frame1_small).cuda()

        engine = TorchNVOpticalFlow.from_tensor(
            frame1_tensor,
            preset="fast",
            grid_size=1
        )

        # Pass numpy arrays (should be converted internally)
        flow = engine.compute_flow(frame1_small, frame2_small, upsample=True)

        assert isinstance(flow, torch.Tensor)
        assert flow.shape[2] == 2

    def test_flow_magnitude_reasonable(self, frame1_tensor, frame2_tensor, skip_if_no_cuda):
        """Test that computed flow has reasonable magnitude."""
        frame1_cuda = frame1_tensor.cuda().byte()
        frame2_cuda = frame2_tensor.cuda().byte()

        engine = TorchNVOpticalFlow.from_tensor(
            frame1_cuda,
            preset="fast",
            grid_size=1
        )

        flow = engine.compute_flow(frame1_cuda, frame2_cuda, upsample=True)

        # Check magnitude
        magnitude = torch.sqrt(flow[:, :, 0]**2 + flow[:, :, 1]**2)

        # Flow magnitude should be reasonable (not too large)
        # For consecutive frames, flow is typically < 100 pixels
        max_magnitude = magnitude.max().item()
        assert max_magnitude < 1000, f"Flow magnitude seems too large: {max_magnitude}"

    def test_flow_not_all_zeros(self, frame1_tensor, frame2_tensor, skip_if_no_cuda):
        """Test that flow is not all zeros (frames should have some motion)."""
        frame1_cuda = frame1_tensor.cuda().byte()
        frame2_cuda = frame2_tensor.cuda().byte()

        engine = TorchNVOpticalFlow.from_tensor(
            frame1_cuda,
            preset="fast",
            grid_size=1
        )

        flow = engine.compute_flow(frame1_cuda, frame2_cuda, upsample=True)

        # Flow should not be all zeros (unless frames are identical)
        flow_mean = flow.abs().mean().item()

        # Check if frames are actually different
        if not torch.equal(frame1_cuda, frame2_cuda):
            assert flow_mean > 0.01, "Flow should have some non-zero values"


@pytest.mark.skipif(not NVOF_AVAILABLE, reason="CUDA or NVOF not available")
class TestNVOFIntegration:
    """Integration tests for NVOF with the test assets."""

    def test_full_pipeline_with_assets(self, frame1_image, frame2_image, temp_output_dir, skip_if_no_cuda):
        """Test complete pipeline: load images -> compute flow -> save results."""
        from of.io import write_flo
        from of.visualization import flow_to_color
        import imageio.v3 as imageio

        # Prepare images
        h, w = min(frame1_image.shape[0], 512), min(frame1_image.shape[1], 512)
        frame1_small = frame1_image[:h, :w]
        frame2_small = frame2_image[:h, :w]

        # Ensure RGB
        if frame1_small.shape[2] == 4:
            frame1_small = frame1_small[:, :, :3]
            frame2_small = frame2_small[:, :, :3]

        # Convert to tensors
        frame1_tensor = torch.from_numpy(frame1_small).cuda()
        frame2_tensor = torch.from_numpy(frame2_small).cuda()

        # Create engine and compute flow
        engine = TorchNVOpticalFlow.from_tensor(frame1_tensor, preset="fast", grid_size=1)
        flow = engine.compute_flow(frame1_tensor, frame2_tensor, upsample=True)

        # Convert to numpy
        flow_np = flow.cpu().numpy()

        # Save flow
        flow_path = temp_output_dir / "test_flow.flo"
        write_flo(flow_path, flow_np)
        assert flow_path.exists()

        # Visualize flow
        flow_vis = flow_to_color(flow_np)
        vis_path = temp_output_dir / "test_flow_vis.png"
        imageio.imwrite(vis_path, flow_vis)
        assert vis_path.exists()
