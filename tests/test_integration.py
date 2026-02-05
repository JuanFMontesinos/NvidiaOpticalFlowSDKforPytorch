"""Integration tests using the assets directory."""

import pytest
import numpy as np

from of.io import read_flo, write_flo
from of.visualization import flow_to_color
from of.metrics import average_endpoint_error, average_angular_error


class TestIntegrationWithAssets:
    """Integration tests using actual test assets."""

    def test_load_frames_from_assets(self, frame1_image, frame2_image):
        """Test loading frames from assets directory."""
        assert frame1_image is not None
        assert frame2_image is not None
        assert frame1_image.shape[0] > 0
        assert frame1_image.shape[1] > 0
        assert frame2_image.shape == frame1_image.shape or frame2_image.shape[:2] == frame1_image.shape[:2]

    def test_load_gt_flow_from_assets(self, gt_flow_path):
        """Test loading ground truth flow if available."""
        if gt_flow_path.exists():
            flow = read_flo(gt_flow_path)
            assert flow.ndim == 3
            assert flow.shape[2] == 2
            assert flow.dtype == np.float32

    def test_visualize_gt_flow(self, gt_flow_path, temp_output_dir):
        """Test visualizing ground truth flow."""
        if not gt_flow_path.exists():
            pytest.skip("Ground truth flow not available")

        import imageio.v3 as imageio

        flow = read_flo(gt_flow_path)
        flow_rgb = flow_to_color(flow)

        # Save visualization
        output_path = temp_output_dir / "gt_flow_vis.png"
        imageio.imwrite(output_path, flow_rgb)

        assert output_path.exists()

    def test_flow_io_roundtrip_with_assets(self, gt_flow_path, temp_output_dir):
        """Test saving and loading flow with actual data."""
        if not gt_flow_path.exists():
            pytest.skip("Ground truth flow not available")

        # Read original
        flow_original = read_flo(gt_flow_path)

        # Save to new location
        new_path = temp_output_dir / "flow_copy.flo"
        write_flo(new_path, flow_original)

        # Read back
        flow_loaded = read_flo(new_path)

        # Compare
        np.testing.assert_array_equal(flow_original, flow_loaded)

    def test_metrics_on_gt_flow(self, gt_flow_path):
        """Test computing metrics between GT flow and slightly perturbed version."""
        if not gt_flow_path.exists():
            pytest.skip("Ground truth flow not available")

        flow_gt = read_flo(gt_flow_path)

        # Create a slightly perturbed version
        noise = np.random.randn(*flow_gt.shape) * 0.1
        flow_pred = flow_gt + noise.astype(np.float32)

        # Compute metrics
        aee = average_endpoint_error(flow_pred, flow_gt)
        aae = average_angular_error(flow_pred, flow_gt)

        # Should have small but non-zero error
        assert 0 < aee < 1.0, "EPE should be small but non-zero"
        assert 0 < aae < 10.0, "Angular error should be small but non-zero"

    def test_multiple_visualizations(self, gt_flow_path, temp_output_dir):
        """Test creating multiple visualizations with different conventions."""
        if not gt_flow_path.exists():
            pytest.skip("Ground truth flow not available")

        import imageio.v3 as imageio

        flow = read_flo(gt_flow_path)

        # Middlebury convention
        flow_mb = flow_to_color(flow, convention="middlebury")
        imageio.imwrite(temp_output_dir / "flow_middlebury.png", flow_mb)

        # KITTI convention
        flow_kitti = flow_to_color(flow, convention="kitti")
        imageio.imwrite(temp_output_dir / "flow_kitti.png", flow_kitti)

        # Check both were created
        assert (temp_output_dir / "flow_middlebury.png").exists()
        assert (temp_output_dir / "flow_kitti.png").exists()

        # They should be different
        assert not np.array_equal(flow_mb, flow_kitti)

    def test_flow_statistics(self, gt_flow_path):
        """Test computing basic statistics on flow."""
        if not gt_flow_path.exists():
            pytest.skip("Ground truth flow not available")

        flow = read_flo(gt_flow_path)

        # Compute magnitude
        u = flow[:, :, 0]
        v = flow[:, :, 1]
        magnitude = np.sqrt(u**2 + v**2)

        # Check that we have some valid flow
        valid_pixels = ~(np.isnan(magnitude) | np.isinf(magnitude))
        assert valid_pixels.sum() > 0, "Should have some valid flow pixels"

        if valid_pixels.sum() > 0:
            mean_mag = magnitude[valid_pixels].mean()
            max_mag = magnitude[valid_pixels].max()

            print("\nFlow statistics:")
            print(f"  Mean magnitude: {mean_mag:.2f}")
            print(f"  Max magnitude: {max_mag:.2f}")
            print(f"  Valid pixels: {valid_pixels.sum()} / {valid_pixels.size}")

            assert mean_mag >= 0
            assert max_mag >= mean_mag

    def test_frame_dimensions_match(self, frame1_image, frame2_image, gt_flow_path):
        """Test that frame and flow dimensions are consistent."""
        if not gt_flow_path.exists():
            pytest.skip("Ground truth flow not available")

        flow = read_flo(gt_flow_path)

        # Flow dimensions should match frame dimensions
        assert flow.shape[0] == frame1_image.shape[0], "Flow height should match frame height"
        assert flow.shape[1] == frame1_image.shape[1], "Flow width should match frame width"
