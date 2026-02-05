"""Tests for optical flow error metrics."""

import pytest
import numpy as np

from of.metrics import (
    endpoint_error,
    average_endpoint_error,
    angular_error,
    average_angular_error,
)


class TestEndpointError:
    """Test endpoint error (EPE) metrics."""

    def test_epe_identical_flows(self):
        """Test EPE between identical flows should be zero."""
        flow = np.random.randn(50, 60, 2).astype(np.float32)
        epe = endpoint_error(flow, flow)

        assert epe.shape == (50, 60), "EPE should be 2D"
        np.testing.assert_array_almost_equal(epe, 0.0, decimal=5)

    def test_epe_zero_flows(self):
        """Test EPE between zero flows."""
        flow1 = np.zeros((50, 60, 2), dtype=np.float32)
        flow2 = np.zeros((50, 60, 2), dtype=np.float32)

        epe = endpoint_error(flow1, flow2)

        np.testing.assert_array_equal(epe, 0.0)

    def test_epe_simple_case(self):
        """Test EPE with simple known case."""
        flow_pred = np.zeros((10, 10, 2), dtype=np.float32)
        flow_gt = np.zeros((10, 10, 2), dtype=np.float32)

        # Create a known error at one pixel
        flow_pred[5, 5, 0] = 3.0  # u = 3
        flow_pred[5, 5, 1] = 4.0  # v = 4
        # Expected EPE at (5, 5) = sqrt(9 + 16) = 5

        epe = endpoint_error(flow_pred, flow_gt)

        assert epe[5, 5] == pytest.approx(5.0)
        assert epe[0, 0] == 0.0

    def test_epe_with_valid_mask(self):
        """Test EPE with valid mask."""
        flow_pred = np.ones((10, 10, 2), dtype=np.float32)
        flow_gt = np.zeros((10, 10, 2), dtype=np.float32)

        # Create a mask that only considers top-left corner
        valid_mask = np.zeros((10, 10), dtype=bool)
        valid_mask[:5, :5] = True

        epe = endpoint_error(flow_pred, flow_gt, valid_mask)

        # Masked region should have EPE
        assert epe[0, 0] > 0

        # Outside mask should be zero (masked out)
        assert epe[9, 9] == 0.0

    def test_average_epe_identical(self):
        """Test average EPE for identical flows."""
        flow = np.random.randn(50, 60, 2).astype(np.float32)
        aee = average_endpoint_error(flow, flow)

        assert aee == pytest.approx(0.0, abs=1e-5)

    def test_average_epe_known_value(self):
        """Test average EPE with known value."""
        flow_pred = np.ones((10, 10, 2), dtype=np.float32)  # All (1, 1)
        flow_gt = np.zeros((10, 10, 2), dtype=np.float32)  # All (0, 0)

        aee = average_endpoint_error(flow_pred, flow_gt)

        # EPE at each pixel = sqrt(1^2 + 1^2) = sqrt(2)
        expected = np.sqrt(2)
        assert aee == pytest.approx(expected)

    def test_average_epe_with_valid_mask(self):
        """Test average EPE with valid mask."""
        flow_pred = np.ones((10, 10, 2), dtype=np.float32)
        flow_gt = np.zeros((10, 10, 2), dtype=np.float32)

        # Mask out half the pixels
        valid_mask = np.zeros((10, 10), dtype=bool)
        valid_mask[:5, :] = True

        aee_full = average_endpoint_error(flow_pred, flow_gt)
        aee_masked = average_endpoint_error(flow_pred, flow_gt, valid_mask)

        # Should be the same since the flow is uniform
        assert aee_full == pytest.approx(aee_masked)


class TestAngularError:
    """Test angular error (AE) metrics."""

    def test_ae_identical_flows(self):
        """Test AE between identical flows should be very close to zero."""
        flow = np.random.randn(50, 60, 2).astype(np.float32)
        ae = angular_error(flow, flow)

        assert ae.shape == (50, 60), "AE should be 2D"
        # Due to floating point precision, allow small tolerance
        np.testing.assert_array_almost_equal(ae, 0.0, decimal=1)

    def test_ae_zero_flows(self):
        """Test AE between zero flows."""
        flow1 = np.zeros((50, 60, 2), dtype=np.float32)
        flow2 = np.zeros((50, 60, 2), dtype=np.float32)

        ae = angular_error(flow1, flow2)

        np.testing.assert_array_almost_equal(ae, 0.0, decimal=5)

    def test_ae_opposite_directions(self):
        """Test AE for flows in opposite directions."""
        flow1 = np.zeros((10, 10, 2), dtype=np.float32)
        flow2 = np.zeros((10, 10, 2), dtype=np.float32)

        flow1[5, 5, 0] = 1.0  # Rightward
        flow2[5, 5, 0] = -1.0  # Leftward

        ae = angular_error(flow1, flow2)

        # Opposite directions should give large angle (close to 90 degrees)
        # Due to the 3D vector (u, v, 1) representation, opposite horizontal flows give ~90 degrees
        assert ae[5, 5] >= 85.0, "Opposite flows should have large angular error"

    def test_ae_perpendicular_directions(self):
        """Test AE for perpendicular flows."""
        flow1 = np.zeros((10, 10, 2), dtype=np.float32)
        flow2 = np.zeros((10, 10, 2), dtype=np.float32)

        flow1[5, 5, 0] = 1.0  # Rightward
        flow2[5, 5, 1] = 1.0  # Downward

        ae = angular_error(flow1, flow2)

        # Perpendicular should give around 45-60 degrees
        # (accounting for the (u,v,1) 3D vector representation)
        assert 30 < ae[5, 5] < 70, "Perpendicular flows should have moderate angular error"

    def test_ae_with_valid_mask(self):
        """Test AE with valid mask."""
        flow_pred = np.ones((10, 10, 2), dtype=np.float32)
        flow_gt = -np.ones((10, 10, 2), dtype=np.float32)

        valid_mask = np.zeros((10, 10), dtype=bool)
        valid_mask[:5, :5] = True

        ae = angular_error(flow_pred, flow_gt, valid_mask)

        # Masked region should have AE
        assert ae[0, 0] > 0

        # Outside mask should be zero
        assert ae[9, 9] == 0.0

    def test_average_ae_identical(self):
        """Test average AE for identical flows should be very small."""
        flow = np.random.randn(50, 60, 2).astype(np.float32)
        aae = average_angular_error(flow, flow)

        # Due to floating point precision, expect very small but not exactly zero
        assert aae < 0.1, f"Average AE should be very small for identical flows, got {aae}"

    def test_average_ae_range(self):
        """Test that average AE is in valid range [0, 180]."""
        flow_pred = np.random.randn(50, 60, 2).astype(np.float32)
        flow_gt = np.random.randn(50, 60, 2).astype(np.float32)

        aae = average_angular_error(flow_pred, flow_gt)

        assert 0 <= aae <= 180, "Average angular error should be in [0, 180] degrees"

    def test_average_ae_with_valid_mask(self):
        """Test average AE with valid mask."""
        flow_pred = np.ones((10, 10, 2), dtype=np.float32)
        flow_gt = np.zeros((10, 10, 2), dtype=np.float32)

        valid_mask = np.zeros((10, 10), dtype=bool)
        valid_mask[:5, :] = True

        aae = average_angular_error(flow_pred, flow_gt, valid_mask)

        assert 0 <= aae <= 180
