"""Tests for dataset loaders."""

import pytest
from pathlib import Path

from of.datasets import FlowSample, OpticalFlowDataset


class TestFlowSample:
    """Test FlowSample dataclass."""

    def test_flow_sample_creation(self):
        """Test creating a FlowSample."""
        sample = FlowSample(
            current_frame=Path("frame1.png"),
            reference_frame=Path("frame2.png"),
            name="test_sample"
        )

        assert sample.current_frame == Path("frame1.png")
        assert sample.reference_frame == Path("frame2.png")
        assert sample.name == "test_sample"
        assert sample.fw_flow is None
        assert sample.bw_flow is None
        assert sample.metadata == {}

    def test_flow_sample_with_flow_paths(self):
        """Test FlowSample with flow paths."""
        sample = FlowSample(
            current_frame=Path("frame1.png"),
            reference_frame=Path("frame2.png"),
            fw_flow=Path("flow_fw.flo"),
            bw_flow=Path("flow_bw.flo"),
            name="test_sample"
        )

        assert sample.fw_flow == Path("flow_fw.flo")
        assert sample.bw_flow == Path("flow_bw.flo")

    def test_flow_sample_with_metadata(self):
        """Test FlowSample with custom metadata."""
        metadata = {"frame_number": 42, "sequence": "test"}
        sample = FlowSample(
            current_frame=Path("frame1.png"),
            reference_frame=Path("frame2.png"),
            metadata=metadata
        )

        assert sample.metadata["frame_number"] == 42
        assert sample.metadata["sequence"] == "test"


class TestOpticalFlowDataset:
    """Test OpticalFlowDataset base class."""

    def test_dataset_abstract_methods(self):
        """Test that OpticalFlowDataset requires setup and __getitem__."""
        # OpticalFlowDataset calls setup() in __init__, which raises NotImplementedError
        with pytest.raises(NotImplementedError):
            OpticalFlowDataset(Path("/fake/path"))

    def test_dataset_len(self):
        """Test dataset length."""
        # Create a mock dataset
        class MockDataset(OpticalFlowDataset):
            def setup(self):
                self.samples = [
                    FlowSample(Path("f1.png"), Path("f2.png")),
                    FlowSample(Path("f2.png"), Path("f3.png")),
                ]

            def __getitem__(self, idx):
                return self.samples[idx]

        dataset = MockDataset(Path("/fake/path"))
        assert len(dataset) == 2
