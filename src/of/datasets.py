"""
Dataset loaders for optical flow benchmarks.
"""

import numpy as np
from pathlib import Path
from typing import List, Dict, Optional, Union
from dataclasses import dataclass
import imageio.v3 as imageio

from .io import read_flo


@dataclass
class FlowSample:
    """A single optical flow sample with metadata."""

    current_frame: Path
    reference_frame: Path
    fw_flow: Optional[Path] = None
    bw_flow: Optional[Path] = None
    fw_valid_mask: Optional[Path] = None
    bw_valid_mask: Optional[Path] = None
    name: str = ""
    metadata: Dict = None

    def __post_init__(self):
        if self.metadata is None:
            self.metadata = {}


class OpticalFlowDataset:
    """Base class for optical flow datasets."""

    def __init__(self, root_dir: Union[str, Path]):
        self.root_dir = Path(root_dir)
        self.samples: List[FlowSample] = []
        self.setup()

    def setup(self):
        raise NotImplementedError("Subclasses should implement this method.")

    def __len__(self) -> int:
        return len(self.samples)

    def __getitem__(self, idx: int) -> FlowSample:
        raise NotImplementedError("Subclasses should implement this method.")


class CSEMDataset(OpticalFlowDataset):
    """
    project_root/
    ├── images/
    │   ├── frame_0001.png
    │   ├── frame_0002.png
    |   ├── ...
    │   └── frame_N.png
    ├── flow/
    │   ├── forward/          # Flow from t to t+1
    │   │   ├── flow_0001.flo
    │   │   ├── flow_0002.flo
    │   │   ├── ...
    │   │   └── flow_<N-1>.flo
    │   └── backward/         # Flow from t to t-1 (for occlusion checks)
    │       ├── flow_0002.flo
    │       ├── flow_0003.flo
    │       ├── ...
    │       └── flow_<N>.flo
    """

    def setup(self):
        img_dir = self.root_dir / "images"
        if not img_dir.exists():
            raise ValueError(
                f"Images directory not found: {img_dir}. Maybe your dataset is not structured as expected?"
            )

        frame_paths = sorted(img_dir.glob("*.png"))
        for i in range(len(frame_paths) - 1):
            current_frame = frame_paths[i]
            reference_frame = frame_paths[i + 1]
            name = current_frame.stem.split("_")[-1]
            frame_number = int(name)
            fw_flow_path = self.root_dir / "flow" / "forward" / f"flow_{name}.flo"
            bw_flow_path = self.root_dir / "flow" / "backward" / f"flow_{name}.flo"
            sample = FlowSample(
                current_frame=current_frame,
                reference_frame=reference_frame,
                fw_flow=fw_flow_path if fw_flow_path.exists() else None,
                bw_flow=bw_flow_path if bw_flow_path.exists() else None,
                name=name,
                metadata={"frame_number": frame_number},
            )
            self.samples.append(sample)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        sample = self.samples[idx]
        data = {
            "current_frame": imageio.imread(sample.current_frame),
            "reference_frame": imageio.imread(sample.reference_frame),
            "name": sample.name,
            "metadata": sample.metadata,
        }
        if sample.fw_flow:
            data["fw_flow"] = read_flo(sample.fw_flow)
        if sample.bw_flow:
            data["bw_flow"] = read_flo(sample.bw_flow)
        return data


class MPISintelDataset(OpticalFlowDataset):
    """
    MPI-Sintel optical flow dataset.

    The dataset has two passes: 'clean' and 'final'.
    Directory structure:
        root/split/pass_name/sequence/frame_xxxx.png
        root/split/flow/sequence/frame_xxxx.flo
        root/split/invalid/sequence/frame_xxxx.png
    """

    def __init__(
        self,
        root_dir: Union[str, Path],
        split: str = "training",
        pass_name: str = "clean",
    ):
        """
        Initialize MPI-Sintel dataset.

        Args:
            root_dir: Path to MPI-Sintel root directory
            split: Either 'training' or 'test'
            pass_name: Either 'clean' or 'final'
        """
        # Set attributes before super().__init__ because it calls setup()
        self.split = split
        self.pass_name = pass_name

        super().__init__(root_dir)

    def setup(self):
        """Build list of samples from directory structure."""
        images_dir = self.root_dir / self.split / self.pass_name

        if not images_dir.exists():
            raise ValueError(f"Images directory not found: {images_dir}")

        flow_dir = self.root_dir / self.split / "flow"
        # Sintel provides 'invalid' masks (occlusions + out of bounds)
        # We map this to fw_valid_mask (loading logic should handle the inversion if needed)
        invalid_dir = self.root_dir / self.split / "invalid"

        # Iterate through sequences
        for seq_dir in sorted(images_dir.iterdir()):
            if not seq_dir.is_dir():
                continue

            seq_name = seq_dir.name
            # Get all image pairs in the sequence
            image_files = sorted(seq_dir.glob("*.png"))

            for i in range(len(image_files) - 1):
                img1 = image_files[i]
                img2 = image_files[i + 1]

                fw_flow_path = None
                fw_valid_mask_path = None

                # Training split has ground truth flow and masks
                if self.split == "training":
                    # Flow filename matches image filename
                    cand_flow = flow_dir / seq_name / img1.name.replace(".png", ".flo")
                    if cand_flow.exists():
                        fw_flow_path = cand_flow

                    cand_mask = invalid_dir / seq_name / img1.name
                    if cand_mask.exists():
                        fw_valid_mask_path = cand_mask

                sample = FlowSample(
                    current_frame=img1,
                    reference_frame=img2,
                    fw_flow=fw_flow_path,
                    bw_flow=None,  # Sintel standard structure doesn't provide backward flow
                    fw_valid_mask=fw_valid_mask_path,
                    bw_valid_mask=None,
                    name=f"{seq_name}_{img1.stem}",
                    metadata={
                        "sequence": seq_name,
                        "pass": self.pass_name,
                        "frame_index": i,
                    },
                )

                self.samples.append(sample)

    def __getitem__(self, idx: int) -> Dict[str, np.ndarray]:
        sample = self.samples[idx]
        data = {
            "current_frame": imageio.imread(sample.current_frame),
            "reference_frame": imageio.imread(sample.reference_frame),
            "name": sample.name,
            "metadata": sample.metadata,
        }

        if sample.fw_flow:
            data["fw_flow"] = read_flo(sample.fw_flow)

        if sample.fw_valid_mask:
            # Sintel masks are PNGs where logic 1 usually means invalid.
            # Reading as is; downstream transforms can invert/threshold.
            data["fw_valid_mask"] = imageio.imread(sample.fw_valid_mask)

        return data


def get_dataset(name: str, root_dir: Union[str, Path], **kwargs) -> OpticalFlowDataset:
    """
    Factory function to get a dataset by name.

    Args:
        name: Dataset name ('mpi-sintel')
        root_dir: Root directory of the dataset
        **kwargs: Additional arguments for the dataset

    Returns:
        OpticalFlowDataset instance
    """
    name = name.lower()

    if name == "mpi-sintel":
        return MPISintelDataset(root_dir, **kwargs)
    else:
        raise ValueError(f"Unknown dataset: {name}. Only 'mpi-sintel' is supported.")
