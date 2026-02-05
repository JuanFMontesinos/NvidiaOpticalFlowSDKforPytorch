"""
Description:
    This script compares the performance of NVIDIA Optical Flow (NVOF)
    using different grid sizes ("fast" with grid=4 vs "slow" with grid=1)
    on the MPI-Sintel dataset. It computes optical flow for each frame pair
    in the dataset, stitches together the original frame, ground truth flow,
    and predicted flows into a 2x2 grid, and saves the results as videos
    for each sequence.

How to run:
    uv run examples/compare_fast_vs_slow_and_render_video.py

"""

from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm

from of import TorchNVOpticalFlow, datasets
from of.visualization import concatenate_flows

# ========== Configuration ==========
SINTEL_ROOT = Path("/mnt/DataNMVE/optical_flow_datasets/MPI-Sintel")  # Path to MPI-Sintel dataset
PASS_NAME = "clean"  # or "final"
OUTPUT_DIR = Path(f"render_of_sintel_comparison_{PASS_NAME}")
FPS = 5

OUTPUT_DIR.mkdir(exist_ok=True)


if not torch.cuda.is_available():
    raise RuntimeError("Error: CUDA is not available. NVOF requires a GPU.")


# ========== 1. Load Dataset ==========
print(f"Loading MPI-Sintel dataset ({PASS_NAME} pass)...")

dataset = datasets.MPISintelDataset(root_dir=SINTEL_ROOT, split="training", pass_name=PASS_NAME)
print(f"Found {len(dataset)} samples")

# ========== 2. Group samples by sequence ==========
print("Grouping samples by sequence...")
sequences = defaultdict(list)
for idx in range(len(dataset)):
    sample = dataset.samples[idx]
    seq_name = sample.metadata["sequence"]
    sequences[seq_name].append(idx)

print(f"Found {len(sequences)} sequences")

# ========== 3. Initialize NVOF engines ==========
print("Initializing NVOF engines...")
flow_engine_fast = None
flow_engine_slow = None

# ========== 4. Process each sequence ==========
for seq_name, sample_indices in tqdm(sequences.items(), desc="Processing sequences"):
    print(f"\nProcessing sequence: {seq_name} ({len(sample_indices)} frames)")

    video_frames = []

    for idx in tqdm(sample_indices, desc=f"  {seq_name}", leave=False):
        # Load sample
        sample_data = dataset[idx]
        image1_np = sample_data["current_frame"]
        image2_np = sample_data["reference_frame"]
        image1_cu = torch.from_numpy(image1_np).cuda()
        image2_cu = torch.from_numpy(image2_np).cuda()
        gt_flow = sample_data.get("fw_flow")
        frame_idx = sample_data["metadata"]["frame_index"]

        # Initialize flow engines on first sample
        if flow_engine_fast is None or flow_engine_slow is None:
            flow_engine_fast = TorchNVOpticalFlow.from_tensor(
                image1_cu,
                preset="fast",
                grid_size=4,
            )

            flow_engine_slow = TorchNVOpticalFlow.from_tensor(
                image1_cu,
                preset="slow",
                grid_size=1,
            )
            print(f"  Initialized NVOF engines for resolution {image1_cu.shape[:2]}")

        # Compute predicted flows
        flow_fast = flow_engine_fast.compute_flow(image1_cu, image2_cu, upsample=True).cpu().numpy()

        flow_slow = flow_engine_slow.compute_flow(image1_cu, image2_cu, upsample=True).cpu().numpy()

        # Prepare arrays for concatenation in 2x2 grid
        # Top row: RGB (left), GT flow (right)
        # Bottom row: Fast grid=4 (left), Slow grid=1 (right)

        arrays = [
            image1_np,  # Top left
            gt_flow if gt_flow is not None else np.zeros_like(flow_fast),  # Top right
            flow_fast,  # Bottom left
            flow_slow,  # Bottom right
        ]

        positions = [
            (0, 0),  # Top left
            (1, 0),  # Top right
            (0, 1),  # Bottom left
            (1, 1),  # Bottom right
        ]

        kwargs_list = [
            {
                "caption": f"Frame {frame_idx}",
                "font_scale": 0.8,
                "font_thickness": 2,
                "font_color": (255, 255, 255),
                "text_pos": (20, 40),
            },
            {
                "caption": "GT Flow",
                "convention": "kitti",
                "font_scale": 0.8,
                "font_thickness": 2,
                "font_color": (255, 255, 255),
                "text_pos": (20, 40),
            },
            {
                "caption": "Fast (grid=4)",
                "convention": "kitti",
                "font_scale": 0.8,
                "font_thickness": 2,
                "font_color": (255, 255, 255),
                "text_pos": (20, 40),
            },
            {
                "caption": "Slow (grid=1)",
                "convention": "kitti",
                "font_scale": 0.8,
                "font_thickness": 2,
                "font_color": (255, 255, 255),
                "text_pos": (20, 40),
            },
        ]

        # Stitch frames together
        stitched_frame = concatenate_flows(
            arrays=arrays,
            positions=positions,
            kwargs_list=kwargs_list,
        )

        video_frames.append(stitched_frame)

    # ========== 5. Write video ==========
    if video_frames:
        output_path = OUTPUT_DIR / f"{seq_name}_{PASS_NAME}_comparison.mp4"
        print(f"  Writing video to {output_path}")

        # Write video with imageio
        imageio.imwrite(
            output_path,
            video_frames,
            fps=FPS,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )

        print(f"  Saved video with {len(video_frames)} frames")

print("\n✓ All sequences processed!")
print(f"Videos saved to: {OUTPUT_DIR}")
