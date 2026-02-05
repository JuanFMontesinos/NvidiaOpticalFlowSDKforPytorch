#!/usr/bin/env python3
"""
Description:
    This script processes the MPI-Sintel optical flow dataset, computing
    optical flow predictions using NVIDIA Optical Flow (NVOF) and comparing
    them against ground truth. For each sequence, it creates a video that
    stitches together the original frame, predicted optical flow, and ground
    truth flow (if available). It also computes detailed metrics (EPE, AE,
    RMSE, FL) for each frame and sequence, saving the results to JSON.

How to run:
    uv run examples/render_sintel_videos_and_compute_metrics.py

"""

import json
from pathlib import Path
from collections import defaultdict

import torch
import numpy as np
import imageio.v3 as imageio
from tqdm import tqdm

from of import TorchNVOpticalFlow, datasets
from of.visualization import concatenate_flows
from of.metrics import compute_all_metrics

# ========== Configuration ==========
SINTEL_ROOT = Path("/mnt/DataNMVE/optical_flow_datasets/MPI-Sintel")  # Path to MPI-Sintel dataset
PASS_NAME = "clean"  # or "final"
PRESET = "slow"  # "slow", "medium", or "fast"
GRID_SIZE = 1  # 1, 2, or 4
OUTPUT_DIR = Path(f"render_of_sintel_{PASS_NAME}_{PRESET}_grid{GRID_SIZE}")
FPS = 5

OUTPUT_DIR.mkdir(exist_ok=True)


if not torch.cuda.is_available():
    raise RuntimeError("Error: CUDA is not available. NVOF requires a GPU.")


def convert_to_serializable(obj):
    """Convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_to_serializable(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_to_serializable(item) for item in obj]
    else:
        return obj


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

# ========== 3. Initialize NVOF engine ==========
flow_engine = None

# ========== 3.5. Initialize metrics storage ==========
all_metrics = {
    "dataset": "MPI-Sintel",
    "split": "training",
    "pass": PASS_NAME,
    "preset": PRESET,
    "grid_size": GRID_SIZE,
    "sequences": {},
}

# ========== 4. Process each sequence ==========
for seq_name, sample_indices in tqdm(sequences.items(), desc="Processing sequences"):
    print(f"\nProcessing sequence: {seq_name} ({len(sample_indices)} frames)")

    frames_for_video = []
    sequence_metrics = {"num_frames": len(sample_indices), "frames": []}

    for idx in tqdm(sample_indices, desc=f"  {seq_name}", leave=False):
        # Load sample
        sample_data = dataset[idx]
        image1 = sample_data["current_frame"]
        image2 = sample_data["reference_frame"]
        gt_flow = sample_data.get("fw_flow")
        frame_idx = sample_data["metadata"]["frame_index"]

        # Initialize flow engine on first sample
        if flow_engine is None:
            image1_cu = torch.from_numpy(image1).cuda()

            flow_engine = TorchNVOpticalFlow.from_tensor(
                image1_cu,
                preset=PRESET,
                grid_size=GRID_SIZE,
            )
            print(f"  Initialized NVOF engine for resolution {image1.shape[:2]}")

        # Convert to CUDA tensors
        image1_cu = torch.from_numpy(image1).cuda()
        image2_cu = torch.from_numpy(image2).cuda()

        # Compute predicted flow with timing
        start = torch.cuda.Event(enable_timing=True)
        end = torch.cuda.Event(enable_timing=True)

        start.record()
        pred_flow = flow_engine.compute_flow(image1_cu, image2_cu, upsample=True).cpu().numpy()
        end.record()
        torch.cuda.synchronize()

        compute_time_ms = start.elapsed_time(end)

        # Compute EPE if ground truth is available
        epe_text = ""
        frame_metrics = {
            "frame_index": frame_idx,
            "frame_name": sample_data["name"],
            "compute_time_ms": compute_time_ms,
        }

        if gt_flow is not None:
            # Create valid mask (Sintel uses invalid mask, so we need to invert it)
            valid_mask = (np.abs(gt_flow[:, :, 0]) < 1e9) & (np.abs(gt_flow[:, :, 1]) < 1e9)

            # Compute detailed metrics
            metrics = compute_all_metrics(pred_flow, gt_flow, valid_mask)
            frame_metrics.update(metrics)

            avg_epe = metrics["EPE"]
            epe_text = f"EPE: {avg_epe:.2f}"

        sequence_metrics["frames"].append(frame_metrics)

        # Prepare arrays for concatenation
        arrays = [image1, pred_flow]
        positions = [(0, 0), (1, 0)]
        kwargs_list = [
            {
                "caption": f"Frame {frame_idx}",
                "font_scale": 0.8,
                "font_thickness": 2,
                "font_color": (255, 255, 255),
                "text_pos": (20, 40),
            },
            {
                "caption": f"Pred Flow {epe_text}",
                "convention": "kitti",
                "font_scale": 0.8,
                "font_thickness": 2,
                "font_color": (255, 255, 255),
                "text_pos": (20, 40),
            },
        ]

        # Add ground truth if available
        if gt_flow is not None:
            arrays.append(gt_flow)
            positions.append((2, 0))
            kwargs_list.append(
                {
                    "caption": "GT Flow",
                    "convention": "kitti",
                    "font_scale": 0.8,
                    "font_thickness": 2,
                    "font_color": (255, 255, 255),
                    "text_pos": (20, 40),
                }
            )

        # Stitch frames together
        stitched_frame = concatenate_flows(
            arrays=arrays,
            positions=positions,
            kwargs_list=kwargs_list,
        )

        frames_for_video.append(stitched_frame)

    # ========== 5. Compute sequence-level metrics ==========
    if sequence_metrics["frames"]:
        # Calculate average metrics for the sequence
        frame_metrics_list = sequence_metrics["frames"]

        # Compute time statistics
        sequence_metrics["average_compute_time_ms"] = np.mean([f["compute_time_ms"] for f in frame_metrics_list])
        sequence_metrics["min_compute_time_ms"] = np.min([f["compute_time_ms"] for f in frame_metrics_list])
        sequence_metrics["max_compute_time_ms"] = np.max([f["compute_time_ms"] for f in frame_metrics_list])

        if frame_metrics_list and "EPE" in frame_metrics_list[0]:
            sequence_metrics["average_EPE"] = np.mean([f["EPE"] for f in frame_metrics_list])
            sequence_metrics["average_AE"] = np.mean([f["AE"] for f in frame_metrics_list])
            sequence_metrics["average_RMSE"] = np.mean([f["RMSE"] for f in frame_metrics_list])

            # FL metrics at different thresholds
            for threshold in [1.0, 3.0, 5.0]:
                key = f"FL-{threshold}px"
                if key in frame_metrics_list[0]:
                    sequence_metrics[f"average_{key}"] = np.mean([f[key] for f in frame_metrics_list])

    all_metrics["sequences"][seq_name] = sequence_metrics

    # ========== 6. Write video ==========
    if frames_for_video:
        output_path = OUTPUT_DIR / f"{seq_name}_{PASS_NAME}.mp4"
        print(f"  Writing video to {output_path}")

        # Write video with imageio
        imageio.imwrite(
            output_path,
            frames_for_video,
            fps=FPS,
            codec="libx264",
            quality=8,
            pixelformat="yuv420p",
        )

        print(f"  Saved video with {len(frames_for_video)} frames")

# ========== 7. Compute overall metrics and save JSON ==========
print("\nComputing overall metrics...")

# Calculate overall average metrics
all_sequences = list(all_metrics["sequences"].values())
if all_sequences:
    # Overall timing statistics
    all_metrics["overall"] = {
        "num_sequences": len(all_sequences),
        "total_frames": sum(s["num_frames"] for s in all_sequences),
        "average_compute_time_ms": np.mean([s["average_compute_time_ms"] for s in all_sequences]),
        "min_compute_time_ms": np.min([s["min_compute_time_ms"] for s in all_sequences]),
        "max_compute_time_ms": np.max([s["max_compute_time_ms"] for s in all_sequences]),
    }

    if "average_EPE" in all_sequences[0]:
        all_metrics["overall"]["average_EPE"] = np.mean([s["average_EPE"] for s in all_sequences])
        all_metrics["overall"]["average_AE"] = np.mean([s["average_AE"] for s in all_sequences])
        all_metrics["overall"]["average_RMSE"] = np.mean([s["average_RMSE"] for s in all_sequences])

        # FL metrics
        for threshold in [1.0, 3.0, 5.0]:
            key = f"average_FL-{threshold}px"
            if key in all_sequences[0]:
                all_metrics["overall"][key] = np.mean([s[key] for s in all_sequences])

# Save metrics to JSON
metrics_path = OUTPUT_DIR / "metrics_summary.json"
print(f"Saving metrics to {metrics_path}")

# Convert numpy types to Python native types for JSON serialization
serializable_metrics = convert_to_serializable(all_metrics)

with open(metrics_path, "w") as f:
    json.dump(serializable_metrics, f, indent=2)

print(f"\n✓ Metrics summary saved to: {metrics_path}")

print("\n✓ All sequences processed!")
print(f"Videos saved to: {OUTPUT_DIR}")
