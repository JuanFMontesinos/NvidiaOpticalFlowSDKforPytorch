#!/usr/bin/env python3
"""
Minimal example demonstrating NVIDIA Optical Flow SDK usage.

This script:
1. Loads two consecutive frames from assets/
2. Computes optical flow using NVOF
3. Visualizes and saves the results
"""

from pathlib import Path

import torch
import imageio.v3 as imageio

from of import TorchNVOpticalFlow
from of.io import read_flo, write_flo
from of.visualization import flow_to_color, concatenate_flows


def main():
    # Check for CUDA availability
    if not torch.cuda.is_available():
        print("Error: CUDA is not available. NVOF requires a GPU.")
        return

    print("NVIDIA Optical Flow - Minimal Example")
    print("=" * 50)

    # ========== 1. Setup Paths ==========
    script_dir = Path(__file__).parent
    project_root = script_dir.parent
    assets_dir = project_root / "assets"
    output_dir = project_root / "output"

    # Create output directory
    output_dir.mkdir(exist_ok=True)

    # Define input frames
    frame1_path = assets_dir / "frame_0001.png"
    frame2_path = assets_dir / "frame_0002.png"
    gt_flow_path = assets_dir / "frame_0001.flo"

    # ========== 2. Load Images ==========
    print(f"\n📁 Loading frames from {assets_dir}")
    frame1 = imageio.imread(frame1_path)
    frame2 = imageio.imread(frame2_path)

    print(f"   Frame 1: {frame1.shape} {frame1.dtype}")
    print(f"   Frame 2: {frame2.shape} {frame2.dtype}")

    # Load ground truth flow if available
    gt_flow = None
    if gt_flow_path.exists():
        gt_flow = read_flo(gt_flow_path)
        print(f"   GT Flow: {gt_flow.shape} {gt_flow.dtype}")

    # ========== 3. Convert to CUDA Tensors ==========
    print("\n🔄 Converting to CUDA tensors...")
    frame1_tensor = torch.from_numpy(frame1).cuda()
    frame2_tensor = torch.from_numpy(frame2).cuda()

    # ========== 4. Initialize Optical Flow Engine ==========
    print("\n⚙️  Initializing NVOF engine...")
    height, width = frame1.shape[:2]

    flow_engine = TorchNVOpticalFlow(
        width=width,
        height=height,
        gpu_id=0,
        preset="fast",  # Options: "slow", "medium", "fast"
        grid_size=4,  # Options: 1, 2, 4 (lower = faster but coarser)
        bidirectional=False,
    )

    print(f"   Resolution: {width}x{height}")
    print(f"   Preset: slow")
    print(f"   Grid size: 1")

    # ========== 5. Compute Optical Flow ==========
    print("\n🔥 Computing optical flow...")

    print(" Upsampling disabled:")
    flow = flow_engine.compute_flow(frame1_tensor, frame2_tensor, upsample=False)
    flow_np = flow.cpu().numpy()
    print(f"   Flow shape: {flow_np.shape}")

    print(" Upsampling enabled:")
    # Without upsampling, output flow is at grid resolution
    flow = flow_engine.compute_flow(frame1_tensor, frame2_tensor, upsample=True)
    flow_np = flow.cpu().numpy()
    print(f"   Flow shape: {flow_np.shape}")

    # ========== 6. Visualize Flow ==========
    print("\n🎨 Generating visualization...")
    flow_rgb = flow_to_color(flow_np, convention="middlebury")

    print(f"   Visualization shape: {flow_rgb.shape}")

    # ========== 7. Save Results ==========
    print(f"\n💾 Saving results to {output_dir}")

    # Save flow as .flo file
    flow_flo_path = output_dir / "flow_output.flo"
    write_flo(flow_flo_path, flow_np)
    print(f"   ✓ Saved flow: {flow_flo_path}")

    # Save visualization as PNG
    flow_vis_path = output_dir / "flow_visualization.png"
    imageio.imwrite(flow_vis_path, flow_rgb)
    print(f"   ✓ Saved visualization: {flow_vis_path}")

    # ========== 8. Create Side-by-Side Comparison ==========
    print("\n📊 Creating side-by-side comparison...")

    # Prepare arrays for concatenation: Frame | GT Flow | Predicted Flow
    arrays = [frame1]
    positions = [(0, 0)]
    kwargs_list = [{"caption": "Input Frame"}]

    if gt_flow is not None:
        arrays.append(gt_flow)
        positions.append((1, 0))
        kwargs_list.append({"caption": "Ground Truth Flow", "convention": "middlebury"})

    arrays.append(flow_np)
    positions.append((2, 0) if gt_flow is not None else (1, 0))
    kwargs_list.append({"caption": "Predicted Flow", "convention": "middlebury"})

    # Create comparison using concatenate_flows
    comparison = concatenate_flows(
        arrays=arrays,
        positions=positions,
        kwargs_list=kwargs_list,
    )

    comparison_path = output_dir / "comparison.png"
    imageio.imwrite(comparison_path, comparison)
    print(f"   ✓ Saved comparison: {comparison_path}")


if __name__ == "__main__":
    main()
