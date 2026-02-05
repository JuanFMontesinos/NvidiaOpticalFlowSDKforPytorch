"""
Visualization utilities for optical flow.

INTERPRETATION GUIDE:
---------------------
To correctly read these visualizations, you must distinguish between the "Color Wheel"
logic and "Masking" logic.

1. PIXEL COLORS (Valid Flow):
   - WHITE:        Zero Motion (Stationary).
   - COLOR:        Motion Direction (Hue) and Speed (Saturation).
                   (Red=Right, Green=Up, Cyan=Left, Blue=Down).
   - BRIGHT/VIVID: Fast motion.
   - PALE/PASTEL:  Slow motion.

2. BLACK PIXELS (Invalid/Occluded):
   - BLACK:        Invalid data, NaN, Occluded, or Unknown.

   *Note on KITTI*: In KITTI ground truth, the sky and upper image are invalid.
   They appear BLACK. The stationary road appears WHITE. If you see a visualization
   that is Black where the road should be, the data is likely NaN/Invalid, not stationary.

3. CONVENTIONS (Normalization):
   - 'middlebury': Adapts to the image's max speed. Good for seeing details in
                   slow scenes.
   - 'kitti':      Uses fixed scaling (usually saturates at 3px or similar).
                   Good for comparing speed between different clips (fast scenes
                   look vivid, slow scenes look white/pale).
"""

from typing import List, Optional, Tuple, Union, Dict, Any

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.axes import Axes
from matplotlib.figure import Figure


def _get_color_wheel() -> np.ndarray:
    """
    Generates the standard optical flow color wheel (Middlebury/KITTI standard).

    Returns:
        np.ndarray: Color wheel palette of shape (55, 3)
    """
    RY = 15
    YG = 6
    GC = 4
    CB = 11
    BM = 13
    MR = 6

    ncols = RY + YG + GC + CB + BM + MR
    colorwheel = np.zeros((ncols, 3))

    col = 0
    # RY
    colorwheel[0:RY, 0] = 255
    colorwheel[0:RY, 1] = np.floor(255 * np.arange(0, RY, 1) / RY)
    col += RY

    # YG
    colorwheel[col : col + YG, 0] = 255 - np.floor(255 * np.arange(0, YG, 1) / YG)
    colorwheel[col : col + YG, 1] = 255
    col += YG

    # GC
    colorwheel[col : col + GC, 1] = 255
    colorwheel[col : col + GC, 2] = np.floor(255 * np.arange(0, GC, 1) / GC)
    col += GC

    # CB
    colorwheel[col : col + CB, 1] = 255 - np.floor(255 * np.arange(0, CB, 1) / CB)
    colorwheel[col : col + CB, 2] = 255
    col += CB

    # BM
    colorwheel[col : col + BM, 2] = 255
    colorwheel[col : col + BM, 0] = np.floor(255 * np.arange(0, BM, 1) / BM)
    col += BM

    # MR
    colorwheel[col : col + MR, 2] = 255 - np.floor(255 * np.arange(0, MR, 1) / MR)
    colorwheel[col : col + MR, 0] = 255

    return colorwheel / 255.0


def flow_to_color(
    flow: np.ndarray, max_flow: Optional[float] = None, convention: str = "middlebury"
) -> np.ndarray:
    """
    Convert optical flow to an RGB image using the standard color wheel.

    Args:
        flow: Optical flow array of shape (H, W, 2).

        max_flow: Normalization factor.
                  - If None: Defaults to the max magnitude in the flow array.
                  - If float: Any flow larger than this is clamped/desaturated.

        convention: Controls the default normalization behavior if max_flow is None.
                   - 'middlebury': Always normalizes to the current image max.
                   - 'kitti': Defaults to a fixed scale (e.g. 3.0) if max_flow is None.

    Returns:
        np.ndarray: RGB image of shape (H, W, 3).
                    Valid flow is colored (0 motion = White).
                    Invalid/NaN flow is Black.
    """

    assert flow.ndim == 3 and flow.shape[2] == 2

    u = flow[:, :, 0]
    v = flow[:, :, 1]

    # 1. Detect Invalid Flow (NaN, Inf, or > 1e9)
    # Middlebury uses 1e9 as a magic number for "unknown"
    idx_unknown = (np.abs(u) > 1e9) | (np.abs(v) > 1e9) | np.isnan(u) | np.isnan(v)

    # Temporarily clean data for calculation (avoid warnings)
    u = np.where(idx_unknown, 0, u)
    v = np.where(idx_unknown, 0, v)

    mag = np.sqrt(u**2 + v**2)
    angle = np.arctan2(-v, -u) / np.pi

    fk = (angle + 1) / 2 * (55 - 1)
    k0 = np.floor(fk).astype(int)
    k1 = k0 + 1
    k1[k1 == 55] = 0
    f = fk - k0

    colorwheel = _get_color_wheel()
    col0 = colorwheel[k0]
    col1 = colorwheel[k1]

    col = (1 - f)[:, :, None] * col0 + f[:, :, None] * col1

    if max_flow is None:
        if convention.lower() == "kitti":
            max_flow = 3.0
        else:
            max_flow = np.max(mag)
            if max_flow == 0:
                max_flow = 1.0

    col *= mag[:, :, None] / max_flow

    # Handle saturation
    idx_sat = mag > max_flow
    if np.any(idx_sat):
        col[idx_sat] = col[idx_sat] * (max_flow / mag[idx_sat])[:, None]
        col[idx_sat] = col[idx_sat] * 0.75 + 0.25

    col = np.clip(col, 0, 1)

    # 2. Apply Black for Unknown/Invalid pixels
    if np.any(idx_unknown):
        col[idx_unknown] = np.array([0, 0, 0])

    return (col * 255).astype(np.uint8)


def visualize_flow(
    flow: np.ndarray,
    ax: Optional[Axes] = None,
    title: str = "Optical Flow",
    max_flow: Optional[float] = None,
    convention: str = "middlebury",
    figsize: Tuple[int, int] = (8, 6),
) -> Tuple[Figure, Axes]:
    """
    Visualize dense optical flow.
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    rgb = flow_to_color(flow, max_flow=max_flow, convention=convention)

    ax.imshow(rgb)
    ax.set_title(title)
    ax.axis("off")

    return fig, ax


def visualize_flow_arrows(
    flow: np.ndarray,
    ax: Optional[Axes] = None,
    image: Optional[np.ndarray] = None,
    step: int = 16,
    scale: float = 1.0,
    color: str = "r",
    title: str = "Flow Vectors",
    figsize: Tuple[int, int] = (8, 6),
) -> Tuple[Figure, Axes]:
    """
    Visualize flow as sparse arrows (quiver plot).
    """
    if ax is None:
        fig, ax = plt.subplots(figsize=figsize)
    else:
        fig = ax.get_figure()

    h, w = flow.shape[:2]

    if image is not None:
        ax.imshow(image, cmap="gray" if image.ndim == 2 else None)
    else:
        ax.imshow(np.zeros((h, w, 3), dtype=np.uint8))

    y, x = np.mgrid[step // 2 : h : step, step // 2 : w : step]
    u = flow[y, x, 0]
    v = flow[y, x, 1]

    ax.quiver(
        x,
        y,
        u,
        -v,
        color=color,
        angles="xy",
        scale_units="xy",
        scale=1 / scale,
        width=0.003,
    )

    ax.set_title(title)
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.axis("off")

    return fig, ax


def compare_flows(
    flow_pred: np.ndarray,
    flow_gt: np.ndarray,
    axes: Optional[Union[np.ndarray, List[Axes]]] = None,
    max_flow: Optional[float] = None,
    convention: str = "middlebury",
    titles: Tuple[str, str, str] = ("Prediction", "Ground Truth", "EPE Error"),
    figsize: Tuple[int, int] = (16, 5),
) -> Tuple[Figure, np.ndarray]:
    """
    Compare prediction vs ground truth and visualize error.
    """
    if axes is None:
        fig, axes = plt.subplots(1, 3, figsize=figsize)
    else:
        fig = axes[0].get_figure()
        if len(axes) != 3:
            raise ValueError("compare_flows requires 3 axes")

    # 1. Visualize Prediction
    visualize_flow(
        flow_pred, ax=axes[0], title=titles[0], max_flow=max_flow, convention=convention
    )

    # 2. Visualize GT
    visualize_flow(
        flow_gt, ax=axes[1], title=titles[1], max_flow=max_flow, convention=convention
    )

    # 3. Visualize Endpoint Error (EPE)
    epe = np.linalg.norm(flow_pred - flow_gt, axis=2)

    # We use a valid mask to compute mean error only on valid pixels
    # Otherwise NaN errors will break the visualization
    mask = (np.abs(flow_gt[:, :, 0]) < 1e9) & (np.abs(flow_gt[:, :, 1]) < 1e9)
    if mask.sum() > 0:
        mean_epe = epe[mask].mean()
    else:
        mean_epe = 0.0

    im_err = axes[2].imshow(epe, cmap="magma")
    axes[2].set_title(f"{titles[2]} (Mean: {mean_epe:.2f})")
    axes[2].axis("off")

    plt.colorbar(im_err, ax=axes[2], fraction=0.046, pad=0.04)

    plt.tight_layout()
    return fig, axes


def concatenate_flows(
    arrays: List[np.ndarray],
    positions: List[Tuple[int, int]],
    kwargs_list: Optional[List[Dict[str, Any]]] = None,
) -> np.ndarray:
    """
    Concatenates strictly typed images (RGB) and flow fields (Float) into a grid.

    The function handles color space conversions internally so you can use standard
    RGB colors for text and input images, while leveraging OpenCV for text rendering.

    Args:
        arrays: List of numpy arrays.
            - 3 Channels: MUST be RGB uint8 (Standard Image).
            - 2 Channels: MUST be Float Optical Flow.
        positions: List of (col, row) tuples determining grid placement.
        kwargs_list: Optional list of dictionaries (one per array) for customization.
            Supported keys:

            [Visual Params]
            - 'caption': str         -> Text to write on the image.
            - 'font_scale': float    -> Size of text (default: 1.0).
            - 'font_thickness': int  -> Thickness of text (default: 2).
            - 'font_color': tuple    -> RGB tuple (default: (255, 255, 255)).
            - 'text_pos': tuple      -> (x, y) bottom-left position (default: (20, 40)).

            [Flow Params (Only used if array has 2 channels)]
            - 'max_flow': float      -> Normalization max magnitude.
            - 'convention': str      -> 'middlebury' or 'kitti'.

    Returns:
        np.ndarray: RGB image (H_grid, W_grid, 3) ready for Matplotlib/TensorBoard.
    """
    import cv2

    H, W = arrays[0].shape[:2]
    N = len(arrays)

    if kwargs_list is None:
        kwargs_list = [{}] * N

    # Canvas Size calculation
    max_u = max(p[0] for p in positions)
    max_v = max(p[1] for p in positions)

    # Create internal BGR canvas for OpenCV operations
    frame_bgr = np.zeros(((max_v + 1) * H, (max_u + 1) * W, 3), dtype=np.uint8)

    for arr, (u, v), kw in zip(arrays, positions, kwargs_list):
        # --- 1. Process Content (Strict Types -> BGR) ---
        if arr.shape[2] == 2:
            # TYPE: FLOW (Float) -> RGB -> BGR
            flow_args = {k: v for k, v in kw.items() if k in ["max_flow", "convention"]}
            rgb_flow = flow_to_color(arr, **flow_args)
            img_chunk_bgr = cv2.cvtColor(rgb_flow, cv2.COLOR_RGB2BGR)

        elif arr.shape[2] == 3:
            # TYPE: IMAGE (RGB uint8) -> BGR
            img_chunk_bgr = cv2.cvtColor(arr, cv2.COLOR_RGB2BGR)

        else:
            raise ValueError(
                f"Array has {arr.shape[2]} channels. Expected 2 (Flow) or 3 (RGB Image)."
            )

        # --- 2. Add Text (OpenCV draws on BGR) ---
        caption = kw.get("caption", None)
        if caption:
            # User specifies color in RGB (e.g., Red=(255,0,0))
            # We flip to BGR because we are drawing on a BGR image
            font_color_rgb = kw.get("font_color", (255, 255, 255))
            font_color_bgr = font_color_rgb[::-1]

            cv2.putText(
                img_chunk_bgr,
                str(caption),
                kw.get("text_pos", (20, 40)),
                fontFace=cv2.FONT_HERSHEY_SIMPLEX,
                fontScale=kw.get("font_scale", 1.0),
                color=font_color_bgr,
                thickness=kw.get("font_thickness", 2),
                lineType=cv2.LINE_AA,
            )

        # --- 3. Place in Grid ---
        y, x = v * H, u * W
        frame_bgr[y : y + H, x : x + W] = img_chunk_bgr

    # Convert final result back to RGB for return
    return cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
