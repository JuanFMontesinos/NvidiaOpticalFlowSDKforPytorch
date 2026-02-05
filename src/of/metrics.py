"""
Error metrics for optical flow evaluation.
"""

import numpy as np
from typing import Optional, Dict, Tuple


def endpoint_error(flow_pred: np.ndarray, flow_gt: np.ndarray, 
                   valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute End-Point Error (EPE) between predicted and ground truth flow.
    
    EPE = sqrt((u_pred - u_gt)^2 + (v_pred - v_gt)^2)
    
    Args:
        flow_pred: Predicted flow of shape (H, W, 2)
        flow_gt: Ground truth flow of shape (H, W, 2)
        valid_mask: Optional boolean mask of valid pixels (H, W)
        
    Returns:
        EPE map of shape (H, W)
    """
    diff = flow_pred - flow_gt
    epe = np.sqrt(np.sum(diff**2, axis=2))
    
    if valid_mask is not None:
        epe = epe * valid_mask
    
    return epe


def average_endpoint_error(flow_pred: np.ndarray, flow_gt: np.ndarray,
                           valid_mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Average End-Point Error (AEE).
    
    Args:
        flow_pred: Predicted flow of shape (H, W, 2)
        flow_gt: Ground truth flow of shape (H, W, 2)
        valid_mask: Optional boolean mask of valid pixels (H, W)
        
    Returns:
        Average EPE as a scalar float
    """
    epe = endpoint_error(flow_pred, flow_gt, valid_mask)
    
    if valid_mask is not None:
        return epe[valid_mask].mean()
    else:
        return epe.mean()


def angular_error(flow_pred: np.ndarray, flow_gt: np.ndarray,
                  valid_mask: Optional[np.ndarray] = None) -> np.ndarray:
    """
    Compute Angular Error (AE) between predicted and ground truth flow.
    
    The angular error is computed as the angle between the 3D vectors
    (u, v, 1) for predicted and ground truth flows.
    
    Args:
        flow_pred: Predicted flow of shape (H, W, 2)
        flow_gt: Ground truth flow of shape (H, W, 2)
        valid_mask: Optional boolean mask of valid pixels (H, W)
        
    Returns:
        Angular error map in degrees of shape (H, W)
    """
    # Convert to 3D vectors
    u_pred, v_pred = flow_pred[:, :, 0], flow_pred[:, :, 1]
    u_gt, v_gt = flow_gt[:, :, 0], flow_gt[:, :, 1]
    
    # Compute dot product of normalized vectors
    num = 1 + u_pred * u_gt + v_pred * v_gt
    denom = np.sqrt(1 + u_pred**2 + v_pred**2) * np.sqrt(1 + u_gt**2 + v_gt**2)
    
    # Avoid division by zero
    denom = np.maximum(denom, 1e-10)
    
    # Compute angle
    cos_angle = np.clip(num / denom, -1.0, 1.0)
    ae = np.arccos(cos_angle) * 180.0 / np.pi
    
    if valid_mask is not None:
        ae = ae * valid_mask
    
    return ae


def average_angular_error(flow_pred: np.ndarray, flow_gt: np.ndarray,
                         valid_mask: Optional[np.ndarray] = None) -> float:
    """
    Compute Average Angular Error (AAE).
    
    Args:
        flow_pred: Predicted flow of shape (H, W, 2)
        flow_gt: Ground truth flow of shape (H, W, 2)
        valid_mask: Optional boolean mask of valid pixels (H, W)
        
    Returns:
        Average angular error in degrees as a scalar float
    """
    ae = angular_error(flow_pred, flow_gt, valid_mask)
    
    if valid_mask is not None:
        return ae[valid_mask].mean()
    else:
        return ae.mean()


def fl_all(flow_pred: np.ndarray, flow_gt: np.ndarray,
           valid_mask: Optional[np.ndarray] = None,
           threshold: float = 3.0) -> float:
    """
    Compute FL-all metric (percentage of outliers).
    
    An outlier is defined as EPE > threshold OR angular error > threshold degrees.
    
    Args:
        flow_pred: Predicted flow of shape (H, W, 2)
        flow_gt: Ground truth flow of shape (H, W, 2)
        valid_mask: Optional boolean mask of valid pixels (H, W)
        threshold: Threshold for outlier detection (default: 3.0 pixels)
        
    Returns:
        Percentage of outliers (0-100)
    """
    epe = endpoint_error(flow_pred, flow_gt, valid_mask)
    ae = angular_error(flow_pred, flow_gt, valid_mask)
    
    outliers = (epe > threshold) | (ae > threshold)
    
    if valid_mask is not None:
        return 100.0 * outliers[valid_mask].sum() / valid_mask.sum()
    else:
        return 100.0 * outliers.sum() / outliers.size


def compute_all_metrics(flow_pred: np.ndarray, flow_gt: np.ndarray,
                       valid_mask: Optional[np.ndarray] = None,
                       thresholds: Tuple[float, ...] = (1.0, 3.0, 5.0)) -> Dict[str, float]:
    """
    Compute all standard optical flow metrics.
    
    Args:
        flow_pred: Predicted flow of shape (H, W, 2)
        flow_gt: Ground truth flow of shape (H, W, 2)
        valid_mask: Optional boolean mask of valid pixels (H, W)
        thresholds: Thresholds for outlier metrics
        
    Returns:
        Dictionary containing all metrics
    """
    metrics = {}
    
    # End-point error
    metrics['EPE'] = average_endpoint_error(flow_pred, flow_gt, valid_mask)
    
    # Angular error
    metrics['AE'] = average_angular_error(flow_pred, flow_gt, valid_mask)
    
    # Outlier percentages at different thresholds
    for threshold in thresholds:
        metrics[f'FL-{threshold}px'] = fl_all(flow_pred, flow_gt, valid_mask, threshold)
    
    # Root mean square error
    epe = endpoint_error(flow_pred, flow_gt, valid_mask)
    if valid_mask is not None:
        metrics['RMSE'] = np.sqrt((epe[valid_mask]**2).mean())
    else:
        metrics['RMSE'] = np.sqrt((epe**2).mean())
    
    return metrics