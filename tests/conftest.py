import os
import pytest
import numpy as np
import SimpleITK as sitk
import torch
from pathlib import Path

@pytest.fixture
def test_data_dir():
    """Return the directory with test data."""
    return Path(__file__) / "test_data"

def dice_loss(pred: torch.Tensor, target: torch.Tensor, smooth: float = 1e-5, reduce: bool = True) -> torch.Tensor:
    """Compute Dice loss between prediction and target tensors.
    
    Args:
        pred (torch.Tensor): Predicted tensor of shape (B, C, ...)
        target (torch.Tensor): Target tensor of shape (B, C, ...)
        smooth (float, optional): Smoothing factor to avoid division by zero. Default: 1e-5
        
    Returns:
        torch.Tensor: Average Dice loss value across batches and channels
    """
    # Flatten all dimensions after channel
    pred = pred.reshape(pred.size(0), pred.size(1), -1)  # (B, C, N)
    target = target.reshape(target.size(0), target.size(1), -1)  # (B, C, N)
    
    # Compute intersection and union
    intersection = torch.sum(pred * target, dim=2)  # (B, C)
    union = torch.sum(pred, dim=2) + torch.sum(target, dim=2)  # (B, C)
    
    # Compute Dice score
    dice = (2. * intersection + smooth) / (union + smooth)  # (B, C)
    
    # Return 1 - mean Dice (as a loss)
    # return 1 - dice.mean()
    return (1 - dice).mean() if reduce else (1 - dice)
