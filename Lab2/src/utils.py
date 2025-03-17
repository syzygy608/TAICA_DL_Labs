import torch
import numpy as np

def dice_score(pred_mask, gt_mask):
    """
    Calculate Dice score between predicted mask and ground truth mask.
    
    Args:
        pred_mask (torch.Tensor): Predicted mask (binary or probabilities)
        gt_mask (torch.Tensor): Ground truth mask (binary)
    
    Returns:
        torch.Tensor: Dice score as a scalar tensor
    """
    # 確保輸入是 Tensor
    if not isinstance(pred_mask, torch.Tensor):
        pred_mask = torch.tensor(pred_mask, dtype=torch.float32)
    if not isinstance(gt_mask, torch.Tensor):
        gt_mask = torch.tensor(gt_mask, dtype=torch.float32)

    # 計算交集 (intersection)
    intersection = torch.sum(pred_mask * gt_mask)
    
    # 計算聯集 (union)
    union = torch.sum(pred_mask) + torch.sum(gt_mask)
    eps = 1e-8
    # 計算 Dice 分數
    dice_score = 2 * intersection / (union + eps)
    
    # 轉成 numpy array
    dice_score = dice_score.cpu().numpy()
    return dice_score