import numpy as np
def dice_score(pred_mask, gt_mask):
    # implement the Dice score here
    intersection = np.sum(pred_mask * gt_mask)
    union = np.sum(pred_mask) + np.sum(gt_mask)
    dice_score = 2 * intersection / union
    return dice_score

