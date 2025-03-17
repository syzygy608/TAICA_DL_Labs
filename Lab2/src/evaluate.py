from torch import nn
from utils import dice_score
import torch
import numpy as np
from utils import dice_loss

def evaluate(net, data, device):
    validation_loss = []
    validation_dice = []

    criterion = nn.BCEWithLogitsLoss()
    with torch.no_grad():
        net.eval()
        for batch in data:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            predictions = net(images)
            loss = criterion(predictions, masks) + dice_loss(predictions, masks)
            validation_loss.append(loss.item())

            score = dice_score(predictions, masks)
            validation_dice.append(score.item())
        print(f'Validation Loss: {np.mean(validation_loss):.4f}, Dice Score: {np.mean(validation_dice):.4f}')
    return np.mean(validation_loss), np.mean(validation_dice)