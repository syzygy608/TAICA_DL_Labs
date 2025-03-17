import argparse
import torch
from oxford_pet import OxfordPetDataset
import numpy as np
from models.unet import UNet

def inference(args):
    if args.model_name == "unet":
        model = UNet(3, 1)
    state_dict = torch.load(args.model)
    model.load_state_dict(state_dict)  # 將參數載入模型
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = OxfordPetDataset(args.data_path, mode="test") # 讀取測試集
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False)

    dice_score = []
    for batch in data_loader:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        predictions = model(images) # 取得模型預測結果
        dice = dice_score(predictions, masks)
        dice_score.append(dice)
    print(f'Model: {args.model}')
    print(f'Dice Score: {np.mean(dice_score):.4f}')


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model_name', default="unet", help="name of model")
    parser.add_argument('--model', default='../saved_models/unet_best_model.pth', help='path to the stored model weoght')
    parser.add_argument('--data_path', type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    inference(args)
    