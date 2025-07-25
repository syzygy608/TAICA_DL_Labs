import argparse
import torch
from oxford_pet import load_dataset
import numpy as np
from models.unet import UNet
from models.resnet34_unet import ResNet34UNet
from utils import dice_score
import tqdm

def inference(args):
    if args.model == "unet":
        model = UNet(in_channels=3, out_channels=1)
    else:
        model = ResNet34UNet(in_channels=3, out_channels=1)
    
    state_dict = torch.load("../saved_models/" + args.model + '_best_model.pth', weights_only=True)
    model.load_state_dict(state_dict)  # 將參數載入模型
    model.eval()
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    data = load_dataset(args.data_path, 'test')
    data_loader = torch.utils.data.DataLoader(data, batch_size=args.batch_size, shuffle=False)

    dice_scores = []
    progress_bar = tqdm.tqdm(enumerate(data_loader), total=len(data_loader))
    for i, batch in progress_bar:
        images = batch['image'].to(device)
        masks = batch['mask'].to(device)
        predictions = model(images) # 取得模型預測結果
        dice = dice_score(predictions, masks)
        dice_scores.append(dice.item())
    print(f'Model: {args.model}')
    print(f'Dice Score: {np.mean(dice_scores):.4f}')


def get_args():
    parser = argparse.ArgumentParser(description='Predict masks from input images')
    parser.add_argument('--model', '-m', default="unet", help="name of model")
    parser.add_argument('--data_path', default="../dataset/oxford-iiit-pet", type=str, help='path to the input data')
    parser.add_argument('--batch_size', '-b', type=int, default=8, help='batch size')
    
    return parser.parse_args()

if __name__ == '__main__':
    args = get_args()
    inference(args)
    