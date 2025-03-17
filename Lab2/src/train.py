import argparse
from oxford_pet import load_dataset
from torch.utils.data import DataLoader
from torch.utils.tensorboard import SummaryWriter
from models import UNet, ResNetUNet
import tqdm
from utils import dice_score
from evaluate import evaluate
import torch
import numpy as np
import torch.nn as nn

def train(args):
    # 取出訓練集和驗證集
    train_data = load_dataset(args.data_path, 'train')
    train_data_loader = DataLoader(train_data, batch_size=args.batch_size, shuffle=True)
    val_data = load_dataset(args.data_path, 'val')
    val_data_loader = DataLoader(val_data, batch_size=args.batch_size, shuffle=False)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # 選擇模型
    if args.model == 'unet':
        model = UNet(in_channels=3, out_channels=1).to(device)
    elif args.model == 'resnet':
        model = ResNetUNet(in_channels=3, out_channels=1).to(device)
    
    # 選擇優化器和損失函數
    optimizer = torch.optim.Adam(model.parameters(), lr=args.learning_rate, weight_decay=args.weight_decay)
    # 在優化器時新增 L2 正則化項

    # 使用 MSE Loss 作為損失函數
    criterion = nn.MSELoss()

    writer = SummaryWriter()
    
    best_score = 0
    best_model_path = args.model + '_best_model.pth'

    for epoch in range(args.epochs):
        train_loss = []
        train_score = []
        model.train()
        progress_bar = tqdm.tqdm(enumerate(train_data_loader), total=len(train_data_loader))
        for i, batch in progress_bar:
            images = batch['image'].to(device)
            masks = batch['mask'].to(device)

            predictions = model(images) # 取得模型預測結果
            loss = criterion(predictions, masks) # 計算損失
            train_loss.append(loss.item()) # 將損失值加入 train_loss 中
            optimizer.zero_grad() # 梯度歸零
            loss.backward() # 反向傳播
            optimizer.step() # 更新權重
            with torch.no_grad():
                score = dice_score(predictions, masks)
                train_score.append(score)
            progress_bar.set_description(f'Epoch {epoch+1}/{args.epochs}, Loss: {loss.item():.4f}, Dice Score: {score:.4f}')

        writer.add_scalar('Loss/train', np.mean(train_loss), epoch)
        writer.add_scalar('Dice/train', np.mean(train_score), epoch)

        validation_loss, validation_score = evaluate(model, val_data_loader, device)
        writer.add_scalar('Loss/val', validation_loss, epoch)
        writer.add_scalar('Dice/val', validation_score, epoch)
        if validation_score > best_score: # 如果驗證集的分數比最佳分數還要高
            best_score = validation_score
            torch.save(model.state_dict(), best_model_path) # 儲存模型權重
    writer.close()

def get_args():
    parser = argparse.ArgumentParser(description='Train the UNet on images and target masks')
    parser.add_argument('--data_path', type=str, help='path of the input data')
    parser.add_argument('--epochs', '-e', type=int, default=5, help='number of epochs')
    parser.add_argument('--batch_size', '-b', type=int, default=1, help='batch size')
    parser.add_argument('--learning-rate', '-lr', type=float, default=1e-5, help='learning rate')
    parser.add_argument('--model', '-m', type=str, choices=['unet', 'resnet'], default='unet', help='model to use')
    parser.add_argument('--weight-decay', '-wd', type=float, default=1e-4, help='weight decay')

    return parser.parse_args()
 
if __name__ == "__main__":
    args = get_args()
    train(args)