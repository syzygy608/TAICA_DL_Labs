from .model_blocks import ResidualBlock, ConvBlock
import torch.nn as nn

class ResNet(nn.Module):
    def __init__(self, channels = [64, 128, 256, 512], num_classes=3):
        super(ResNet, self).__init__()
        
        # initial layers before resnet
        self.init = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )
        self.layers = nn.ModuleList()


       
        
    def forward(self, x):
        
        return x
