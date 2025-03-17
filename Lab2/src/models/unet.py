import torch
import torch.nn as nn
from .model_blocks import ConvBlock

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, feature_sizes = [64, 128, 256, 512, 1024]):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        for feature_size in feature_sizes:
            self.downs.append(ConvBlock(in_channels, feature_size))
            in_channels = feature_size
        self.bottleneck = ConvBlock(feature_sizes[-1], feature_sizes[-1] * 2)
        for feature_size in reversed(feature_sizes):
            self.ups.append(nn.ConvTranspose2d(feature_size * 2, feature_size, kernel_size=2, stride=2))
            self.ups.append(ConvBlock(feature_size * 2, feature_size))
        self.final_conv = nn.Conv2d(feature_sizes[0], out_channels, kernel_size=1)

    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = nn.MaxPool2d(kernel_size=2)(x)
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2] # 透過 // 2 取得對應的 skip connection
            if x.shape != skip.shape: # 如果 x 和 skip 的形狀不同，進行 bilinear interpolation
                x = nn.functional.interpolate(x, size=skip.shape[2:], mode='bilinear', align_corners=False)
            x = torch.cat([x, skip], dim=1) # 進行 skip connection
            x = self.ups[i + 1](x) # 進行 convolution
        return self.final_conv(x) # 輸出預測結果
    
if __name__ == '__main__':
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 1024, 1024)
    print(model(x).shape)
    print(model)
    
