import torch
import torch.nn as nn
from .model_blocks import ConvBlock

class UNet(nn.Module):
    def __init__(self, in_channels, out_channels, feature_sizes = [64, 128, 256, 512], dropout=0.1):
        super(UNet, self).__init__()
        self.downs = nn.ModuleList()
        self.ups = nn.ModuleList()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2)
        
        for feature_size in feature_sizes:
            self.downs.append(ConvBlock(in_channels, feature_size))
            in_channels = feature_size
        self.bottleneck = nn.Sequential(
            ConvBlock(feature_sizes[-1], feature_sizes[-1] * 2),
            nn.Dropout2d(p=dropout),
        )
        for feature_size in reversed(feature_sizes):
            self.ups.append(nn.ConvTranspose2d(feature_size * 2, feature_size, kernel_size=2, stride=2, padding=0))
            self.ups.append(ConvBlock(feature_size * 2, feature_size))

        self.final_conv = nn.Sequential(
            nn.Conv2d(feature_sizes[0], out_channels, kernel_size=1),
            nn.Sigmoid()
        )
        self._initialize_weights()
    
    def forward(self, x):
        skips = []
        for down in self.downs:
            x = down(x)
            skips.append(x)
            x = self.pool(x)

        x = self.bottleneck(x)
        skips = skips[::-1]

        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2] # 透過 // 2 取得對應的 skip connection
            x = torch.cat([x, skip], dim=1) # 進行 skip connection
            x = self.ups[i + 1](x) # 進行 convolution
        return self.final_conv(x) # 輸出預測結果
    
    def _initialize_weights(self):
        for module in self.modules():
            if isinstance(module, nn.Conv2d):
                # 使用 Kaiming initialization
                nn.init.kaiming_normal_(module.weight, mode='fan_out', nonlinearity='relu')
                if module.bias is not None: 
                    nn.init.constant_(module.bias, 0)
            elif isinstance(module, nn.BatchNorm2d):
                nn.init.constant_(module.weight, 1)
                nn.init.constant_(module.bias, 0)

    
if __name__ == '__main__':
    model = UNet(in_channels=3, out_channels=1)
    x = torch.randn(2, 3, 512, 512)
    print(model(x).shape)
    print(model)
    
