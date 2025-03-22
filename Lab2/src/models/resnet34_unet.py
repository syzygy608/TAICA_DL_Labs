from .model_blocks import ResidualBlock, ConvBlock
import torch.nn as nn
import torch

class ResNetEncoder(nn.Module):
    def __init__(self, in_channels=3, layers=[3, 4, 6, 3], feature_sizes=[64, 128, 256, 512]):
        # ResNet34 的 block 數量分別為 [3, 4, 6, 3]
        super(ResNetEncoder, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(in_channels, 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
        )
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.layers = nn.ModuleList()

        # 64 to 64
        self.layers.append(self._make_layer(ResidualBlock, feature_sizes[0], feature_sizes[0], layers[0]))
        # 64 to 128
        self.layers.append(self._make_layer(ResidualBlock, feature_sizes[0], feature_sizes[1], layers[1]))
        # 128 to 256
        self.layers.append(self._make_layer(ResidualBlock, feature_sizes[1], feature_sizes[2], layers[2]))
        # 256 to 512
        self.layers.append(self._make_layer(ResidualBlock, feature_sizes[2], feature_sizes[3], layers[3]))

       
    def _make_layer(self, block, in_channels, out_channels, counts):
        layer = [block(in_channels, out_channels, downsample=True if in_channels != out_channels else None)]
        for _ in range(1, counts):
            layer.append(block(out_channels, out_channels))
        return nn.Sequential(*layer)


    def forward(self, x):
        x1 = self.conv1(x)
        x1 = self.maxpool(x1)
        skips = []
        for layer in self.layers:
            x1 = layer(x1)
            skips.append(x1)
        return x1, skips

class UNetDecoder(nn.Module):
    def __init__(self, feature_sizes = [512, 256, 128, 64], out_channels=1, dropout=0.1):
        super(UNetDecoder, self).__init__()
        
        # same as UNet upsample path
        self.bottleneck = nn.Sequential(
            nn.ConvTranspose2d(feature_size * 2, feature_size, kernel_size=2, stride=2, padding=0),
            ConvBlock(feature_sizes[0], feature_sizes[0] * 2),
            nn.Dropout2d(p=dropout),
        )
        self.ups = nn.ModuleList()
        for feature_size in feature_sizes:
            self.ups.append(nn.ConvTranspose2d(feature_size * 2, feature_size, kernel_size=2, stride=2, padding=0))
            self.ups.append(ConvBlock(feature_size * 2, feature_size))

        self.final_conv = nn.Sequential(
            nn.Conv2d(feature_sizes[-1], out_channels, kernel_size=1),
            nn.Sigmoid()
        )

    def forward(self, x, skips):
        x = self.bottleneck(x)
        skips = skips[::-1]
        for i in range(0, len(self.ups), 2):
            x = self.ups[i](x)
            skip = skips[i // 2] # 透過 // 2 取得對應的 skip connection
            x = torch.cat([x, skip], dim=1) # 進行 skip connection
            x = self.ups[i + 1](x) # 進行 convolution
        return self.final_conv(x) # 輸出預測結果
    
    
class ResNet34UNet(nn.Module):
    def __init__(self, in_channels=3, out_channels=1):
        super(ResNet34UNet, self).__init__()
        self.encoder = ResNetEncoder(in_channels=in_channels)
        self.decoder = UNetDecoder(out_channels=out_channels)
        self._initialize_weights()

    def forward(self, x):
        x, skips = self.encoder(x)
        return self.decoder(x, skips)
    
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
    model = ResNet34UNet()
    x = torch.randn(2, 3, 256, 256)
    print(model(x).shape)