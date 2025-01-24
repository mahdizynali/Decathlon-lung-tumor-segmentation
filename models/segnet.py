import torch
import torch.nn as nn
import torch.nn.functional as F

class BaseSegNet(nn.Module):
    def __init__(self, n_classes=2, vgg_level=3):
        super(BaseSegNet, self).__init__()

        self.encoder1 = self._block(3, 64)
        self.encoder2 = self._block(64, 128)
        self.encoder3 = self._block(128, 256, num_convs=3)
        self.encoder4 = self._block(256, 512, num_convs=3)
        self.encoder5 = self._block(512, 512, num_convs=3)

        self.features = [self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5]

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        
        self.vgg_level = vgg_level

    def _block(self, in_channels, out_channels, num_convs=2):
        layers = []
        for _ in range(num_convs):
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
            layers.append(nn.BatchNorm2d(out_channels))
            layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.vgg_level:
                features.append(x)
        
        x = features[0]
        
        x = self.decoder(x)
        x = self.final_conv(x)
        
        return x













import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channels)
        
        self.shortcut = nn.Sequential()
        if in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm2d(out_channels)
            )
        
    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
        self.fc1 = nn.Conv2d(in_channels, in_channels // reduction, kernel_size=1)
        self.relu = nn.ReLU(inplace=True)
        self.fc2 = nn.Conv2d(in_channels // reduction, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        avg_pool = self.avg_pool(x)
        max_pool = self.max_pool(x)
        avg_out = self.fc2(self.relu(self.fc1(avg_pool)))
        max_out = self.fc2(self.relu(self.fc1(max_pool)))
        out = avg_out + max_out
        out = self.sigmoid(out)
        return x * out

class SuperSegNet(nn.Module):
    def __init__(self, n_classes=2, vgg_level=3):
        super(SuperSegNet, self).__init__()

        self.encoder1 = self._block(3, 64, use_residual=True)
        self.encoder2 = self._block(64, 128, use_residual=True)
        self.encoder3 = self._block(128, 256, num_convs=3, use_residual=True)
        self.encoder4 = self._block(256, 512, num_convs=3, use_residual=True)
        self.encoder5 = self._block(512, 512, num_convs=3, use_residual=True)

        self.features = [self.encoder1, self.encoder2, self.encoder3, self.encoder4, self.encoder5]

        self.attention = ChannelAttention(512)

        self.decoder = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1),
            nn.BatchNorm2d(512),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(512, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(256, 128, kernel_size=3, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),

            nn.Conv2d(128, 64, kernel_size=3, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.Upsample(scale_factor=2, mode='bilinear', align_corners=False),
        )

        self.final_conv = nn.Conv2d(64, n_classes, kernel_size=3, padding=1)
        
        self.vgg_level = vgg_level

    def _block(self, in_channels, out_channels, num_convs=2, use_residual=False):
        layers = []
        for _ in range(num_convs):
            if use_residual:
                layers.append(ResidualBlock(in_channels, out_channels))
            else:
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1))
                layers.append(nn.BatchNorm2d(out_channels))
                layers.append(nn.ReLU(inplace=True))
            in_channels = out_channels
        layers.append(nn.MaxPool2d(kernel_size=2, stride=2))
        return nn.Sequential(*layers)

    def forward(self, x):
        features = []

        for i, layer in enumerate(self.features):
            x = layer(x)
            if i == self.vgg_level:
                features.append(x)

        x = features[0]

        x = self.attention(x)

        x = self.decoder(x)
        x = self.final_conv(x)

        return x
