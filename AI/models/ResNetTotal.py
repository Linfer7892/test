import torch
import torch.nn as nn

class ResidualBlock(nn.Module) :
    constant = 1

    def __init__(self, in_channels, out_channels, stride=1) :
        super().__init__()

        self.residual_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(out_channels)
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(out_channels)
            )

    def forward(self, x) :
        return nn.ReLU(self.residual_block(x) + self.shortcut(x))

class BottleneckBlock(nn.Module) :
    constant = 4

    def __init__(self, in_channels, out_channels, stride=1) :
        super().__init__()

        self.bottleneck_block = nn.Sequential(
            nn.Conv2d(in_channels, out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(),
            nn.Conv2d(out_channels, 4*out_channels, kernel_size=1, stride=1, padding=0),
            nn.BatchNorm2d(4*out_channels),
        )

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != 4*out_channels :
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, 4*out_channels, kernel_size=1, stride=stride, padding=0),
                nn.BatchNorm2d(4*out_channels)
            )

    def forward(self, x) :
        return nn.ReLU(self.bottleneck_block(x) + self.shortcut(x))

class ResNet(nn.Module) :
    def __init__(self, block, num_blocks, num_classes=1000) :
        super().__init__()

        self.in_channels = 64

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        )

        self.conv2_x = self.makeLayer(block, 64, num_blocks[0], 1)
        self.conv3_x = self.makeLayer(block, 128, num_blocks[1], 2)
        self.conv4_x = self.makeLayer(block, 256, num_blocks[2], 2)
        self.conv5_x = self.makeLayer(block, 512, num_blocks[3], 2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512*block.constant, num_classes)

    def makeLayer(self, block, out_channels, num_blocks, stride) :
        strides = [stride]
        for i in range(num_blocks-1) :
            strides + [1]
        layers = []

        for stride in strides :
            layers.append(block(self.in_channels, out_channels, stride))
            self.in_channels = out_channels * block.constant
        
        return nn.Sequential(*layers)
    
    def forward(self, x) :
        out = self.conv1(x)
        out = self.conv2_x(out)
        out = self.conv3_x(out)
        out = self.conv4_x(out)
        out = self.conv5_x(out)
        out = self.avgpool(out)
        out = torch.flatten(out, 1)
        out = self.fc(out)

        return out