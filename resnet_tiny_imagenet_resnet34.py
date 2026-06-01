import torch
import torch.nn as nn
import torch.nn.functional as F

class ResidualBlock(nn.Module):
    def __init__(self, inchannel, outchannel, stride=1):
        super(ResidualBlock, self).__init__()
        self.left = nn.Sequential(
            nn.Conv2d(inchannel, outchannel, kernel_size=3, stride=stride, padding=1, bias=False),
            nn.BatchNorm2d(outchannel),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel, outchannel, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(outchannel)
        )
        self.shortcut = nn.Sequential()
        if stride != 1 or inchannel != outchannel:
            self.shortcut = nn.Sequential(
                nn.Conv2d(inchannel, outchannel, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(outchannel)
            )

    def forward(self, x):
        out = self.left(x)
        out += self.shortcut(x)
        preact = out
        out = F.relu(out)
        return out, preact

class ResNet(nn.Module):
    def __init__(self, ResidualBlock, num_classes=200):
        super(ResNet, self).__init__()
        self.inchannel = 64
        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=1, padding=1, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(),
        )
        self.layer1 = self.make_layer(ResidualBlock, 64,  3, stride=1)
        self.layer2 = self.make_layer(ResidualBlock, 128, 4, stride=2)
        self.layer3 = self.make_layer(ResidualBlock, 256, 6, stride=2)
        self.layer4 = self.make_layer(ResidualBlock, 512, 3, stride=2)
        self.avg_pool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512, num_classes)

    def make_layer(self, block, channels, num_blocks, stride):
        strides = [stride] + [1] * (num_blocks - 1)   #strides=[1,1]
        layers = []
        for stride in strides:
            layers.append(block(self.inchannel, channels, stride))
            self.inchannel = channels
        return nn.ModuleList(layers)

    def forward(self, x, is_feat=False, preact=False):
        out = self.conv1(x)
        f0 = out

        f1_pre = None
        for block in self.layer1:
            out, f1_pre = block(out)
        f1 = out

        # Layer 2
        f2_pre = None
        for block in self.layer2:
            out, f2_pre = block(out)
        f2 = out

        # Layer 3
        f3_pre = None
        for block in self.layer3:
            out, f3_pre = block(out)
        f3 = out

        # Layer 4
        f4_pre = None
        for block in self.layer4:
            out, f4_pre = block(out)
        f4 = out


        out = self.avg_pool(out)
        out = out.view(out.size(0), -1)
        f5 = out
        out = self.fc(out)

        if is_feat:
            if preact:
                return [f0, f1_pre, f2_pre, f3_pre, f4_pre, f5], out
            else:
                return [f0, f1, f2, f3, f4, f5], out
        else:
            return out


def tiny_imagenet_resnet34(**kwargs):

    return ResNet(ResidualBlock)