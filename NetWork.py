import torch
from torch import nn


class Block(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(Block, self).__init__()
        self.block = nn.Sequential(
            nn.Conv2d(in_channels=in_channel, out_channels=out_channel, kernel_size=3, stride=1, padding=1),
            nn.BatchNorm2d(num_features=out_channel),
            nn.LeakyReLU(),
        )

    def forward(self, x):
        return self.block(x)


class DownSample(nn.Module):
    def __init__(self, in_channel, out_channel):
        super(DownSample, self).__init__()
        # 下采样，尺寸缩小一半
        self.down = nn.Sequential(Block(in_channel=in_channel, out_channel=in_channel * 3),
                                  Block(in_channel=in_channel * 3, out_channel=out_channel),
                                  nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True))

    def forward(self, x):
        y = self.down(x)
        return y


class ScaleNet(nn.Module):
    def __init__(self):
        super(ScaleNet, self).__init__()
        self.base_block_list = []
        self.beside_block_list = []
        self.channel_list = [16, 64, 128,64, 16]
        num_down = len(self.channel_list) - 1
        self.first_block = Block(in_channel=3, out_channel=self.channel_list[0])
        self.last_block = nn.Sequential(Block(in_channel=self.channel_list[-1], out_channel=3),
                                        nn.Upsample(scale_factor=2))
        self.up_sample = nn.Upsample(scale_factor=2)
        for num in range(num_down):
            self.base_block_list.append(
                Block(in_channel=self.channel_list[num], out_channel=self.channel_list[num + 1]))
            self.beside_block_list.append(
                DownSample(in_channel=self.channel_list[num], out_channel=self.channel_list[num + 1]))
        self.base_block_list = nn.ModuleList(self.base_block_list)
        self.beside_block_list = nn.ModuleList(self.beside_block_list)

    def forward(self, x):
        y1 = self.first_block(x)
        y2 = y1
        for num in range(len(self.base_block_list)):
            base_block = self.base_block_list[num]
            beside_block = self.beside_block_list[num]
            y1 = base_block(y1)
            y2 = beside_block(y2)
            y3 = self.up_sample(y2)
            for i in range(num):
                y3 = self.up_sample(y3)
            y1 = y1 + y3
        y = self.last_block(y1)
        return y


def t():
    x = torch.ones(size=(3, 3, 512, 512))
    net = ScaleNet()
    print(net)
    y = net(x)
    print(y.shape)


if __name__ == '__main__':
    t()
