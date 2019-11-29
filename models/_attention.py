from torch import nn
import torch

class SELayerTHW(nn.Module):
    # Excite channels
    # Altered channel//reduction for 1 when time (max dim=16) is squeeze
    def __init__(self, channel):
        super(SELayerTHW, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, 1, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(1, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _  = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class SELayerCHW(nn.Module):
    # Excite time
    # Altered reduction to 0 when channels squeeze (min dim=18)
    def __init__(self, channel, reduction=9):
        super(SELayerCHW, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, t, h, w = x.size()
        # x.view(b,t,c,h,w) or permute? I think view but not sure
        y = self.avg_pool(x.view(b,t,c,h,w)).view(b, t)
        y = self.fc(y).view(b, 1, t, 1, 1)
        return x * y.expand_as(x)


if __name__ == '__main__':
    x = torch.randn((2, 18, 16, 112, 112)).cuda()

    c3p0 = SELayerTHW(x.size()[1]).cuda()
    out = c3p0(x)
    print(out.size())

    x = torch.randn((2, 18, 16, 112, 112)).cuda()

    c3p0 = SELayerCHW(x.size()[2]).cuda()
    out = c3p0(x)
    print(out.size())