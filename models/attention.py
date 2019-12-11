from torch import nn
import torch

class SELayerTHW(nn.Module):
    # Excite channels
    # With reduction 9 smallest bottleneck is 18/9 =2
    def __init__(self, channel, reduction=9):
        super(SELayerTHW, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(channel, channel // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(channel // reduction, channel, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, c, _, _, _  = x.size()
        y = self.avg_pool(x).view(b, c)
        y = self.fc(y).view(b, c, 1, 1, 1)
        return x * y.expand_as(x)

class SELayerCHW(nn.Module):
    # Excite time
    # With reduction of 2 smallest botteneck is 2/2 = 1
    def __init__(self, temporal, reduction=2):
        super(SELayerCHW, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool3d(1)
        self.fc = nn.Sequential(
            nn.Linear(temporal, temporal // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(temporal // reduction, temporal, bias=False),
            nn.Sigmoid()
        )

    def forward(self, x):
        b, _, t, _, _ = x.size()
        # Permute insteda of view
        y = self.avg_pool(x.permute(0,2,1,3,4)).view(b, t)
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