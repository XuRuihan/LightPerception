import torch
import torch.nn as nn


def autopad(k, p=None, d=1):
    """Pad to 'same' shape outputs."""
    if d > 1:
        k = d * (k - 1) + 1 if isinstance(k, int) else [d * (x - 1) + 1 for x in k]
    if p is None:
        p = k // 2 if isinstance(k, int) else [x // 2 for x in k]
    return p


class Conv(nn.Module):
    """Conv-BN-Act block used by the depth module sync from 3dppe_x1."""

    default_act = nn.ReLU()

    def __init__(self, c1, c2, k=1, s=1, p=None, g=1, d=1, act=True, zero_init=False):
        super().__init__()
        self.conv = nn.Conv2d(
            c1,
            c2,
            k,
            s,
            autopad(k, p, d),
            groups=g,
            dilation=d,
            bias=False)
        self.bn = nn.BatchNorm2d(c2)
        self.act = self.default_act if act is True else act if isinstance(act, nn.Module) else nn.Identity()
        self._init_weight()
        if zero_init:
            nn.init.zeros_(self.conv.weight)

    def forward(self, x):
        return self.act(self.bn(self.conv(x)))

    def forward_fuse(self, x):
        return self.act(self.conv(x))

    def _init_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                torch.nn.init.kaiming_normal_(m.weight)
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
