import torch
import torch.nn as nn

class DW(nn.Module):
    def __init__(self, c_in, c_out, kernel_size, stride=1, padding=0, norm=None, act=None):
        super().__init__()

        self.d = nn.Conv2d(c_in, c_in, kernel_size=kernel_size, stride=stride, padding=padding, groups=c_in, bias= (norm is None))
        self.p = nn.Conv2d(c_in, c_out, 1, bias= (norm is None))
        self.norm = norm(c_out) if norm is not None else nn.Identity()
        self.act = act(inplace=True) if act is not None else nn.Identity()

    def forward(self, x):
        return self.act(self.norm(self.p(self.d(x))))

class CDW(nn.Module):
    def __init__(self, c_in, c_out, k_size=3, exp=2):
        super().__init__()

        self.c1 = nn.Conv2d(c_in, c_in, kernel_size=1)

        self.dw1 = DW(c_in=c_in//2, c_out=int(c_in*exp), kernel_size=k_size, norm=nn.BatchNorm2d, padding=k_size//2)
        self.dw2 = DW(c_in=int(c_in*exp), c_out=c_in//2, kernel_size=k_size, act=nn.SiLU, padding=k_size//2)

        self.co = Conv(3*(c_in//2), c_out, k=1)


    def forward(self, x):
        x1, x2 = self.c1(x).chunk(2, 1)

        intermidiate = self.dw2(self.dw1(x2))+x2   #shortcut connection
        return self.co(torch.cat([x1, x2, intermidiate], 1))
