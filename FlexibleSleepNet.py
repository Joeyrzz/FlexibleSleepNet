import torch
import torch.nn as nn
import torch.nn.functional as F


def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)
    random_tensor = keep_prob + torch.rand(shape, dtype=x.dtype, device=x.device)
    random_tensor.floor_()
    output = x.div(keep_prob) * random_tensor
    return output


class DropPath(nn.Module):
    def __init__(self, drop_prob=None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob

    def forward(self, x):
        return drop_path(x, self.drop_prob, self.training)


class LayerNorm(nn.Module):
    def __init__(self, normalized_shape, eps=1e-6, data_format="channels_last"):
        super().__init__()
        self.weight = nn.Parameter(torch.ones(normalized_shape), requires_grad=True)
        self.bias = nn.Parameter(torch.zeros(normalized_shape), requires_grad=True)
        self.eps = eps
        self.data_format = data_format
        if self.data_format not in ["channels_last", "channels_first"]:
            raise ValueError(f"not support data format '{self.data_format}'")
        self.normalized_shape = (normalized_shape,)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        if self.data_format == "channels_last":
            return F.layer_norm(x, self.normalized_shape, self.weight, self.bias, self.eps)
        elif self.data_format == "channels_first":
            mean = x.mean(1, keepdim=True)
            var = (x - mean).pow(2).mean(1, keepdim=True)
            x = (x - mean) / torch.sqrt(var + self.eps)
            x = self.weight[:, None, None] * x + self.bias[:, None, None]
            return x


class AFE_Module(nn.Module):
    r"""Define AFE Module"""
    def __init__(self, dim, drop_rate=0., layer_scale_init_value=1e-6):
        super().__init__()
        self.fe = nn.Conv2d(dim, dim, kernel_size=7, padding=3, groups=dim)
        self.norm = LayerNorm(dim, eps=1e-6, data_format="channels_last")
        self.ce = nn.Linear(dim, 4 * dim)  # Channel Excitation
        self.cs = nn.Linear(4 * dim, dim)  # Channel Squeeze
        self.act = nn.LeakyReLU()
        self.gamma = nn.Parameter(layer_scale_init_value * torch.ones((dim,)),
                                  requires_grad=True) if layer_scale_init_value > 0 else None
        self.drop_path = DropPath(drop_rate) if drop_rate > 0. else nn.Identity()

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        shortcut = x
        x = self.fe(x)
        x = x.permute(0, 2, 3, 1)
        x = self.norm(x)
        x = self.cs(self.act(self.ce(x)))
        if self.gamma is not None:
            x = self.gamma * x
        x = x.permute(0, 3, 1, 2)
        x = shortcut + self.drop_path(x)
        return x


class FlexibleSleepNet(nn.Module):
    def __init__(self, in_chans: int = 3, num_classes: int = 5, depths: list = [3, 3, 9, 3],
                 dims: list = [96, 192, 384, 768], drop_path_rate: float = 0.1, layer_scale_init_value: float = 1e-6,
                 head_init_scale: float = 1.):
        super().__init__()
        # Define Scale-Vary Compression Module
        self.svc = nn.ModuleList()
        stem = nn.Sequential(nn.Conv2d(in_chans, dims[0], kernel_size=4, stride=4),
                             nn.BatchNorm2d(dims[0]),
                             nn.Dropout(0.1))
        self.svc.append(stem)
        for i in range(3):
            svc = nn.Sequential(nn.BatchNorm2d(dims[i], eps=1e-6),
                                nn.Conv2d(dims[i], dims[i+1], kernel_size=2, stride=2),
                                nn.Dropout(0.1))
            self.svc.append(svc)
        # Define Stage
        self.stages = nn.ModuleList()
        dp_rates = [x.item() for x in torch.linspace(0, drop_path_rate, sum(depths))]
        cur = 0
        for i in range(4):
            stage = nn.Sequential(
                *[AFE_Module(dim=dims[i], drop_rate=dp_rates[cur + j], layer_scale_init_value=layer_scale_init_value)
                  for j in range(depths[i])]
            )
            self.stages.append(stage)
            cur += depths[i]

        self.norm = nn.LayerNorm(dims[-1], eps=1e-6)  # final norm layer
        self.classifier = nn.Linear(dims[-1], num_classes)
        self.apply(self._init_weights)
        self.classifier.weight.data.mul_(head_init_scale)
        self.classifier.bias.data.mul_(head_init_scale)
        self.dropout = nn.Dropout(p=0.5)

    def _init_weights(self, m):
        if isinstance(m, (nn.Conv2d, nn.Linear)):
            nn.init.trunc_normal_(m.weight, std=0.2)
            nn.init.constant_(m.bias, 0)

    def dep_extract(self, x: torch.Tensor) -> torch.Tensor:
        for i in range(4):
            x = self.svc[i](x)
            x = self.stages[i](x)
            x = self.dropout(x)
        return self.norm(x.mean([-2, -1]))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = F.interpolate(x, size=(32, 128), mode='bicubic', align_corners=None)
        x = self.dep_extract(x)
        x = self.classifier(x)
        return x
