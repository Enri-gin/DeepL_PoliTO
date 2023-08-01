import torch.nn as nn
import torch.nn.functional as F


class Bottleneck_V3(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=3, expansion_factor=4, stride=1, reduction_ratio=4,
                 dropout_prob=0.2):
        super(Bottleneck_V3, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)
        self.DoBatchNorm = True

        # Pointwise expansion
        mid_channels = in_channels * expansion_factor
        self.conv1 = nn.Conv2d(in_channels, mid_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(mid_channels)
        self.act1 = nn.Hardswish()

        # Depthwise convolution
        self.conv2 = nn.Conv2d(mid_channels, mid_channels, kernel_size, stride=stride, padding=(kernel_size - 1) // 2,
                               groups=mid_channels, bias=False)
        self.bn2 = nn.BatchNorm2d(mid_channels)
        self.act2 = nn.Hardswish()

        # Squeeze-and-excitation
        squeeze_channels = max(1, in_channels // reduction_ratio)
        self.se_conv1 = nn.Conv2d(mid_channels, squeeze_channels, kernel_size=1)
        self.se_act1 = nn.Hardswish()
        self.se_conv2 = nn.Conv2d(squeeze_channels, mid_channels, kernel_size=1)
        self.se_act2 = nn.Sigmoid()

        # Pointwise linear projection
        self.conv3 = nn.Conv2d(mid_channels, out_channels, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels)

        # Dropout
        self.dropout = nn.Dropout2d(dropout_prob)

    def forward(self, x):
        identity = x

        out = self.conv1(x)

        if self.DoBatchNorm:
            out = self.bn1(out)

        out = self.act1(out)

        out = self.conv2(out)

        if self.DoBatchNorm:
            out = self.bn2(out)

        out = self.act2(out)

        # Squeeze-and-excitation
        se = F.avg_pool2d(out, out.size(2))
        se = self.se_conv1(se)
        se = self.se_act1(se)
        se = self.se_conv2(se)
        se = self.se_act2(se)
        out = out * se

        out = self.conv3(out)

        if self.DoBatchNorm:
            out = self.bn3(out)

        # Dropout
        if self.dropout.p != 0:
            out = self.dropout(out)

        if self.use_residual:
            out += identity

        out = F.hardswish(out)
        return out

    def disable_bn(self):
        self.DoBatchNorm = False

    def enable_bn(self):
        self.DoBatchNorm = True


class ConvNeXtBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size=7, expansion_factor=4, stride=1):
        super(ConvNeXtBlock, self).__init__()
        self.use_residual = (stride == 1 and in_channels == out_channels)

        # Depthwise 7x7 convolutional layer
        self.depthwise_conv = nn.Conv2d(in_channels, in_channels, kernel_size=kernel_size, stride=stride,
                                        padding=kernel_size // 2, groups=in_channels)

        # Linear normalization
        self.linear_norm = nn.LayerNorm(in_channels)

        # 1x1 convolution with expansion factor
        self.pointwise_conv = nn.Linear(in_channels, out_channels * expansion_factor)

        # GELU activation
        self.gelu = nn.GELU()

        # 1x1 convolution to restore output size
        self.restore_conv = nn.Linear(out_channels * expansion_factor, out_channels)

    def forward(self, x):
        identity = x

        # Depthwise 7x7 convolution
        out = self.depthwise_conv(x)
        # Permute
        out = out.permute(0, 2, 3, 1)  # (N, C, H, W) -> (N, H, W, C)
        # Linear normalization
        out = self.linear_norm(out)
        # 1x1 convolution with expansion
        out = self.pointwise_conv(out)
        # GELU activation
        out = self.gelu(out)
        # 1x1 convolution to restore output size
        out = self.restore_conv(out)
        # Re-permute
        out = out.permute(0, 3, 1, 2)  # (N, H, W, C) -> (N, C, H, W)
        # Residual connection
        if self.use_residual:
            out += identity

        return out
