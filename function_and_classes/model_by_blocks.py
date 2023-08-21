from torch import nn
from .Blocks import InvertedResidual, ConvNeXtBlock


class ModelByBlocks(nn.Module):
    def __init__(self, seq_block):
        super(ModelByBlocks, self).__init__()
        self.seq_block = seq_block
        self.blocks = nn.Sequential()
        prev_channel = 3
        for blocco in seq_block:
            if blocco[0] == 'c':  # Abbiamo un blocco ConvNext
                if len(blocco) == 5:

                    _, out_channel, kernel_size, exp_fact, stride = blocco
                    self.blocks.append(ConvNeXtBlock(prev_channel, out_channel,
                                                     kernel_size=kernel_size,
                                                     expansion_factor=exp_fact, stride=stride))
                    prev_channel = out_channel

                else:
                    _, out_channel, kernel_size, exp_fact = blocco
                    self.blocks.append(ConvNeXtBlock(prev_channel, out_channel,
                                                     kernel_size=kernel_size,
                                                     expansion_factor=exp_fact))
                    prev_channel = out_channel

            if blocco[0] == 'i':  # Abbiamo una inverted bottleneck (mobile net v3)
                if len(blocco) == 5:

                    _, out_channel, kernel_size, exp_fact, stride = blocco
                    self.blocks.append(InvertedResidual(prev_channel, out_channel,
                                                        kernel_size=kernel_size,
                                                        expansion_factor=exp_fact,
                                                        stride=stride))
                    prev_channel = out_channel

                else:
                    _, out_channel, kernel_size, exp_fact = blocco
                    self.blocks.append(InvertedResidual(prev_channel, out_channel,
                                                        kernel_size=kernel_size,
                                                        expansion_factor=exp_fact))
                    prev_channel = out_channel

        # Global Average Pooling
        self.global_avg_pool = nn.AdaptiveAvgPool2d(1)

        # Fully Connected Layer
        self.fc = nn.Linear(prev_channel, 2)

    def forward(self, x):
        x = self.blocks(x)
        x = self.global_avg_pool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x
