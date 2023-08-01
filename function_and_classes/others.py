from .Blocks import *
import numpy as np
import torch
from thop import profile


def disable_batch_norm(blocks_model):
    for block in blocks_model.blocks:
        if isinstance(block, Bottleneck_V3):
            block.disable_bn()


def enable_batch_norm(blocks_model):
    for block in blocks_model.blocks:
        if isinstance(block, Bottleneck_V3):
            block.enable_bn()


def count_params(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params.item()


def get_macs_and_params(model: nn.Module, input_shape: list):
    model_device = next(model.parameters()).device
    input_ = torch.rand(input_shape, device=model_device)
    macs, params = profile(model, inputs=(input_,), verbose=False)
    return macs, params
