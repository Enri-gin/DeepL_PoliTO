# import torch
# import torch.nn as nn
# import torch.optim as optim
# from torch.optim import lr_scheduler
# import numpy as np
import torchvision
# from torchvision import datasets, models, transforms
# import matplotlib.pyplot as plt
# import pyvww
# import time
# import os
# import copy
from zero_cost_nas.foresight.pruners.predictive import find_measures
from function_and_classes.model_by_blocks import ModelByBlocks
from function_and_classes.others import *

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
data_fake = torchvision.datasets.CIFAR10(".\data", train=False,
                                         transform=torchvision.transforms.ToTensor(),
                                         download=False)


batch_size = 5
dataloader = torch.utils.data.DataLoader(data_fake, batch_size=batch_size, shuffle=True)

model = ModelByBlocks([['i', 32, 7, 1], ['c', 32, 5, 1], ['i', 64, 5, 3],
                       ['i', 256, 3, 2], ['c', 256, 7, 4],
                       ['i', 32, 3, 2], ['i', 32, 3, 2]])


num_classes = 2

measure_names = ['synflow']
n_batches = 2

disable_batch_norm(model)

measures = find_measures(model, dataloader, dataload_info=('random', n_batches, num_classes),
                         device=device, measure_names=measure_names)
enable_batch_norm(model)

score = measures['synflow']
print(f'LogSynflow score = {score}')

masc, n_param = get_macs_and_params(model, [16, 3, 96, 96])
print(f'model has {masc:.2f} MACs and {n_param} parameters')
