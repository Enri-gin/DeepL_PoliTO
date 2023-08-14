import numpy as np
from torch import nn


def compute_NASWOT(model: nn.Module, dataloader_train):

    x, target = next(iter(dataloader_train))
    batch_size = x.shape[0]

    def counting_forward_hook(module, inp, out):
        inp = inp[0].reshape(inp[0].size(0), -1)
        x = (inp > 0).float()  # binary indicator
        K = x @ x.t()
        K2 = (1. - x) @ (1. - x.t())
        model.K = model.K + K.cpu().numpy() + K2.cpu().numpy()

    def counting_backward_hook(module, inp, out):
        module.visited_backwards = True

    # This is the logarithm of the determinant of K
    def hooklogdet(K, labels=None):
        s, ld = np.linalg.slogdet(K)
        return ld

    model.K = np.zeros((batch_size, batch_size))
    for name, module in model.named_modules():
        if 'ReLU' in str(type(module)):
            module.register_backward_hook(counting_backward_hook)
            module.register_forward_hook(counting_forward_hook)

    # run batch through network
    model(x)

    score = hooklogdet(model.K, target)
    return score
