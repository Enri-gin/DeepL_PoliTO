import numpy as np
import torch
from thop import profile
from torch import nn
import random
import copy
from NASWOT import compute_naswot_score
from zero_cost_nas.foresight.pruners.predictive import find_measures


def generate_random_block():
    first_param = random.choice(['c', 'i'])
    second_param = random.randrange(32, 257, 2)
    third_param = random.choice([3, 5, 7, 9])
    fourth_param = random.choice([1, 2, 3, 4, 5])
    fifth_param = random.choice([1, 2])
    return [first_param, second_param, third_param, fourth_param, fifth_param]


def generate_random_list_of_lists(num_lists, num_blocks_per_list, seed=None):
    if seed is not None:
        random.seed(seed)

    result = []
    for _ in range(num_lists):
        new_list = [generate_random_block() for _ in range(num_blocks_per_list)]
        result.append(new_list)
    return result


def calculate_fitness(list1, list2, N):
    # Calculate the maximum of each list
    max_list1 = max(list1)
    max_list2 = max(list2)

    # Divide each list by its own maximum
    normalized_list1 = [val / max_list1 for val in list1]
    normalized_list2 = [val / max_list2 for val in list2]

    # Sum the two normalized lists
    fitness_list = [x + y for x, y in zip(normalized_list1, normalized_list2)]

    # Get the top N indices in the fitness_list
    top_indices = sorted(range(len(fitness_list)), key=lambda i: fitness_list[i], reverse=True)[:N]

    return fitness_list, top_indices


def count_params(model: nn.Module):
    model_parameters = filter(lambda p: p.requires_grad, model.parameters())
    params = sum([np.prod(p.size()) for p in model_parameters])
    return params.item()


def get_macs_and_params(model: nn.Module, input_shape: list):
    model_device = next(model.parameters()).device
    input_ = torch.rand(input_shape, device=model_device)
    macs, params = profile(model, inputs=(input_,), verbose=False)
    return macs, params


def respect_constraints(model: nn.Module, max_params=2_500_000, max_flops=200_000_000, dim_image=96) -> bool:
    macs, n_params = get_macs_and_params(model, [1, 3, dim_image, dim_image])
    if n_params <= max_params and macs * 2 <= max_flops:
        return True
    else:
        return False


def generate_mutation(best_model: nn.Module, n_mutation=2):
    seq_to_mute = copy.deepcopy(best_model.seq_block)

    for _ in range(n_mutation):
        j = random.choice(range(len(seq_to_mute)))

        type_block, out_channels, kernel_size, exp_factor, stride = seq_to_mute[j]

        out_channels += random.choice([-32, -24, -16, -8, 8, 16, 24, 32])  # Channels mutation
        kernel_size += random.choice([-2, 0, 2])  # Kernel size mutation
        exp_factor += random.choice([-1, 0, 1])  # exp_factor mutation
        stride = random.choice([1, 2])  # stride mutation (new selection)

        if exp_factor <= 0: exp_factor = 1  # Making sure exp_factor is positive
        if exp_factor >= 5: exp_factor = 5  # Making sure exp_factor is less than 5

        if kernel_size <= 3: kernel_size = 3  # Making sure the kernel size is positive
        if kernel_size >= 9: kernel_size = 9  # Making sure the kernel size is less than 9

        if out_channels >= 256: out_channels = 256  # Making sure the output channels size is less than 256
        if out_channels <= 32: out_channels = 32  # Making sure the output channels size is bigger than 32

        seq_to_mute[j] = [type_block, out_channels, kernel_size, exp_factor, stride]

    return seq_to_mute
