import numpy as np
import torch
from thop import profile
from torch import nn
import random


def generate_random_block():
    first_param = random.choice(['c', 'i'])
    second_param = random.choice([32, 64, 128, 256])
    third_param = random.choice([3, 5, 7, 9])
    fourth_param = random.choice([1, 2, 3, 4])
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
    if n_params <= max_params and macs*2 <= max_flops:
        return True
    else:
        return False
