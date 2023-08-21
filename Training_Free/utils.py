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

def find_best_model(population: nn.ModuleList, inp, target, dataloader):
    """Score the population of the genetic algorithm according
    to the FreeRea score function"""
    num_classes = 2
    measure_names = ['synflow']
    n_batches = 2
    score_syn = []
    score_nas = []
    for model in population:
        measures = find_measures(model, dataloader, dataload_info=('random', n_batches, num_classes),
                                 device=device, measure_names=measure_names)

        score_syn.append(measures['synflow'])
        score_nas.append(compute_naswot_score(model, inp, target, device))
    _, best_index = calculate_fitness(score_syn, score_nas, 1)
    return best_index


def generate_mutation(best_model: nn.Module):
    pr = 0.0  # Probabilità aggiunta blocco

    seq_to_mute = copy.deepcopy(best_model.seq_block)
    j = random.choice(range(len(seq_to_mute)))
    # print(j, seq_to_mute[j])

    prev_channel = seq_to_mute[j - 1][1]

    seq_to_mute[j][1] += random.choice([-4, -3, -2, -1, 1, 2, 3, 4])  # Channels mutation
    seq_to_mute[j][2] += random.choice([-2, 0, 2])  # Kernel size mutation
    seq_to_mute[j][3] += random.choice([-1, 0, 1])  # Modifying the expansion factor

    if len(seq_to_mute[j]) == 5:  # if there is also the stride
        seq_to_mute[j][4] += random.choice([-1, 0, 1])
    else:
        seq_to_mute[j].append(random.choice([1, 2]))

    for k in range(j, len(seq_to_mute)):  # Making sure the number of channels along the network is not decreasing
        if seq_to_mute[k][1] < seq_to_mute[j][1]:
            seq_to_mute[k][1] = seq_to_mute[j][1]

    if seq_to_mute[j][3] <= 0: seq_to_mute[j][3] = 1  # Making sure exp_factor is positive
    if seq_to_mute[j][4] <= 0: seq_to_mute[j][4] = 1  # Making sure stride is positive
    if seq_to_mute[j][4] >= 3: seq_to_mute[j][4] = 2  # Making sure stride is less than 3
    if seq_to_mute[j][2] <= 0: seq_to_mute[j][2] = 3  # Making sure the kernel size is positive

    if seq_to_mute[j][1] <= prev_channel: seq_to_mute[j][1] = prev_channel

    # if torch.rand(1).item() < pr:
    #   # aggiungi blocco
    # print('Original Sequence:', best_model.seq_block)
    # print('Mutated Sequence: ', seq_to_mute)
    # print()
    return seq_to_mute
