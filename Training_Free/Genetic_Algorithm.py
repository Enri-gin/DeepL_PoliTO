import torch
import torch as nn
from NASWOT import compute_naswot_score
from zero_cost_nas.foresight.pruners.predictive import find_measures
from function_and_classes.model_by_blocks import ModelByBlocks
from Training_Free.utils import *
from torchvision import datasets
from torchvision import transforms as T
import copy
import time
import warnings
import random
import json


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

    seq_to_mute[j][1] += random.choice([-40, -30, -20, -10, 10, 20, 30, 40])  # Channels mutation
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

    if seq_to_mute[j][1] >= 256: seq_to_mute[j][1] = 256  # Making sure the output channels size is less than 256
    if seq_to_mute[j][1] <= 32: seq_to_mute[j][1] = 32  # Making sure the output channels size is bigger than 32

    if seq_to_mute[j][1] <= prev_channel: seq_to_mute[j][1] = prev_channel
    # if torch.rand(1).item() < pr:
    #   # aggiungi blocco
    # print('Original Sequence:', best_model.seq_block)
    # print('Mutated Sequence: ', seq_to_mute)
    # print()
    return seq_to_mute


device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean = [0.4701, 0.4468, 0.4075]
std = [0.2617, 0.2573, 0.2734]

transform = T.Compose([
    T.Resize((96, 96)),
    T.RandAugment(num_ops=2, magnitude=8),
    T.ToTensor(),
    T.Normalize(mean, std)
])

data_fake = datasets.CIFAR10(r"C:\Users\Enrico\DeepLProject\data", train=False,
                             transform=transform,
                             download=False)

batch_size = 2
dataloader = torch.utils.data.DataLoader(data_fake, batch_size=batch_size, shuffle=True)

inputs, target = next(iter(dataloader))

# Initialization
M = 50
r = 0.2
Tmax = 20
S = int(np.ceil(r * M))
p = 2
history = 0
count = 0
population = nn.ModuleList()
ages = []
m = 0
with open(r'C:\Users\Enrico\DeepLProject\Models\history_random.json') as f:
    search_space = json.load(f)
for string in search_space:
    m += 1
    if m == M:
        break
    population.append(ModelByBlocks(string))
    ages.append(0)

warnings.filterwarnings("ignore")

# Algorithm
while history < Tmax:
    print(f"Iteration {history}/{Tmax}")
    count = 0
    while count < S:  # randomly discard S models
        j = random.choice(range(len(population)))
        del population[j]
        del ages[j]
        count += 1
        # print(len(population))

    count = 0
    while count < p:
        oldest_index = ages.index(max(ages))
        del ages[oldest_index]
        del population[oldest_index]
        count += 1

    # Finding the best parent
    j = find_best_model(population, inputs, target, dataloader)
    best_model = population[j[0]]
    while len(population) < M:
        new_seq = generate_mutation(best_model)
        print(new_seq)
        new_model = ModelByBlocks(new_seq)
        if respect_constraints(new_model):
            population.append(new_model)
            ages.append(0)

    # Increase all the ages and history counter
    ages = [x + 1 for x in ages]
    history += 1
