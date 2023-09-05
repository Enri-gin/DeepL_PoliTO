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
import gc

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
dataloader_train = torch.utils.data.DataLoader(data_fake, batch_size=batch_size, shuffle=True)

# Initialization
P = 50
C = 100
n_blocks = 10
S = 6  # sub-sample random
seed_value = 42
n_blocks = 10
dim_image = 96
count_crossover = 0
num_classes = 2
measure_names = ['synflow']

with open(r'/Models/population/Initial_population.json') as f_json:
    initial_population = json.load(f_json)
x, target = next(iter(dataloader_train))
# Disabilita tutti i messaggi di avviso
warnings.filterwarnings("ignore")

random_seed = 42
random.seed(random_seed)
history = []
for i in range(P):
    history.append(initial_population[i])
print("Progression in percentage:")

for i in range(C):
    percent = (i / (C - 1)) * 100
    print(f"\r{percent:.2f}%", end="")
    time.sleep(0.001)  # Aggiungi un ritardo per simulare l'aggiornamento della barra

    sample_S = random.sample(history[-P:], S)
    score_sample_nas = []
    score_sample_syn = []

    for j in range(S):
        model = ModelByBlocks(sample_S[j])
        score_nas = compute_naswot_score(model, x, target, device)
        score_sample_nas.append(score_nas)

        measures = find_measures(model, dataloader_train,
                                 dataload_info=('random', batch_size, num_classes),
                                 device=device, measure_names=measure_names)
        score_syn = np.log(measures['synflow'])
        score_sample_syn.append(score_syn)

        model.cpu()
        del model
        gc.collect()
        torch.cuda.empty_cache()

    NN = 2
    _, list_index = calculate_fitness(score_sample_nas, score_sample_syn, NN)
    parent1 = sample_S[list_index[0]]
    parent2 = sample_S[list_index[1]]

    parent_1 = copy.deepcopy(parent1)
    parent_2 = copy.deepcopy(parent2)

    crossover_1 = []
    crossover_1.extend(parent_1[:int(n_blocks / 2)])
    crossover_1.extend(parent_2[-int(n_blocks / 2):])

    group = (i - 1) // int(C / 4)  # Calcola il gruppo di iterazioni (0, 1, 2, 3)
    num_blocks_to_mutate = 4 - group

    mod_cross_1 = ModelByBlocks(crossover_1)
    if respect_constraints(mod_cross_1):
        history.append(crossover_1)
        count_crossover += 1
    mod_cross_1.cpu()
    del mod_cross_1
    gc.collect()
    torch.cuda.empty_cache()

    mutation_1 = mutate(parent_1, n_mutation=num_blocks_to_mutate)
    mod_mut_1 = ModelByBlocks(mutation_1)

    while not respect_constraints(mod_mut_1):
        mutation_1 = mutate(parent_1, n_mutation=num_blocks_to_mutate)
        mod_mut_1 = ModelByBlocks(mutation_1)

    history.append(mutation_1)
    mod_mut_1.cpu()
    del mod_mut_1
    gc.collect()
    torch.cuda.empty_cache()

    mutation_2 = mutate(parent_2, n_mutation=num_blocks_to_mutate)
    mod_mut_2 = ModelByBlocks(mutation_2)

    while not respect_constraints(mod_mut_2):
        mutation_2 = mutate(parent_2, n_mutation=num_blocks_to_mutate)
        mod_mut_2 = ModelByBlocks(mutation_2)

    history.append(mutation_2)
    mod_mut_2.cpu()
    del mod_mut_2
    gc.collect()
    torch.cuda.empty_cache()

print()  # Vai a capo dopo aver completato la progressione
print(count_crossover, 'accepted crossovers')
print('Expected history lenght:', P+C*2+count_crossover)
print('Actual Lenght', len(history))
# Riabilita i messaggi di avviso
warnings.resetwarnings()
