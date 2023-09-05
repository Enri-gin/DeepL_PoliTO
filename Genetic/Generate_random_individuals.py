import random
import torch
import warnings
import time
import gc
from utils import generate_random_list_of_lists
from utils import get_macs_and_params, count_params
from ..function_and_classes.model_by_blocks import ModelByBlocks

N_SEARCH = 1000
seed_value = 42
MAX_PARAMS = 2_500_000
MAX_FLOPS = 200_000_000
n_blocks = 10
dim_image = 96

random_search = generate_random_list_of_lists(N_SEARCH, n_blocks, seed=seed_value)
times_rejected = []

# Disabilita tutti i messaggi di avviso
warnings.filterwarnings("ignore")

print("Progression in percentage:")

for i in range(N_SEARCH):

    percent = (i / (N_SEARCH - 1)) * 100
    print(f"\r{percent:.2f}%", end="")
    time.sleep(0.001)  # Aggiungi un ritardo per simulare l'aggiornamento della barra

    random_model = ModelByBlocks(random_search[i])
    macs, params = get_macs_and_params(random_model, [1, 3, 96, 96])

    count = 0

    if params > MAX_PARAMS or macs * 2 > MAX_FLOPS:  # If the model that we just created overcomes the costraints:
        flag_to_replace = 0
        while flag_to_replace == 0:
            seed_value = seed_value + 1
            fictional_pop = generate_random_list_of_lists(1, n_blocks, seed=seed_value)
            model_to_replace = ModelByBlocks(fictional_pop[0])
            macs, _ = get_macs_and_params(model_to_replace, [1, 3, dim_image, dim_image])
            if count_params(model_to_replace) <= MAX_PARAMS and macs * 2 <= MAX_FLOPS:
                random_search[i] = fictional_pop[0]
                flag_to_replace = 1
            else:
                count += 1
            model_to_replace.cpu()
            del model_to_replace
            gc.collect()
            torch.cuda.empty_cache()

    times_rejected.append(count)
    random_model.cpu()
    del random_model
    gc.collect()
    torch.cuda.empty_cache()

print()  # Vai a capo dopo aver completato la progressione
print(f"Average acception rate = {sum(times_rejected) / len(times_rejected)}")

# Riabilita i messaggi di avviso
warnings.resetwarnings()
