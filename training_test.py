from function_and_classes.model_by_blocks import ModelByBlocks
from Training_Free.utils import *
from torchvision import datasets
from torchvision import transforms as T
import torch.nn
from torch.optim import lr_scheduler
from function_and_classes.train_model import train_model

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

mean = [0.4701, 0.4468, 0.4075]
std = [0.2617, 0.2573, 0.2734]

transform = T.Compose([
    T.Resize((96, 96)),
    T.RandAugment(num_ops=2, magnitude=8),
    T.ToTensor(),
    T.Normalize(mean, std)
])

data_fake = datasets.CIFAR10(".\data", train=False,
                             transform=transform,
                             download=False)


batch_size = 2
dataloader = torch.utils.data.DataLoader(data_fake, batch_size=batch_size, shuffle=True)
dataloaders = {'train': dataloader, 'val': dataloader}

hand_model = ModelByBlocks([['i', 32, 5, 1], ['c', 64, 5, 1],
                              ['i', 64, 5, 3], ['i', 128, 3, 2],
                              ['c', 128, 7, 4], ['i', 256, 3, 2],
                              ['c', 256, 3, 2]])

# augmentation_function = T.RandAugment(num_ops=2, magnitude=8)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(hand_model.parameters(), lr=0.05)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

fitted_hand_model = train_model(hand_model, dataloaders, criterion, optimizer,
                                scheduler, device=device, num_epochs=5)
