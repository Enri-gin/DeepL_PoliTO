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
datasizes = {'train': len(data_fake), 'val': len(data_fake)}

# hand_model = ModelByBlocks([['i', 64, 7, 2, 2], ['c', 32, 3, 1, 2], ['i', 128, 3, 4, 2], ['c', 128, 4, 1, 1],
#                             ['i', 64, 5, 3, 1], ['c', 128, 3, 3, 1], ['c', 128, 3, 1, 1], ['i', 256, 5, 1, 2],
#                             ['i', 32, 3, 4, 1], ['i', 256, 7, 4, 2]])
hand_model = ModelByBlocks([['i', 128, 3, 4, 2], ['c', 128, 3, 1, 1]])

# augmentation_function = T.RandAugment(num_ops=2, magnitude=8)

criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(hand_model.parameters(), lr=0.05)
scheduler = lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.1)

fitted_hand_model = train_model(hand_model, dataloaders, datasizes, criterion, optimizer,
                                scheduler, device=device, num_epochs=5)
