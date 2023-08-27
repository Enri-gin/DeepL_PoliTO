import torch
import torch.nn as nn


def train_model(model: nn.Module, dataloaders: dict, dataset_sizes: dict, criterion, optimizer,
                scheduler, device=torch.device("cpu"), num_epochs=5):
    import time
    import copy
    import gc

    since = time.time()
    dict_stats = {}

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    train_loss = []
    val_loss = []

    train_acc = []
    val_acc = []

    for epoch in range(num_epochs):
        start_training_epoch = time.time()
        print('Epoch {}/{}'.format(epoch+1, num_epochs))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train()  # Set model to training mode
            else:
                model.eval()  # Set model to evaluate mode

            running_loss = 0.0
            running_corrects = 0

            for inputs, labels in dataloaders[phase]:
                inputs = inputs.to(device)
                labels = labels.to(device)

                # forward
                # track history if only in train
                with torch.set_grad_enabled(phase == 'train'):
                    outputs = model(inputs)
                    _, preds = torch.max(outputs, 1)
                    loss = criterion(outputs, labels)

                    # backward + optimize only if in training phase
                    if phase == 'train':
                        optimizer.zero_grad()
                        loss.backward()
                        optimizer.step()

                # statistics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            if phase == 'train':
                train_loss.append(epoch_loss)
                train_acc.append(epoch_acc)

            if phase == 'val':
                val_loss.append(epoch_loss)
                val_acc.append(epoch_acc)

            print('{} Loss: {:.4f} Acc: {:.4f}'.format(
                phase, epoch_loss, epoch_acc))
            # deep copy the model
            if phase == 'val' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

        time_epoch = time.time() - start_training_epoch
        print('Epoch training time: {:.0f}m {:.0f}s'.format(time_epoch // 60, time_epoch % 60))
        print()

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))
    print('Best val Acc: {:4f}'.format(best_acc))

    if torch.cuda.current_device(): # deleting inputs if using torch cuda
        inputs.to(torch.device("cpu"))
        labels.to(torch.device("cpu"))
        del inputs
        del labels
        gc.collect()
        torch.cuda.empty_cache()

    # load best model weights
    model.load_state_dict(best_model_wts)

    dict_stats['Training Accuracy'] = train_acc
    dict_stats['Validation Accuracy'] = val_acc

    dict_stats['Training Loss'] = train_loss
    dict_stats['Validation Loss'] = val_loss

    return model, dict_stats
