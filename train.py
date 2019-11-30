import os
import time

import numpy as np
import torch
from torch import nn, optim
from torch.utils import data
from torch.utils.data import DataLoader
from tqdm import tqdm

from dataset import VideoDataset, VideoDataset1M
#from network import R2Plus1DClassifier
import fire
from models.cls_hrnet_2dplus1 import get_cls_net
from default import _C as config


# Use GPU if available else revert to CPU
#device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
#print("Device being used:", device)

def run(directory="/datasets/Moments_in_Time_256x256_30fps",
        num_epochs=45,
        path="/datasets/models/model_data.pth.tar",
        save=True,
        *options, cfg=None, local_rank=0):

    """Args:
            directory (str): Directory where the data is to be loaded from
            num_epochs (int, optional): Number of epochs to train for. Defaults to 45. 
            save (bool, optional): If true, the model will be saved to path. Defaults to True. 
            path (str, optional): The directory to load a model checkpoint from, and if save == True, save to. Defaults to "model_data.pth.tar".
    """

    config.defrost()
    config.merge_from_file(cfg)
    torch.backends.cudnn.benchmark = config.CUDNN.BENCHMARK

    #Distributed
    world_size = 8
    torch.cuda.set_device(local_rank)
    torch.distributed.init_process_group(backend="nccl", init_method="env://")

    # Dataloaders
    train_set = VideoDataset(directory, clip_len=32)
    train_sampler = torch.utils.data.distributed.DistributedSampler(
        train_set, num_replicas=world_size, rank=local_rank
    )
    train_loader = data.DataLoader(
        train_set, batch_size=3, num_workers=4, sampler=train_sampler,
    )

    val_set = VideoDataset(directory, mode='val', clip_len=32)
    val_sampler = torch.utils.data.distributed.DistributedSampler(
        val_set, num_replicas=world_size, rank=local_rank)
    val_loader = data.DataLoader(
        val_set, batch_size=3, num_workers=4, sampler=val_sampler,
    )

    dataloaders = {'train': train_loader, 'val': val_loader}

    dataset_sizes = {x: len(dataloaders[x].dataset) for x in ['train', 'val']}

    if torch.cuda.is_available():
        device = "cuda"
    else:
        raise Exception("Cannot find GPU")

    model = get_cls_net(config).to(device)
    model = torch.nn.SyncBatchNorm.convert_sync_batchnorm(model)
    model = torch.nn.parallel.DistributedDataParallel(
        model, device_ids=[device], find_unused_parameters=True)

    
    criterion = nn.CrossEntropyLoss() # standard crossentropy loss for classification
    optimizer = optim.SGD(model.parameters(), lr=0.01)  # hyperparameters as given in paper sec 4.1
    scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=10, gamma=0.1)  # the scheduler divides the lr by 10 every 10 epochs

    # saves the time the process was started, to compute total time at the end
    start = time.time()
    epoch_resume = 0

    # check if there was a previously saved checkpoint
    if os.path.exists(path):
        # loads the checkpoint
        checkpoint = torch.load(path)
        print("Reloading from previously saved checkpoint")

        # restores the model and optimizer state_dicts
        model.load_state_dict(checkpoint['state_dict'])
        optimizer.load_state_dict(checkpoint['opt_dict'])
        
        # obtains the epoch the training is to resume from
        epoch_resume = checkpoint["epoch"]

    for epoch in tqdm(range(epoch_resume, num_epochs), unit="epochs", initial=epoch_resume, total=num_epochs):
        # each epoch has a training and validation step, in that order
        for phase in ['train', 'val']:

            # reset the running loss and corrects
            running_loss = 0.0
            running_corrects = 0

            # set model to train() or eval() mode depending on whether it is trained
            # or being validated. Primarily affects layers such as BatchNorm or Dropout.
            if phase == 'train':
                model.train()
            else:
                model.eval()


            for inputs, labels in dataloaders[phase]:
                # move inputs and labels to the device the training is taking place on
                inputs = inputs.to(device)
                labels = labels.to(device)
                optimizer.zero_grad()

                # keep intermediate states iff backpropagation will be performed. If false, 
                # then all intermediate states will be thrown away during evaluation, to use
                # the least amount of memory possible.
                with torch.set_grad_enabled(phase=='train'):
                    outputs = model(inputs)
                    # we're interested in the indices on the max values, not the values themselves
                    _, preds = torch.max(outputs, 1)  
                    loss = criterion(outputs, labels)

                    # Backpropagate and optimize iff in training mode, else there's no intermediate
                    # values to backpropagate with and will throw an error.
                    if phase == 'train':
                        loss.backward()
                        optimizer.step()   

                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            if phase == 'train':
                # scheduler.step() is to be called once every epoch during training
                # FLAG should be called after optimizer step?
                scheduler.step()

            epoch_loss = running_loss / dataset_sizes[phase]
            epoch_acc = running_corrects.double() / dataset_sizes[phase]

            print(f"{phase} Loss: {epoch_loss} Acc: {epoch_acc}")

    # save the model if save=True
    if save:
        torch.save({
        'epoch': epoch + 1,
        'state_dict': model.state_dict(),
        'acc': epoch_acc,
        'opt_dict': optimizer.state_dict(),
        }, path)

    # print the total time needed, HH:MM:SS format
    time_elapsed = time.time() - start    
    print(f"Training complete in {time_elapsed//3600}h {(time_elapsed%3600)//60}m {time_elapsed %60}s")

if __name__ == "__main__":
    fire.Fire(run)
