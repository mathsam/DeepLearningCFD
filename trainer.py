import time
import os
import torch
from torch import nn
from ml_models import SuperResolutionNet
from datasets import SRDataset
from utils import *
from torch.utils.tensorboard import SummaryWriter


# Data parameters
exp_name = "GAN_Sep10_exp2"
summary_writer = SummaryWriter(log_dir=exp_name)
data_folder = './exp2'  # folder with JSON data files
scaling_factor = 8  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor
save_dir = os.path.join("../azblob/ssres_model", exp_name)
os.makedirs(save_dir, exist_ok=True)

# Model parameters
num_layers = 16
large_kernel_size = 15  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size = 5  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks
# n_channels = 64  # number of channels in-between, i.e. the input and output channels for the residual and subpixel convolutional blocks
# n_blocks = 16  # number of residual blocks

# Learning parameters
checkpoint = "/home/juchai/azblob/sres_model/GAN_Sep10_exp1/epoch_240_checkpoint_srnet.pth.tar"  # path to model checkpoint, None if none
batch_size = 48  # batch size
start_epoch = 0  # start at this epoch
num_epochs = 100  # number of training epochs
workers = 8  # number of workers for loading data in the DataLoader
print_freq = 3  # print training status once every __ batches
lr = 1e-4  # learning rate
grad_clip = None  # clip if gradients are exploding

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        model = SuperResolutionNet(large_kernel_size=large_kernel_size, small_kernel_size=small_kernel_size,
                                   scaling_factor=scaling_factor, num_layers=num_layers).cpu()
        # Initialize the optimizer
        optimizer = torch.optim.Adam(params=filter(lambda p: p.requires_grad, model.parameters()),
                                     lr=lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        model = checkpoint['model']
        optimizer = checkpoint['optimizer']

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        model = nn.DataParallel(model)
    # Move to default device
    model = model.to(device)

    criterion = nn.MSELoss().to(device)

    # Custom dataloaders
    train_dataset = SRDataset(data_folder,
                              scaling_factor=scaling_factor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)  # note that we're passing the collate function here


    # Epochs
    for epoch in range(start_epoch, start_epoch + num_epochs):
        # One epoch's training
        train(train_loader=train_loader,
              model=model,
              criterion=criterion,
              optimizer=optimizer,
              epoch=epoch)

        # Save checkpoint
        save_path = os.path.join(save_dir, f'epoch_{epoch}_checkpoint_srnet.pth.tar')
        if isinstance(model, torch.nn.parallel.DataParallel):
            torch.save({'epoch': epoch,
                       'model': model.module,
                       'optimizer': optimizer},
                       save_path)
        else:
            torch.save({'epoch': epoch,
                       'model': model,
                       'optimizer': optimizer},
                       save_path)


def train(train_loader, model, criterion, optimizer, epoch):
    """
    One epoch's training.
    :param train_loader: DataLoader for training data
    :param model: model
    :param criterion: content loss function (Mean Squared-Error loss)
    :param optimizer: optimizer
    :param epoch: epoch number
    """
    model.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses = AverageMeter()  # loss

    start = time.time()

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 1, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 1, 96, 96), in [-1, 1]

        # Forward prop.
        sr_imgs = model(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

        # Loss
        loss = criterion(sr_imgs, hr_imgs)  # scalar

        # Backward prop.
        optimizer.zero_grad()
        loss.backward()

        # Clip gradients, if necessary
        if grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)

        # Update model
        optimizer.step()

        # Keep track of loss
        losses.update(loss.item(), lr_imgs.size(0))

        # Keep track of batch time
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        global_step = i + epoch * len(train_loader)
        summary_writer.add_scalar(os.path.join(exp_name, "loss"), losses.val, global_step)

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses))
    print('Epoch: [{0}][{1}/{2}]----'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})----'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})----'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})'.format(epoch, i, len(train_loader),
                                                                    batch_time=batch_time,
                                                                    data_time=data_time, loss=losses))

    del lr_imgs, hr_imgs, sr_imgs  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()