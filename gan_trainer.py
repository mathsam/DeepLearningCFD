import time
import os
import torch.backends.cudnn as cudnn
from torch import nn
from ml_models import Generator, Discriminator
from datasets import SRDataset
from utils import *

exp_name = "GAN_Sep8_exp1"
from torch.utils.tensorboard import SummaryWriter
summary_writer = SummaryWriter(log_dir=exp_name)


# Data parameters
data_folder = './exp2'  # folder with JSON data files
scaling_factor = 4  # the scaling factor for the generator; the input LR images will be downsampled from the target HR images by this factor
save_dir = "../azblob/gan_model"
os.makedirs(save_dir, exist_ok=True)

# Generator parameters
num_layers_g = 16  # number of residual blocks
srresnet_checkpoint = "/home/juchai/azblob/sres_model/epoch_30_checkpoint_srnet.pth.tar"
large_kernel_size_g = 15  # kernel size of the first and last convolutions which transform the inputs and outputs
small_kernel_size_g = 5  # kernel size of all convolutions in-between, i.e. those in the residual and subpixel convolutional blocks

# Discriminator parameters
kernel_size_d = 5  # kernel size in all convolutional blocks
n_channels_d = 64  # number of output channels in the first convolutional block, after which it is doubled in every 2nd block thereafter
n_blocks_d = 8  # number of convolutional blocks
fc_size_d = 1024  # size of the first fully connected layer

# Learning parameters
checkpoint = "/home/juchai/azblob/an_model/epoch100_checkpoint_srgan.pth.tar"
batch_size = 24  # batch size
start_epoch = 0  # start at this epoch
iterations = 2e5  # number of training iterations
workers = 8  # number of workers for loading data in the DataLoader
beta = 0.01  # the coefficient to weight the adversarial loss in the perceptual loss
print_freq = 5  # print training status once every __ batches
lr = 1e-4  # learning rate
grad_clip = None  # clip if gradients are exploding

# Default device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

cudnn.benchmark = True


def main():
    """
    Training.
    """
    global start_epoch, epoch, checkpoint, srresnet_checkpoint

    # Initialize model or load checkpoint
    if checkpoint is None:
        # Generator
        generator = Generator(large_kernel_size=large_kernel_size_g,
                              small_kernel_size=small_kernel_size_g,
                              num_layers=num_layers_g,
                              scaling_factor=scaling_factor)

        # Initialize generator network with pretrained SRResNet
        generator.initialize_with_srresnet(srresnet_checkpoint=srresnet_checkpoint)

        # Initialize generator's optimizer
        optimizer_g = torch.optim.Adam(params=filter(lambda p: p.requires_grad, generator.parameters()),
                                       lr=lr)

        # Discriminator
        discriminator = Discriminator(kernel_size=kernel_size_d,
                                      n_channels=n_channels_d,
                                      n_blocks=n_blocks_d,
                                      fc_size=fc_size_d)

        # Initialize discriminator's optimizer
        optimizer_d = torch.optim.Adam(params=filter(lambda p: p.requires_grad, discriminator.parameters()),
                                       lr=lr)

    else:
        checkpoint = torch.load(checkpoint)
        start_epoch = checkpoint['epoch'] + 1
        generator = checkpoint['generator']
        discriminator = checkpoint['discriminator']
        optimizer_g = checkpoint['optimizer_g']
        optimizer_d = checkpoint['optimizer_d']
        print("\nLoaded checkpoint from epoch %d.\n" % (checkpoint['epoch'] + 1))


    # Loss functions
    content_loss_criterion = nn.MSELoss()
    adversarial_loss_criterion = nn.BCEWithLogitsLoss()

    if torch.cuda.device_count() > 1:
        print("Let's use", torch.cuda.device_count(), "GPUs!")
        generator = nn.DataParallel(generator)
        discriminator = nn.DataParallel(discriminator)
    # Move to default device
    generator = generator.to(device)
    discriminator = discriminator.to(device)

    content_loss_criterion = content_loss_criterion.to(device)
    adversarial_loss_criterion = adversarial_loss_criterion.to(device)

    # Custom dataloaders
    train_dataset = SRDataset(data_folder,
                              scaling_factor=scaling_factor)
    train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=batch_size, shuffle=True, num_workers=workers,
                                               pin_memory=True)

    # Total number of epochs to train for
    epochs = int(iterations // len(train_loader) + 1)

    # Epochs
    for epoch in range(start_epoch, epochs):

        # At the halfway point, reduce learning rate to a tenth
        if epoch == int((iterations / 2) // len(train_loader) + 1):
            adjust_learning_rate(optimizer_g, 0.1)
            adjust_learning_rate(optimizer_d, 0.1)

        # One epoch's training
        train(train_loader=train_loader,
              generator=generator,
              discriminator=discriminator,
              content_loss_criterion=content_loss_criterion,
              adversarial_loss_criterion=adversarial_loss_criterion,
              optimizer_g=optimizer_g,
              optimizer_d=optimizer_d,
              epoch=epoch)

        # Save checkpoint
        if isinstance(generator, torch.nn.parallel.DataParallel):
            torch.save({'epoch': epoch,
                        'generator': generator.module,
                        'discriminator': discriminator.module,
                        'optimizer_g': optimizer_g,
                        'optimizer_d': optimizer_d},
                        os.path.join(save_dir, f'epoch{epoch}_checkpoint_srgan.pth.tar'))
        else:
            torch.save({'epoch': epoch,
                        'generator': generator,
                        'discriminator': discriminator,
                        'optimizer_g': optimizer_g,
                        'optimizer_d': optimizer_d},
                        os.path.join(save_dir, f'epoch{epoch}_checkpoint_srgan.pth.tar'))


def train(train_loader, generator, discriminator, content_loss_criterion, adversarial_loss_criterion,
          optimizer_g, optimizer_d, epoch):
    """
    One epoch's training.
    :param train_loader: train dataloader
    :param generator: generator
    :param discriminator: discriminator
    :param truncated_vgg19: truncated VGG19 network
    :param content_loss_criterion: content loss function (Mean Squared-Error loss)
    :param adversarial_loss_criterion: adversarial loss function (Binary Cross-Entropy loss)
    :param optimizer_g: optimizer for the generator
    :param optimizer_d: optimizer for the discriminator
    :param epoch: epoch number
    """
    # Set to train mode
    generator.train()
    discriminator.train()  # training mode enables batch normalization

    batch_time = AverageMeter()  # forward prop. + back prop. time
    data_time = AverageMeter()  # data loading time
    losses_c = AverageMeter()  # content loss
    losses_a = AverageMeter()  # adversarial loss in the generator
    losses_d = AverageMeter()  # adversarial loss in the discriminator

    start = time.time()

    # Batches
    for i, (lr_imgs, hr_imgs) in enumerate(train_loader):
        data_time.update(time.time() - start)

        # Move to default device
        lr_imgs = lr_imgs.to(device)  # (batch_size (N), 3, 24, 24), imagenet-normed
        hr_imgs = hr_imgs.to(device)  # (batch_size (N), 3, 96, 96), imagenet-normed

        # GENERATOR UPDATE

        # Generate
        sr_imgs = generator(lr_imgs)  # (N, 3, 96, 96), in [-1, 1]

        # Discriminate super-resolved (SR) images
        sr_discriminated = discriminator(sr_imgs)  # (N)

        # Calculate the Perceptual loss
        content_loss = content_loss_criterion(sr_imgs, hr_imgs)
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.ones_like(sr_discriminated))
        perceptual_loss = content_loss + beta * adversarial_loss

        # Back-prop.
        optimizer_g.zero_grad()
        perceptual_loss.backward()


        # Update generator
        optimizer_g.step()

        # Keep track of loss
        losses_c.update(content_loss.item(), lr_imgs.size(0))
        losses_a.update(adversarial_loss.item(), lr_imgs.size(0))

        # DISCRIMINATOR UPDATE

        # Discriminate super-resolution (SR) and high-resolution (HR) images
        hr_discriminated = discriminator(hr_imgs)
        sr_discriminated = discriminator(sr_imgs.detach())
        # But didn't we already discriminate the SR images earlier, before updating the generator (G)? Why not just use that here?
        # Because, if we used that, we'd be back-propagating (finding gradients) over the G too when backward() is called
        # It's actually faster to detach the SR images from the G and forward-prop again, than to back-prop. over the G unnecessarily
        # See FAQ section in the tutorial

        # Binary Cross-Entropy loss
        adversarial_loss = adversarial_loss_criterion(sr_discriminated, torch.zeros_like(sr_discriminated)) + \
                           adversarial_loss_criterion(hr_discriminated, torch.ones_like(hr_discriminated))

        # Back-prop.
        optimizer_d.zero_grad()
        adversarial_loss.backward()


        # Update discriminator
        optimizer_d.step()

        # Keep track of loss
        losses_d.update(adversarial_loss.item(), hr_imgs.size(0))

        # Keep track of batch times
        batch_time.update(time.time() - start)

        # Reset start time
        start = time.time()

        # Print status
        if i % print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]--'
                  'Batch Time {batch_time.val:.3f} ({batch_time.avg:.3f})--'
                  'Data Time {data_time.val:.3f} ({data_time.avg:.3f})--'
                  'Cont. Loss {loss_c.val:.4f} ({loss_c.avg:.4f})--'
                  'Adv. Loss {loss_a.val:.4f} ({loss_a.avg:.4f})--'
                  'Disc. Loss {loss_d.val:.4f} ({loss_d.avg:.4f})'.format(epoch,
                                                                          i,
                                                                          len(train_loader),
                                                                          batch_time=batch_time,
                                                                          data_time=data_time,
                                                                          loss_c=losses_c,
                                                                          loss_a=losses_a,
                                                                          loss_d=losses_d))
        
        global_step = i + epoch * len(train_loader)
        summary_writer.add_scalar(os.path.join(exp_name, "cont_loss"), losses_c.val, global_step)
        summary_writer.add_scalar(os.path.join(exp_name, "adv_loss"), losses_a.val, global_step)
        summary_writer.add_scalar(os.path.join(exp_name, "desc_loss"), losses_d.val, global_step)

    del lr_imgs, hr_imgs, sr_imgs, hr_discriminated, sr_discriminated  # free some memory since their histories may be stored


if __name__ == '__main__':
    main()