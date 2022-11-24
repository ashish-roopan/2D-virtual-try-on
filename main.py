import cv2
import random
import numpy as np
import torch.backends.cudnn as cudnn
import torch
import torch.optim as optim
from torchvision import models
import argparse
from models.fc import Fc
from dataloaders.dataloader import get_dataloader
from scripts.train import train_epoch
from scripts.validate import validate_epoch
from util.debug_disp import debug_disp
import wandb
wandb.init(project="cloth_project")

def deterministic(rank):
    torch.manual_seed(rank)
    torch.cuda.manual_seed(rank)
    np.random.seed(rank)
    random.seed(rank)
    cudnn.deterministic = True
    cudnn.benchmark = False

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_dir', type=str, default='./dataset/', help='data directory')
    parser.add_argument('--batch_size', type=int, default=1, help='batch size')
    parser.add_argument('--num_epochs', type=int, default=10, help='number of epochs')
    parser.add_argument('--lr', type=float, default=0.001, help='learning rate')
    parser.add_argument('--weight_decay', type=float, default=0.0005, help='weight decay')
    parser.add_argument('--momentum', type=float, default=0.9, help='momentum')
    parser.add_argument('--log_interval', type=int, default=10, help='log interval')
    parser.add_argument('--save_model', type=bool, default=True, help='save model')
    parser.add_argument('--load_model', type=bool, default=False, help='load_model')
    parser.add_argument('--model_path', type=str, default='checkpoints/model.pt', help='model path')
    parser.add_argument('--seed', type=int, default=42, help='random seed')
    parser.add_argument('--device', type=str, default='cuda:0', help='device')
    parser.add_argument('--debug', type=bool, default=False, help='debug')
    parser.add_argument('--num_images', type=int, default=3200, help='number of images to train')
    return parser.parse_args()
args = parse_args()

# Set random seed
deterministic(args.seed)


############# Prepare Data ############
data_dir = args.data_dir
train_dataloader = get_dataloader(data_dir + 'train_set/', batch_size=args.batch_size, num_images=args.num_images, split='train')
valid_dataloader = get_dataloader(data_dir + 'valid_set/', batch_size=args.batch_size, split='valid')

print('Data loaded')
print('Number of training images: ', len(train_dataloader))
print('Number of validation images: ', len(valid_dataloader))
print()

############ Prepare Model ##########
model = models.resnet18(pretrained=True)
model.fc = Fc(in_features=model.fc.in_features, out_features=4)
model = model.to(args.device)

############ HYPERPARAMETERS ############
best_valid_loss = 10
lr = 0.00001
momentum = 0.9
weight_decay = 0.01
epochs = args.num_epochs
optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)
scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader), epochs=epochs)

if args.load_model:
    checkpoint = torch.load(args.model_path)
    model.load_state_dict(checkpoint['model'])
    # optimizer.load_state_dict(checkpoint['optimizer'])
    # scheduler = checkpoint['lr_sched']
    start_epoch = checkpoint['epoch']
    # best_valid_loss = checkpoint['best_valid_loss']
    best_valid_loss = 0.12
    lr = scheduler.get_last_lr()


wandb.config = {
    "learning_rate": lr,
    "epochs": epochs,
    "batch_size": args.batch_size,
    "weight_decay" : 0.01,
    "momentum" : 0.9,
    "optimizer" : optimizer,
    "scheduler" : scheduler
}

############ TRAINING ############
for epoch in range(epochs):
    train_loss = train_epoch(model, optimizer, train_dataloader, scheduler, args.device, wandb)
    val_loss = validate_epoch(model, valid_dataloader, args.device, wandb)

    avg_train_loss = train_loss / len(train_dataloader)
    avg_val_loss = val_loss / len(valid_dataloader)

    print('Epoch: {} \tTraining Loss: {:.6f} \tValidation Loss: {:.6f} '.format(start_epoch + epoch, avg_train_loss, avg_val_loss))

    if args.debug and epoch % 1 == 0:
        train_debug_disp = debug_disp(model, train_dataloader, args.device)
        valid_debug_disp = debug_disp(model, valid_dataloader, args.device)

        cv2.imshow('train', train_debug_disp)
        cv2.imshow('valid', valid_debug_disp)
        cv2.waitKey(1)

    ##########  Save Model ##########
    if args.save_model:
        model_name = model.__class__.__name__
        if avg_val_loss < best_valid_loss:
            model_path = f'{args.model_path.split("/")[0]}/{model_name}__{epoch}__{avg_train_loss:.3f}__{avg_val_loss:.3f}.pt'
            print(f'saving model: {model_path}')
            checkpoint = { 
                'epoch': start_epoch + epoch,
                'model': model.state_dict(),
                'optimizer': optimizer.state_dict(),
                'lr_sched': scheduler,
                'best_valid_loss': best_valid_loss,
                }
            torch.save(checkpoint, model_path)
            val_loss_threshold = avg_val_loss