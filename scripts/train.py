import cv2
import torch


def train_epoch(model, optimizer, dataloader,criterion, scheduler, device, wandb):
    train_loss = 0.0
    model.train()
    for i, (full_image, image, labels) in enumerate(dataloader):
        inputs = image.to(device)
        labels = labels.to(device)
        
        outputs = model(inputs)
        
        loss = criterion(outputs, labels)
        # Compute loss
        # diff = outputs - labels
        # loss = torch.norm(diff, dim=1, p=2).square().mean(dim=0)
        
        train_loss += loss.item() * inputs.size(0)

        optimizer.zero_grad()

        loss.backward()
        
        optimizer.step()

        if scheduler:
            scheduler.step()
            wandb.log({'lr': scheduler.get_lr()[0], 'train_loss': loss.item()})

        else:
            wandb.log({'train_loss': loss.item()})
    return train_loss






       
    