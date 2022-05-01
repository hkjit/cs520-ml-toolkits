import torch
import numpy as np
from tqdm import tqdm
import wandb

def train(model, optimizer, train_loader, val_loader, args):
    use_cuda = False
    if use_cuda and torch.cuda.is_available():
        model.cuda()

    for epoch in range(args.epochs):
        train_loss = 0.0
        valid_loss = 0.0
        """
        TRAINING PHASE
        """
        print("started training phase")
        model.train()
        with tqdm(desc='Training', total=len(train_loader)) as pbar:
            for images, labels in train_loader:
                if use_cuda and torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                pbar.update()
                optimizer.zero_grad()
                output = model(images)
                loss = model.criterion(output, labels)
                if args.use_wandb:
                    wandb.log({"train loss": loss})
                loss.backward()
                optimizer.step()
                train_loss += loss.item()

        if args.use_wandb:
            wandb.log({"training loss": train_loss, "epoch": epoch})

        """
        VALIDATION PHASE
        """
        model.eval()
        with tqdm(desc='Validating', total=len(val_loader)) as pbar:
            for images, labels in val_loader:
                if use_cuda and torch.cuda.is_available():
                    images, labels = images.cuda(), labels.cuda()
                pbar.update()
                output = model(images)
                loss = model.criterion(output, labels)
                if args.use_wandb:
                    wandb.log({"val loss": loss})
                valid_loss += loss.item()

        if args.use_wandb:
            wandb.log({"validation loss": valid_loss, "epoch": epoch})

        train_loss = train_loss/len(train_loader)
        valid_loss = valid_loss/len(val_loader)

