import torch
import numpy as np
from tqdm import tqdm

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
                optimizer.zero_grad()
                output = model(images)
                loss = model.criterion(output, labels)
                loss.backward()
                optimizer.step()
                train_loss += loss.item()
        """
        VALIDATION PHASE
        """
        model.eval()
        for images, labels in val_loader:
            if use_cuda and torch.cuda.is_available():
                images, labels = images.cuda(), labels.cuda()
            output = model(images)
            loss = model.criterion(output, labels)
            valid_loss += loss.item()

        train_loss = train_loss/len(train_loader)
        valid_loss = valid_loss/len(val_loader)

