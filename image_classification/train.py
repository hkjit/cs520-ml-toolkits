import torch
import numpy as np
from tqdm import tqdm
import wandb
from comet_ml import Experiment

def train(model, optimizer, train_loader, val_loader, use_cuda, args):
    if use_cuda and torch.cuda.is_available():
        model.cuda()

    if args.use_comet:
        print("using comet")
        experiment = Experiment(
            api_key="NhCJtV4broSjyc6xXMzrvozJ8",
            project_name="MNIST digit recognition",
            workspace="amandeepc",
        )
        experiment.add_tag('pytorch')
        experiment.log_parameters(
            {
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size
            }
        )

    if args.use_comet:
        experiment.train()

    if args.use_wandb:
        wandb.watch(model, log="all", log_freq=10)

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
        elif args.use_comet:
            experiment.log_metric("training loss", train_loss, step=epoch)

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
        elif args.use_comet:
            experiment.log_metric("validation loss", valid_loss, step=epoch)

        train_loss = train_loss/len(train_loader)
        valid_loss = valid_loss/len(val_loader)

