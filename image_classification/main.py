import torch
from argparse import ArgumentParser
from train import train
from model import Model
from dataloader import getdataloader
import wandb

def main(args):
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    train_loader, val_loader, test_loader = getdataloader(args)
    if args.use_wandb:
        print("using wandb")
        wandb.init(project="MNIST", entity="520-helloworld")
        wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size
        }
    train(model, optimizer, train_loader, val_loader, args)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--use_wandb', help='Use cuda?', action='store_true')
    parser.add_argument('--batch_size', help='Batch size', type=int, default=8)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--checkpoint_dir',help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    main(parser.parse_args())
