import torch
from argparse import ArgumentParser
from train import train
from model import Model
from dataloader import getdataloader

def main(args):
    model = Model()
    optimizer = torch.optim.Adam(model.parameters(), lr = args.lr)
    train_loader, val_loader, test_loader = getdataloader(args)
    train(model, optimizer, train_loader, val_loader, args)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--batch_size', help='Batch size', type=int, default=8)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--checkpoint_dir',help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    main(parser.parse_args())
