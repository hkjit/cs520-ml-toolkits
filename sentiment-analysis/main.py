from argparse import ArgumentParser
import wandb
from train.train import train
from test.test import test


def main(args):
    if args.use_wandb:
        print("using wandb")
        wandb.init(project="Twitter US Airline Sentiment Analysis")
        wandb.config = {
            "learning_rate": args.lr,
            "epochs": args.epochs,
            "batch_size": args.batch_size,
            "train_mode": args.train,
            "test_mode": args.test,
            "checkpoint_dir": args.checkpoint_dir
        }
    if args.train:
        print("Starting the training process...")
        train(args)
    if args.test:
        print("Starting the testing process...")
        test(args)


if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--use_wandb', help='Use wandb?', action='store_true')
    parser.add_argument('--batch_size', help='Batch size', type=int, default=50)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--checkpoint_dir',help='Checkpoint directory', default='./checkpoints')
    parser.add_argument('--train', help='Training mode?', action='store_true')
    parser.add_argument('--test', help='Testing mode?', action='store_true')
    main(parser.parse_args())