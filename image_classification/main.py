import torch
from argparse import ArgumentParser
from train import train
from model import Model
from dataloader import getdataloader
import wandb
import torch.optim as optim

def build_optimizer(network, optimizer, learning_rate):
    """
    Returns an optimizer object for training.
    """
    if optimizer == "sgd":
        optimizer = optim.SGD(network.parameters(),
                              lr=learning_rate, momentum=0.9)
    elif optimizer == "adam":
        optimizer = optim.Adam(network.parameters(),
                               lr=learning_rate)
    return optimizer

def sweep(config = None):
    """
    Setup the wandb sweeping for hyper-parameters and train
    """
    with wandb.init(config=config, project="MNIST", entity="520-helloworld", name="MNIST digit recognition"):
        config = wandb.config
        model = Model()
        optimizer = build_optimizer(model, config.optimizer, config.lr)
        train_loader, val_loader, test_loader = getdataloader(config)
        use_cuda = False
        train(model, optimizer, train_loader, val_loader, use_cuda, config)    

def main(args):

    # Define hyper-parameters sweep config
    if args.sweep:
        sweep_config = {
            'method': 'random'
        }
        metric = {
            'name': 'train loss',
            'goal': 'minimize'   
            }
        parameters_dict = {
            'optimizer': {
                'values': ['adam', 'sgd']
                },
            'lr': {
                # a flat distribution between 0 and 0.1
                'distribution': 'uniform',
                'min': 0,
                'max': 0.1
            },
            'batch_size': {
                # integers between 32 and 256
                # with evenly-distributed logarithms 
                'distribution': 'q_log_uniform_values',
                'q': 8,
                'min': 32,
                'max': 256,
            },
            "epochs" : {
                'value': 3
            },
            "use_comet" : {
                'value': False
            },
            "use_wandb" : {
                'value': True
            }
            
        }
        sweep_config['parameters'] = parameters_dict
        sweep_config['metric'] = metric
        sweep_id = wandb.sweep(sweep_config, project="MNIST", entity="520-helloworld")
        wandb.agent(sweep_id, sweep, count=1)
        
        
    else:
        # Define the model
        model = Model()
        # Get an optimizeer
        optimizer = build_optimizer(model, args.optimizer, args.lr)
        train_loader, val_loader, test_loader = getdataloader(args)
        # Setup wandb for training
        if args.use_wandb:
            print("using wandb")
            wandb.init(project="MNIST", entity="520-helloworld", name="MNIST digit recognition")
            wandb.config = {
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size
            }
        use_cuda = False
        # Check for CUDA
        if args.use_cuda:
            use_cuda = True
        # Train the model with defined hyper-parameters
        train(model, optimizer, train_loader, val_loader, use_cuda, args)
    

if __name__ == '__main__':
    parser = ArgumentParser()
    parser.add_argument('--cuda', help='Use cuda?', action='store_true')
    parser.add_argument('--optimizer', help='Use cuda?', default='adam')
    parser.add_argument('--use_wandb', help='Use cuda?', action='store_true')
    parser.add_argument('--sweep', help='Run sweep in wandb?', action='store_true')
    parser.add_argument('--use_comet', help='Use cuda?', action='store_true')
    parser.add_argument('--batch_size', help='Batch size', type=int, default=8)
    parser.add_argument('--epochs', help='Number of epochs to run', type=int, default=10)
    parser.add_argument('--lr', help='learning rate', type=float, default=0.001)
    parser.add_argument('--checkpoint_dir',help='Checkpoint directory', default='checkpoints')
    parser.add_argument('--stop_after', help='Number of batches to stop after', default=None, type=int)
    main(parser.parse_args())
