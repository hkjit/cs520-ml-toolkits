import numpy as np
import pandas as pd
from numpy import loadtxt
import torch
from torch.utils.data import TensorDataset, DataLoader
import torch.nn as nn
import wandb
import sys
from comet_ml import Experiment
sys.path.insert(0, "./models")
from rnn import RNN


def train(args):
    # load array
    train_x = loadtxt('./data/preprocessed_training_inputs.csv', delimiter=',')
    train_y = loadtxt('./data/preprocessed_training_labels.csv', delimiter=',')
    val_x = loadtxt('./data/preprocessed_validation_inputs.csv', delimiter=',')
    val_y = loadtxt('./data/preprocessed_validation_labels.csv', delimiter=',')
    test_x = loadtxt('./data/preprocessed_testing_inputs.csv', delimiter=',')
    test_y = loadtxt('./data/preprocessed_testing_labels.csv', delimiter=',')
    batch_size = args.batch_size

    # print out the shapes of the resultant feature data
    print("\t\t\tFeature Shapes:")
    print("Train set: \t\t{}".format(train_x.shape),
          "\nValidation set: \t{}".format(val_x.shape),
          "\nTest set: \t\t{}".format(test_x.shape))

    # create Tensor datasets
    train_data = TensorDataset(torch.from_numpy(train_x), torch.from_numpy(train_y))
    valid_data = TensorDataset(torch.from_numpy(val_x), torch.from_numpy(val_y))
    test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))

    # make sure the SHUFFLE the training data
    train_loader = DataLoader(train_data, shuffle=True, batch_size=batch_size)
    valid_loader = DataLoader(valid_data, shuffle=True, batch_size=batch_size)
    test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)

    dataiter = iter(train_loader)
    sample_x, sample_y = dataiter.next()

    # First checking if GPU is available
    train_on_gpu = torch.cuda.is_available()

    if train_on_gpu:
        print('Training on GPU.')
    else:
        print('No GPU available, training on CPU.')

    # Adding comet experiment
    if args.use_comet:
        print("using comet")
        experiment = Experiment(
            api_key="",
            project_name="Twitter Airline sentiment analysis",
            workspace="",
        )
        experiment.add_tag('pytorch')
        experiment.log_parameters(
            {
                "learning_rate": args.lr,
                "epochs": args.epochs,
                "batch_size": args.batch_size
            }
        )
        experiment.train()

    # Instantiate the model w/ hyperparams
    vocab_size = 29237  # +1 for the 0 padding + our word tokens
    output_size = 1
    embedding_dim = 200
    hidden_dim = 128
    n_layers = 2

    model = RNN(vocab_size, output_size, embedding_dim, hidden_dim, n_layers, train_on_gpu=train_on_gpu)

    # loss and optimization functions
    lr = args.lr

    criterion = nn.BCELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    # training params
    epochs = args.epochs
    counter = 0
    print_every = 100
    clip = 5  # gradient clipping
    checkpoint_freq = 5

    # move model to GPU, if available
    if train_on_gpu:
        model.cuda()

    # keep the model in training mode.
    model.train()
    # train for some number of epochs
    for e in range(epochs):
        # initialize hidden state
        h = model.init_hidden(batch_size)

        # batch loop
        for inputs, labels in train_loader:
            counter += 1

            if (train_on_gpu):
                inputs, labels = inputs.cuda(), labels.cuda()

            # Creating new variables for the hidden state, otherwise
            # we'd backprop through the entire training history
            h = tuple([each.data for each in h])

            # zero accumulated gradients
            model.zero_grad()

            # get the output from the model
            output, h = model(inputs, h)

            # calculate the loss and perform backprop
            loss = criterion(output.squeeze(), labels.float())
            loss.backward()
            # `clip_grad_norm` helps prevent the exploding gradient problem in RNNs / LSTMs.
            nn.utils.clip_grad_norm_(model.parameters(), clip)
            optimizer.step()

            # loss stats
            if counter % print_every == 0:
                # Get validation loss
                val_h = model.init_hidden(batch_size)
                val_losses = []
                model.eval()
                for inputs, labels in valid_loader:

                    # Creating new variables for the hidden state, otherwise
                    # we'd backprop through the entire training history
                    val_h = tuple([each.data for each in val_h])

                    if (train_on_gpu):
                        inputs, labels = inputs.cuda(), labels.cuda()

                    output, val_h = model(inputs, val_h)
                    val_loss = criterion(output.squeeze(), labels.float())

                    val_losses.append(val_loss.item())

                model.train()
                # Logging parameters to Wandb
                if args.use_wandb:
                    wandb.log({"train loss": loss})
                    wandb.log({"validation loss": np.mean(val_losses)})
                    wandb.log({"epoch": e + 1})
                    wandb.log({"step": counter})
                # Logging parameters to comet
                if args.use_comet:
                    experiment.log_metric("training loss", loss, step=counter)
                    experiment.log_metric("validation loss", np.mean(val_losses), step=counter)
                print("Epoch: {}/{}...".format(e + 1, epochs),
                      "Step: {}...".format(counter),
                      "Loss: {:.6f}...".format(loss.item()),
                      "Val Loss: {:.6f}".format(np.mean(val_losses)))
        # Best Practice: Save checkpoints after every checkpoint_freq
        if (e + 1) % checkpoint_freq == 0:
            print("Saving the checkpoint..")
            torch.save(model, f'{args.checkpoint_dir}/model.pt')
