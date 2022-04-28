import torch
from numpy import loadtxt
from torch.utils.data import TensorDataset, DataLoader
import numpy as np
import torch.nn as nn
import sys
sys.path.insert(0, "./models")
from rnn import RNN

# First checking if GPU is available
train_on_gpu = torch.cuda.is_available()

if train_on_gpu:
    print('Training on GPU.')
else:
    print('No GPU available, training on CPU.')

# load model from checkpoint
model = torch.load('./checkpoints/model.pt')
print("successfully loaded the model.")

# load the test set
test_x = loadtxt('./data/preprocessed_testing_inputs.csv', delimiter=',')
test_y = loadtxt('./data/preprocessed_testing_labels.csv', delimiter=',')

# Get test data loss and accuracy
test_losses = []  # track loss
num_correct = 0
batch_size = 50

test_data = TensorDataset(torch.from_numpy(test_x), torch.from_numpy(test_y))
test_loader = DataLoader(test_data, shuffle=True, batch_size=batch_size)
criterion = nn.BCELoss()

# init hidden state
h = model.init_hidden(batch_size)

# keep the model in evaluation mode
model.eval()
# iterate over test data
for inputs, labels in test_loader:

    # Creating new variables for the hidden state, otherwise
    # we'd backprop through the entire training history
    h = tuple([each.data for each in h])

    if train_on_gpu:
        inputs, labels = inputs.cuda(), labels.cuda()

    # get predicted outputs
    output, h = model(inputs, h)

    # calculate loss
    test_loss = criterion(output.squeeze(), labels.float())
    test_losses.append(test_loss.item())

    # convert output probabilities to predicted class (0 or 1)
    pred = torch.round(output.squeeze())  # rounds to the nearest integer

    # compare predictions to true label
    correct_tensor = pred.eq(labels.float().view_as(pred))
    correct = np.squeeze(correct_tensor.numpy()) if not train_on_gpu else np.squeeze(correct_tensor.cpu().numpy())
    num_correct += np.sum(correct)

# -- stats! -- ##
# avg test loss
print("Test loss: {:.3f}".format(np.mean(test_losses)))

# accuracy over all test data
test_acc = num_correct / len(test_loader.dataset)
print("Test accuracy: {:.3f}".format(test_acc))