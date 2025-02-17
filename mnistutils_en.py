import random
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

data_folder = "data"                    # the folder in which data will be downloaded

def load_data() -> tuple:
    ''' Load training and test set. '''
    train_data = datasets.MNIST(root=data_folder, train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root=data_folder, train=False, download=True, transform=ToTensor())
    return train_data, test_data


def show_images(test_data) -> None:
    ''' Randomly display 10 images. '''
    plt.figure(figsize=(10, 4))
    for i in range(1, 11):
        X, y = test_data[random.randint(0, len(test_data))]
        plt.subplot(1, 10, i)
        plt.title(f"corr: {y}")
        plt.axis("off")
        plt.imshow(X.squeeze(), cmap="gray")
    plt.show()


def train_loop(dataset, batch_size, num_batches, network, loss_fn, optimizer) -> float:
    ''' Basic training. '''
    network.train()                     # set training mode
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    for i in range(num_batches):        # for each batch:
        X, y = next(dataloader)         # read training data: images (X) and labels (y) 
        pred = network(X)               # compute prediction and error
        loss = loss_fn(pred, y)
        loss.backward()                 # compute backpropagation
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()                  # return error


def test_loop(dataset, batch_size, network) -> float:
    ''' Basic inference. '''
    network.eval()                              # set test mode
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    with torch.no_grad():                       # don't compute gradients
        X, y = next(dataloader)                 # read test data: images (X) and labels (y) 
        pred = network(X).argmax(1)
        correct = (pred == y).type(torch.float).sum().item()
    return correct


def test_with_prob(test_data, network) -> None:
    ''' Inference on 1 random image and display of result. '''
    network.eval()                              # set test mode
    X, y = test_data[random.randint(0, len(test_data))]
    X = X.unsqueeze(0)
    with torch.no_grad():                       # non calcolare i gradienti
        logits = network(X)                     # don't compute gradients
    pred_probab = nn.Softmax(dim=1)(logits)     # convert prediction in a probability distribution
    pred_probab_np = pred_probab.cpu().detach().numpy()[0]
    y_pred = pred_probab.argmax(1)              # find the most probable class
    classes = np.arange(10)
    fig, axs = plt.subplots(1, 2, figsize=(8,3), gridspec_kw={'width_ratios':[1,4]})
    fig.tight_layout(pad=5.0)
    axs[0].set_title(f"pred: {y_pred.item()}\ncorr: {y}")
    axs[0].axis("off")
    axs[0].imshow(X.squeeze(), cmap="gray")
    axs[1].bar(classes, pred_probab_np)
    axs[1].set_xticks(classes)
    axs[1].set_xlabel('Digit')
    axs[1].set_ylabel('Probability')
    axs[1].set_title('Probability distribution')
    plt.show()


def test_loop_with_trace(test_data, network) -> None:
    ''' Inference on 10 random image and display of result. '''
    network.eval()
    plt.figure(figsize=(10, 4))
    with torch.no_grad():
        for i in range(1, 11):
            X, y = test_data[random.randint(0, len(test_data))]
            X = X.unsqueeze(0)                  
            pred = network(X)
            plt.subplot(1, 10, i)
            plt.title(f"pred: {pred.argmax(1).item()}\ncorr: {y}")
            plt.axis("off")
            plt.imshow(X.squeeze(), cmap="gray")
    plt.show()