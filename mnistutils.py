import random
import torch
from torch import nn
from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import numpy as np
import matplotlib.pyplot as plt

data_folder = "data"                    # la cartella in cui sono mantenute le immagini

def load_data() -> tuple:
    ''' Carica i dati di addestramento e di test. '''
    train_data = datasets.MNIST(root=data_folder, train=True, download=True, transform=ToTensor())
    test_data = datasets.MNIST(root=data_folder, train=False, download=True, transform=ToTensor())
    return train_data, test_data


def load_batch(data, shuffle=False) -> tuple:
    ''' Carica un batch di dati di addestramento. '''
    dataloader = iter(DataLoader(data, batch_size=len(data), shuffle=shuffle))
    X, y = next(dataloader)
    return X, y


def show_images(test_data, random_indexes) -> None:
    ''' Visualizza n immagini scelte a caso. '''
    n = len(random_indexes)
    plt.figure(figsize=(n, 4))
    for i in range(1, n+1):
        X, y = test_data[random_indexes[i-1]]
        plt.subplot(1, n, i)
        plt.title(f"corr: {y}")
        plt.axis("off")
        plt.imshow(X.squeeze(), cmap="gray")
    plt.show()


def train_loop(dataset, batch_size, num_batches, network, loss_fn, optimizer) -> float:
    ''' Addestramento di base. '''
    network.train()                     # metti in modalità di addestramento
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    for i in range(num_batches):        # per ogni batch:
        X, y = next(dataloader)         # leggi i dati per l'addestramento: immagini (X) e cifre corrette (y) 
        pred = network(X)               # calcola la previsione e l'errore
        loss = loss_fn(pred, y)
        loss.backward()                 # esegui la backpropagation
        optimizer.step()
        optimizer.zero_grad()
    return loss.item()                  # restituisci l'errore


def test_loop(dataset, batch_size, network) -> float:
    ''' Inferenza di base. '''
    network.eval()                              # metti in modalità di test
    dataloader = iter(DataLoader(dataset, batch_size=batch_size, shuffle=True))
    with torch.no_grad():                       # non calcolare i gradienti
        X, y = next(dataloader)                 # leggi i dati per il test: immagini (X) e cifre corrette (y) 
        pred = network(X).argmax(1)
        correct = (pred == y).type(torch.float).sum().item()
    return correct


def test_with_prob(test_data, random_index, network) -> None:
    ''' Inferenza su un'immagine scelta a caso e visualizzazione del risultato. '''
    network.eval()                              # metti in modalità di test
    X, y = test_data[random_index]
    X = X.unsqueeze(0)                          # aggiunge la dimensione del batch (necessario per la parte convoluzionale, prima veniva fatto automaticamente da dataloader)
    with torch.no_grad():                       # non calcolare i gradienti
        logits = network(X)                     # calcola la previsione
    pred_probab = nn.Softmax(dim=1)(logits)     # converti la previsione in una distribuzione di probabilità
    y_pred = pred_probab.argmax(1)              # trova la classe più probabile         
    pred_probab_np = pred_probab.cpu().detach().numpy()[0]
    classes = np.arange(10)
    fig, axs = plt.subplots(1, 2, figsize=(8,3), gridspec_kw={'width_ratios':[1,4]})
    fig.tight_layout(pad=5.0)
    axs[0].set_title(f"prev: {y_pred.item()}\ncorr: {y}")
    axs[0].axis("off")
    axs[0].imshow(X.squeeze(), cmap="gray")
    axs[1].bar(classes, pred_probab_np)
    axs[1].set_xticks(classes)
    axs[1].set_xlabel('Cifra')
    axs[1].set_ylabel('Probabilità')
    axs[1].set_title('Distribuzione di probabilità')
    plt.show()


def test_loop_with_trace(test_data, random_indexes, network) -> None:
    ''' Inferenza su n immagini scelte a caso e visualizzazione del risultato. '''
    n = len(random_indexes)
    network.eval()
    plt.figure(figsize=(n, 4))
    with torch.no_grad():
        for i in range(1, n+1):
            X, y = test_data[random_indexes[i-1]]
            X = X.unsqueeze(0)                  
            pred = network(X)
            plt.subplot(1, n, i)
            color = "black" if pred.argmax(1).item() == y else "red"
            plt.title(f"prev: {pred.argmax(1).item()}\ncorr: {y}", color=color)
            plt.axis("off")
            plt.imshow(X.squeeze(), cmap="gray")
    plt.show()