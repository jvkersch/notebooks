from collections import Counter
from itertools import product
import time

import pandas as pd

import torch
from torch import nn, optim
from torchmetrics import MatthewsCorrCoef


K = 4  # Kmer length
NREC = None  # Number of records to train on, or None for whole dataset
RANK = "Family"  # Rank to use for classification

DEVICE = torch.device("mps")


def load_data(fname, rank, retain=None):
    df = pd.read_csv(fname)
    if retain is not None:
        df = df.iloc[:retain]

    sequences = df["Sequence"].to_numpy()
    labels = df[rank].to_numpy()
    return sequences, labels


def encode_labels(*label_sets):
    seen = {}
    counter = 0
    cat_label_sets = []
    for label_set in label_sets:
        cat_labels = []
        for label in label_set:
            if label not in seen:
                seen[label] = counter
                counter += 1
            cat_labels.append(seen[label])
        cat_label_sets.append(cat_labels)

    label_map = {id_: label for label, id_ in seen.items()}
    return label_map, cat_label_sets


def all_kmers(k):
    return ["".join(letters) for letters in product(*k*("ACGT",))]


def kmerize(seq, k):
    counts = Counter(seq[i:i+k] for i in range(len(seq) - k + 1))
    spectrum = [counts[kmer] for kmer in all_kmers(k)]
    return spectrum


def to_categorical(labels):
    seen = {}
    counter = 0
    cat_labels = []
    for label in labels:
        if label not in seen:
            seen[label] = counter
            counter += 1
        cat_labels.append(seen[label])

    label_map = {id_: label for label, id_ in seen.items()}
    return cat_labels, label_map


def create_dataset(sequences):
    dataset = torch.tensor([kmerize(seq, K) for seq in sequences]) / 255
    dataset = dataset.reshape(dataset.shape[0], 1, dataset.shape[1])
    return dataset


def create_network(k, nclasses):
    # dense_input = 10*((4**k - 4)//2 - 4)//2
    dense_input = 4**k
    return nn.Sequential(
        # nn.Conv1d(1, 5, kernel_size=5),
        # nn.ReLU(),
        # nn.MaxPool1d(2),
        # nn.Conv1d(5, 10, kernel_size=5),
        # nn.ReLU(),
        # nn.MaxPool1d(2),
        nn.Flatten(),
        nn.Linear(dense_input, 500),
        # nn.Dropout(0.5),
        nn.Tanh(),
        nn.Linear(500, nclasses),
    ).to(device=DEVICE)


def train_network(network, train_loader, val_loader, n_epochs=100, learning_rate=1e-2):

    loss_fn = nn.CrossEntropyLoss()
    optimizer = optim.SGD(network.parameters(), lr=learning_rate, momentum=0.9)

    train_losses = []
    validation_losses = []

    network.train()
    for epoch in range(n_epochs):
        running_loss = 0
        nbatches = 0
        start = time.time()

        for seqs, labels in train_loader:
            seqs = seqs.to(device=DEVICE)
            labels = labels.to(device=DEVICE)

            outputs = network(seqs)
            loss = loss_fn(outputs, labels)

            running_loss += loss.item()
            nbatches += 1

            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        with torch.no_grad():
            running_val_loss = 0
            n_val_batches = 0

            for seqs, labels in val_loader:
                seqs = seqs.to(device=DEVICE)
                labels = labels.to(device=DEVICE)

                outputs = network(seqs)
                loss = loss_fn(outputs, labels)

                running_val_loss += loss.item()
                n_val_batches += 1

        avg_loss = running_loss / nbatches
        avg_val_loss = running_val_loss / n_val_batches

        train_losses.append(avg_loss)
        validation_losses.append(avg_val_loss)

        delta = time.time() - start
        print(f"Epoch: {epoch}, train loss: {avg_loss}, "
              f"validation loss: {avg_val_loss}, time: {delta:.02f}s")

    return network, train_losses, validation_losses


if __name__ == "__main__":
    print("Loading data")
    sequences, labels = load_data("df_train_0.csv", RANK, NREC)
    val_sequences, val_labels = load_data("df_val_0.csv", RANK)
    print(f"Dataset size: {len(sequences)}")
    print(f"Dataset size (validation): {len(val_sequences)}")

    print("Preparing data")
    label_map, (labels, val_labels) = encode_labels(labels, val_labels)
    n_classes = max(label_map) + 1

    train_kmers = create_dataset(sequences)
    val_kmers = create_dataset(val_sequences)

    train = [(s, label) for s, label in zip(train_kmers, labels)]
    train_loader = torch.utils.data.DataLoader(
        train, batch_size=10, shuffle=True)
    val = [(s, label) for s, label in zip(val_kmers, labels)]
    val_loader = torch.utils.data.DataLoader(
        train, batch_size=10, shuffle=False)

    print(f"Labels: {n_classes}")
    print(f"Training spectra: {train_kmers.shape}")
    print(f"Validation spectra: {val_kmers.shape}")

    network = create_network(K, n_classes)
    nparams = sum(p.numel() for p in network.parameters())
    print(f"Network parameters: {nparams}")

    print("Training network")
    network, train_losses, val_losses = train_network(
        network, train_loader, val_loader, n_epochs=120
    )

    # FIXME: for some reason, MCC doesn't work on the GPU
    network_cpu = network.to(torch.device("cpu"))
    print("Accuracy")
    mcc = MatthewsCorrCoef(num_classes=n_classes)
    with torch.no_grad():
        network.eval()
        for seqs, labels in val_loader:
            outputs = network_cpu(seqs).to(torch.device("cpu"))
            _, preds = torch.max(outputs, dim=1)
            mcc.update(preds, labels)
    print(f"MCC: {mcc.compute()}")

    print("Saving trained network")
    # torch.save(network.state_dict(), "cnn.pt")
    torch.save(network.state_dict(), "mlp2.pt")
