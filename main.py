from dataset import CWRUDataset
from model import SirHongsCNN

from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report

import torch
from torch.utils.data import TensorDataset, DataLoader


def main():
    # options
    dataset_root = 'D:/datasets/CWRU-dataset'
    device = torch.device('cuda')
    epoch = 10
    batch_size = 100

    # load dataset
    dataset = CWRUDataset(dataset_root, slice_dim=10000)
    X, y = dataset[slice(None)]

    # preprocess:
    X = X.unsqueeze(1)

    # train-test split:
    X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=0.8)
    print(X_train.shape)
    print(X_test.shape)

    train_set = TensorDataset(X_train, y_train)
    test_set = TensorDataset(X_test, y_test)

    train_loader = DataLoader(train_set, batch_size=batch_size)
    test_loader = DataLoader(test_set, batch_size=batch_size)

    model = SirHongsCNN(num_classes=len(dataset.classes)).to(device=device)

    # train
    model.train()
    optim = torch.optim.Adam(model.parameters(), lr=1e-2)
    criterion = torch.nn.CrossEntropyLoss()
    for epoch_id in range(epoch):
        for batch_id, (X, y) in enumerate(train_loader):
            optim.zero_grad()
            X = X.to(device=device)
            y = y.to(device=device)
            y_pred = model(X)
            loss = criterion(y_pred, y)
            loss.backward()
            optim.step()
            del X, y
            print(f"[epoch: {epoch_id + 1}/{epoch} - batch: {batch_id + 1}/{len(train_loader)}]"
                  f" loss={loss.item():.03f}")

    # evaluate
    model.eval()
    y_pred = []
    for batch_id, (X, y) in enumerate(test_loader):
        X = X.to(device)
        y_pred.append(model(X).argmax(-1).cpu())
        del X, y
    y_pred = torch.cat(y_pred)

    report = classification_report(y_test, y_pred, target_names=dataset.classes)
    print(report)


if __name__ == '__main__':
    main()
