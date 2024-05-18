import numpy as np
from pathlib import Path
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_wine
import torch
from typing import Tuple

_DEVICE = torch.device("cuda:0")


def create_wine_dataset(n_features: int, test_size: float) \
        -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
    wine = load_wine()

    X_train, X_test, y_train, y_test = train_test_split(
        wine.data[:, :n_features],
        wine.target,
        test_size=test_size,
        shuffle=True)

    return (
        torch.FloatTensor(X_train).to(_DEVICE),
        torch.FloatTensor(X_test).to(_DEVICE),
        torch.LongTensor(y_train).to(_DEVICE),
        torch.LongTensor(y_test).to(_DEVICE),
    )


class WineNet(torch.nn.Module):
    def __init__(self, n_input, n_output):
        n_internal = 13
        super(WineNet, self).__init__()
        self.fc1 = torch.nn.Linear(n_input, n_internal)
        self.activ1 = torch.nn.ReLU()
        self.fc2 = torch.nn.Linear(n_internal, n_internal)
        self.activ2 = torch.nn.ReLU()
        self.fc3 = torch.nn.Linear(n_internal, n_output)
        self.sm = torch.nn.Softmax(dim=0)

    def forward(self, x):
        x = self.fc1(x)
        x = self.activ1(x)
        x = self.fc2(x)
        x = self.activ2(x)
        x = self.fc3(x)
        return x

    def inference(self, x):
        x = self.forward(x)
        x = self.sm(x)
        return x

    @staticmethod
    def load_net(file: Path, n_input: int, n_output: int):
        wine_net = WineNet(n_input, n_output)

        if file.exists():
            wine_net.load_state_dict(torch.load(file))
            wine_net.eval()
        else:
            raise FileExistsError(file)

        wine_net.to(torch.device('cuda'))
        return wine_net


def test_learning(tmp_file: Path = Path("./tmp/model.weights")):
    tmp_file.parents[0].mkdir(exist_ok=True)

    X_train, X_test, y_train, y_test = create_wine_dataset(13, 0.3)
    wine_net = WineNet(13, 3)
    try:
        if tmp_file.exists():
            wine_net.load_state_dict(torch.load(tmp_file))
            wine_net.eval()
    except RuntimeError:
        print("weights not loaded")
    wine_net = wine_net.to(_DEVICE)
    loss = torch.nn.CrossEntropyLoss().to(_DEVICE)
    optimizer = torch.optim.Adam(wine_net.parameters(), lr=1.0e-2)

    batch_size = 100

    for epoch in range(2000):
        order = np.random.permutation(len(X_train))
        for start_index in range(0, len(X_train), batch_size):
            optimizer.zero_grad()

            batch_indexes = order[start_index:start_index + batch_size]

            x_batch = X_train[batch_indexes].to(_DEVICE)
            y_batch = y_train[batch_indexes].to(_DEVICE)

            preds = wine_net.forward(x_batch)

            loss_value = loss(preds, y_batch)
            loss_value.backward()

            optimizer.step()

        if epoch % 10 == 0:
            test_preds = wine_net.forward(X_test)
            test_preds = test_preds.argmax(dim=1)

        test_preds = test_preds.cpu()
        yy = y_test.cpu()
        print(wine_net.fc1.in_features, np.asarray((test_preds == yy).float().mean()))
    torch.save(wine_net.state_dict(), tmp_file)


if __name__ == "__main__":
    test_learning()
