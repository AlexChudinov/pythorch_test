import torch
import matplotlib.pyplot as plt


class RegressionNet(torch.nn.Module):
    def __init__(self, n_hidden_neurons):
        super(RegressionNet, self).__init__()
        self.__input_layer = torch.nn.Linear(1, n_hidden_neurons)
        self.__activation = torch.nn.Tanh()
        self.__sub_layer_fc1 = torch.nn.Linear(n_hidden_neurons, n_hidden_neurons * 10)
        self.__sub_layer_fc2 = torch.nn.Linear(n_hidden_neurons * 10, n_hidden_neurons)
        self.__output_layer = torch.nn.Linear(n_hidden_neurons, 1)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x = self.__input_layer(x)
        x = self.__activation(x)
        x = self.__sub_layer_fc1(x)
        x = self.__activation(x)
        x = self.__sub_layer_fc2(x)
        x = self.__activation(x)
        return self.__output_layer(x)

    @staticmethod
    def loss_mse(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return ((y - y_pred) ** 2).sum()

    @staticmethod
    def loss_mae(y: torch.Tensor, y_pred: torch.Tensor) -> torch.Tensor:
        return torch.abs(y - y_pred).mean()

    def optimize_mse(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            lr: float = 0.01,
            n_steps: int = 2000):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(n_steps):
            optimizer.zero_grad()
            y_pred = self.forward(x_train)
            loss_val = self.loss_mse(y_pred, y_train)
            loss_val.backward()
            optimizer.step()

    def optimize_mae(
            self,
            x_train: torch.Tensor,
            y_train: torch.Tensor,
            lr: float = 0.001,
            n_steps: int = 2000):
        optimizer = torch.optim.Adam(self.parameters(), lr=lr)
        for _ in range(n_steps):
            optimizer.zero_grad()
            y_pred = self.forward(x_train)
            loss_val = self.loss_mae(y_pred, y_train)
            loss_val.backward()
            optimizer.step()

    def predict(self, x: torch.Tensor) -> torch.Tensor:
        return self.forward(x)


def predict(net: RegressionNet, x: torch.Tensor, y: torch.Tensor):
    y_pred = net.predict(x)
    plt.plot(x.numpy(), y.numpy(), label="Ground Truth")
    plt.plot(x.numpy(), y_pred.data.numpy(), c="r", label="prediction")
    plt.legend(loc="upper left")
    plt.xlabel("$x$")
    plt.ylabel("$y$")


def run_test():
    x_validation = torch.linspace(-10, 5, 100)
    y_validation = torch.pow(2, x_validation.data) * torch.sin(torch.pow(2, -x_validation.data))
    x_validation.unsqueeze_(1)
    y_validation.unsqueeze_(1)
    net = RegressionNet(100)
    net.optimize_mae(x_validation, y_validation)
    predict(net, x_validation, y_validation)
    plt.title(f"metric{net.loss_mae(y_validation, net.forward(x_validation))}")
    plt.show()


if __name__ == "__main__":
    run_test()
