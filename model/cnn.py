import torch


class GlobalAveragePool1d(torch.nn.Module):
    def __init__(self):
        super(GlobalAveragePool1d, self).__init__()

    def forward(self, x):
        return x.mean(-1)


class SirHongsCNN(torch.nn.Module):
    def __init__(self, num_classes):
        super(SirHongsCNN, self).__init__()
        self.conv1 = torch.nn.Conv1d(1, 16, kernel_size=(50,))
        self.bn1 = torch.nn.BatchNorm1d(16)
        self.pool1 = torch.nn.MaxPool1d(kernel_size=3)

        self.conv2 = torch.nn.Conv1d(16, 32, kernel_size=(30,))
        self.bn2 = torch.nn.BatchNorm1d(32)
        self.pool2 = torch.nn.MaxPool1d(kernel_size=3)

        self.conv3 = torch.nn.Conv1d(32, 64, kernel_size=(30,))
        self.bn3 = torch.nn.BatchNorm1d(64)
        self.pool3 = torch.nn.MaxPool1d(kernel_size=3)

        self.conv4 = torch.nn.Conv1d(64, 128, kernel_size=(10,))
        self.bn4 = torch.nn.BatchNorm1d(128)
        self.pool4 = torch.nn.MaxPool1d(kernel_size=3)

        self.avg_pool = GlobalAveragePool1d()  # torch.nn.AvgPool1d(kernel_size=115)

        self.fc = torch.nn.Linear(128, num_classes)

    def forward(self, x):
        # x [batch_size, 1, 10000]
        x = self.conv1(x).relu()
        x = self.bn1(x)
        x = self.pool1(x)

        x = self.conv2(x).relu()
        x = self.bn2(x)
        x = self.pool2(x)

        x = self.conv3(x).relu()
        x = self.bn3(x)
        x = self.pool3(x)

        x = self.conv4(x).relu()
        x = self.bn4(x)
        x = self.pool4(x)

        x = self.avg_pool(x).flatten(1)
        x = self.fc(x)
        return x.softmax(-1)
