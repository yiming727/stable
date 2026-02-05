import torch
import torch.nn as nn


# --- DNN Model ---
class DNN(nn.Module):
    def __init__(self, input_dim=3):
        super(DNN, self).__init__()
        self.layer1 = nn.Linear(input_dim * 125, 512)
        self.layer2 = nn.Linear(512, 30)

    def forward(self, x):
        x = x.view(x.size(0), -1)  # 展平
        x = torch.relu(self.layer1(x))
        x = torch.relu(self.layer2(x))
        return x

# --- CNN Model ---
class CNN(nn.Module):
    def __init__(self, input_dim=3):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv1d(input_dim, 256, kernel_size=3, stride=1, padding=1)
        self.conv2 = nn.Conv1d(256, 64, kernel_size=3, stride=1, padding=1)
        self.conv3 = nn.Conv1d(64, 32, kernel_size=3, stride=1, padding=1)
        self.fc = nn.Linear(32 * 15, 30)  # 125 -> 62 -> 31 -> 15
        self.pool = nn.MaxPool1d(2)
        self.bn = nn.BatchNorm1d(32)

    def forward(self, x):
        x = x.permute(0, 2, 1)  # [B, 125, 3] -> [B, 3, 125]
        x = self.pool(torch.relu(self.conv1(x)))  # [B, 256, 62]
        x = self.pool(torch.relu(self.conv2(x)))  # [B, 64, 31]
        x = self.pool(torch.relu(self.conv3(x)))  # [B, 32, 15]
        x = torch.relu(self.bn(x))
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x

# --- LSTM Model ---
class LSTM(nn.Module):
    def __init__(self, input_dim=3, hidden_dim=16, output_dim=30):
        super(LSTM, self).__init__()
        self.lstm1 = nn.LSTM(input_dim, hidden_dim, batch_first=True)
        self.lstm2 = nn.LSTM(hidden_dim, output_dim, batch_first=True)

    def forward(self, x):
        x, _ = self.lstm1(x)
        x, _ = self.lstm2(x)
        return x[:, -1, :]

# --- Feature Fusion Network ---
class FeatureFusionNetwork(nn.Module):
    def __init__(self, input_dim=30):
        super(FeatureFusionNetwork, self).__init__()
        self.fc1 = nn.Linear(input_dim * 3, 512)
        self.fc2 = nn.Linear(512, 3)

    def forward(self, dnn_feat, cnn_feat, lstm_feat):
        x = torch.cat((dnn_feat, cnn_feat, lstm_feat), dim=-1)
        x = torch.relu(self.fc1(x))
        x = self.fc2(x)
        return x

# --- Main Fusion Network ---
class MultiModelEnsembleNet(nn.Module):
    def __init__(self, input_dim=3):
        super(MultiModelEnsembleNet, self).__init__()
        self.dnn = DNN(input_dim)
        self.cnn = CNN(input_dim)
        self.lstm = LSTM(input_dim)
        self.fusion = FeatureFusionNetwork()

    def forward(self, x):
        dnn_feat = self.dnn(x)
        cnn_feat = self.cnn(x)
        lstm_feat = self.lstm(x)
        output = self.fusion(dnn_feat, cnn_feat, lstm_feat)
        return output