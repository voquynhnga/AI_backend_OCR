import torch.nn as nn
import torch.nn.functional as F


class CNNBiLSTMCTC(nn.Module):
    def __init__(self, num_classes, input_height=100):
        super(CNNBiLSTMCTC, self).__init__()

        self.num_classes = num_classes

        # Block 1
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.bn1   = nn.BatchNorm2d(32)

        self.conv2 = nn.Conv2d(32, 32, kernel_size=3, stride=1, padding=1)
        self.bn2   = nn.BatchNorm2d(32)

        self.pool1     = nn.MaxPool2d(kernel_size=2, stride=2)  # H/2, W/2
        self.dropout1  = nn.Dropout(0.2)

        # Block 2
        self.conv3 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.bn3   = nn.BatchNorm2d(64)

        self.conv4 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn4   = nn.BatchNorm2d(64)

        self.conv5 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn5   = nn.BatchNorm2d(64)

        self.conv6 = nn.Conv2d(64, 64, kernel_size=3, stride=1, padding=1)
        self.bn6   = nn.BatchNorm2d(64)

        self.pool2    = nn.MaxPool2d(kernel_size=2, stride=2)   # H/4, W/4
        self.dropout2 = nn.Dropout(0.3)

        # Block 3 (residual)
        self.conv7  = nn.Conv2d(64,  128, kernel_size=3, stride=1, padding=1)
        self.bn7    = nn.BatchNorm2d(128)
        self.conv8  = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn8    = nn.BatchNorm2d(128)

        self.conv9  = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn9    = nn.BatchNorm2d(128)
        self.conv10 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn10   = nn.BatchNorm2d(128)

        self.conv11 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn11   = nn.BatchNorm2d(128)
        self.conv12 = nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1)
        self.bn12   = nn.BatchNorm2d(128)

        self.res_proj = nn.Conv2d(64, 128, kernel_size=1, stride=1, padding=0)
        self.dropout3 = nn.Dropout(0.3)
        self.rnn_input_size = 128

        # BiLSTM
        self.lstm1 = nn.LSTM(
            input_size=self.rnn_input_size,
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=False
        )

        self.dropout_lstm = nn.Dropout(0.3)

        self.lstm2 = nn.LSTM(
            input_size=512,   # 256 * 2 (bidirectional)
            hidden_size=256,
            num_layers=1,
            bidirectional=True,
            batch_first=False
        )

        # Output
        self.fc1         = nn.Linear(512, 512)
        self.dropout_fc  = nn.Dropout(0.5)
        self.fc2         = nn.Linear(512, num_classes)

    def forward(self, x):
        # ── Block 1 ──────────────────────────────────────────
        x = F.relu(self.bn1(self.conv1(x)))
        x = F.relu(self.bn2(self.conv2(x)))
        x = self.pool1(x)
        x = self.dropout1(x)
        # shape: (B, 32, H/2, W/2)

        # ── Block 2 ──────────────────────────────────────────
        x = F.relu(self.bn3(self.conv3(x)))
        x = F.relu(self.bn4(self.conv4(x)))
        x = F.relu(self.bn5(self.conv5(x)))
        x = F.relu(self.bn6(self.conv6(x)))
        x = self.pool2(x)
        x = self.dropout2(x)
        # shape: (B, 64, H/4, W/4)

        # ── Block 3  ─────────────────
        residual = self.res_proj(x)            # (B, 128, H/4, W/4)
        x = F.relu(self.bn7(self.conv7(x)))
        x = F.relu(self.bn8(self.conv8(x)))
        x = x + residual                       # skip connection #1

        residual = x
        x = F.relu(self.bn9(self.conv9(x)))
        x = F.relu(self.bn10(self.conv10(x)))
        x = x + residual                       # skip connection #2

        residual = x
        x = F.relu(self.bn11(self.conv11(x)))
        x = F.relu(self.bn12(self.conv12(x)))
        x = x + residual                       # skip connection #3
        x = self.dropout3(x)
        # shape: (B, 128, H/4, W/4)

        # Column-wise pooling (max + avg)
        x_max = F.max_pool2d(x, kernel_size=(x.size(2), 1))  # (B, 128, 1, W/4)
        x_avg = F.avg_pool2d(x, kernel_size=(x.size(2), 1))  # (B, 128, 1, W/4)
        x = x_max + x_avg                                     # (B, 128, 1, W/4)

        # ── Reshape cho RNN ───────────────────────────────────
        x = x.squeeze(2)        # (B, 128, W/4)
        x = x.permute(2, 0, 1)  # (W/4, B, 128) — time-first cho LSTM

        # ── BiLSTM ────────────────────────────────────────────
        x, _ = self.lstm1(x)    # (T, B, 512)
        x = self.dropout_lstm(x)

        x, _ = self.lstm2(x)    # (T, B, 512)

        # ── Output ────────────────────────────────────────────
        x = self.fc1(x)         # (T, B, 512)
        x = self.dropout_fc(x)

        x = self.fc2(x)         # (T, B, num_classes)

        return F.log_softmax(x, dim=2)