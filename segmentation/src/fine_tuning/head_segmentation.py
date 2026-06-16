import torch
import torch.nn as nn


class ProsodicSegmentationHead(nn.Module):

    def __init__(
        self,
        input_dim: int,
        conv_channels: int = 256,
        lstm_hidden: int = 256,
        lstm_layers: int = 2,
        dropout: float = 0.2
    ):
        super().__init__()

        # normalization
        self.norm = nn.LayerNorm(input_dim)

        # local temporal encoder
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_channels, kernel_size=5, padding=2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Conv1d(conv_channels, conv_channels, kernel_size=3, padding=1),
            nn.ReLU()
        )

        # sequence model
        self.lstm = nn.LSTM(
            input_size=conv_channels,
            hidden_size=lstm_hidden,
            num_layers=lstm_layers,
            batch_first=True,
            bidirectional=True,
            dropout=dropout if lstm_layers > 1 else 0.0
        )

        lstm_out = lstm_hidden * 2

        # classifier
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, lstm_out // 2),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_out // 2, 1)
        )

    def forward(self, x):
        """
        x: (B, T, D)
        returns: (B, T)
        """

        # LayerNorm
        x = self.norm(x)

        # Conv1D expects (B, D, T)
        x = x.transpose(1, 2)
        x = self.conv(x)

        # back to (B, T, D)
        x = x.transpose(1, 2)

        # BiLSTM temporal modeling
        x, _ = self.lstm(x)

        # classification
        logits = self.classifier(x).squeeze(-1)

        return logits
