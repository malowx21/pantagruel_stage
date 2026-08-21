import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ProsodicSegmentationHeadText(nn.Module):

    def __init__(self, input_dim, conv_dim, dropout, lstm_hidden, nbr_lstm):
        super().__init__()

        self.norm = nn.LayerNorm(input_dim)
        self.text_norm = nn.LayerNorm(input_dim)
        fused_dim = input_dim * 2

        self.conv = nn.Sequential(
            nn.Conv1d(fused_dim, conv_dim, kernel_size=5, padding=2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        self.lstm = nn.LSTM(input_size=conv_dim, hidden_size=lstm_hidden, num_layers=nbr_lstm,
                             batch_first=True, dropout=dropout if nbr_lstm > 1 else 0,
                             bidirectional=True)

        lstm_out = lstm_hidden * 2
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, lstm_hidden),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden, 1),
        )

    def forward(self, x, lengths, text=None):
        x = self.norm(x)

        if text is not None:
            text = self.text_norm(text)
            x_fused = torch.cat([x, text], dim=-1)
        else:
            zeros = torch.zeros_like(x)
            x_fused = torch.cat([x, zeros], dim=-1)

        x_fused = x_fused.transpose(1, 2)  # (B, 2*D, T)
        x_fused = self.conv(x_fused)
        x_fused = x_fused.transpose(1, 2)  # (B, T, Conv_D)

        if lengths is not None:
            lengths_cpu = lengths.detach().cpu()
            packed = pack_padded_sequence(
                x_fused, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            x_lstm, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=x_fused.shape[1]
            )
        else:
            x_lstm, _ = self.lstm(x_fused)

        logits = self.classifier(x_lstm).squeeze(-1)
        return logits
