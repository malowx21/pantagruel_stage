import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ProsodicSegmentationHeadText(nn.Module):

    def __init__(self, input_dim, conv_dim, dropout, lstm_hidden, nbr_lstm,
                 num_attention_heads=4):
        super().__init__()

        self.norm = nn.LayerNorm(input_dim)
        self.text_norm = nn.LayerNorm(input_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        self.attn_gate = nn.Parameter(torch.zeros(1))
        nn.init.zeros_(self.cross_attention.out_proj.weight)
        nn.init.zeros_(self.cross_attention.out_proj.bias)

        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=5, padding=2),
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

    def _padding_mask(self, lengths, total_length, device):
        """
        Builds a boolean mask of shape (B, T), True at padded positions.
        This is the format expected by nn.MultiheadAttention's
        key_padding_mask: True means "ignore this position".
        """
        arange = torch.arange(total_length, device=device)[None, :]
        valid = arange < lengths.to(device)[:, None]
        return ~valid

    def forward(self, x, lengths, text=None):
        x = self.norm(x)

        if text is not None:
            text = self.text_norm(text)
            key_padding_mask = (
                self._padding_mask(lengths, x.shape[1], x.device)
                if lengths is not None else None
            )
            attn_out, _ = self.cross_attention(
                query=x, key=text, value=text,
                key_padding_mask=key_padding_mask,
            )
            # Residual connection: the audio signal is preserved even
            # where the text stream carries no useful information (e.g.
            # silence, or a frame where alignment failed), the attention
            # output there should stay close to zero and simply add
            # nothing on top of x.
            x = x + self.attn_gate*attn_out

        x = x.transpose(1, 2)  # (B,D,T)
        x = self.conv(x)
        x = x.transpose(1, 2)  # (B,T,D)

        if lengths is not None:
            lengths_cpu = lengths.detach().cpu()

            packed = pack_padded_sequence(
                x, lengths_cpu, batch_first=True, enforce_sorted=False
            )
            packed_out, _ = self.lstm(packed)
            x, _ = pad_packed_sequence(
                packed_out, batch_first=True, total_length=x.shape[1]
            )
        else:
            x, _ = self.lstm(x)

        logits = self.classifier(x).squeeze(-1)
        return logits
