import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ProsodicSegmentationHeadText(nn.Module):

    def __init__(self, input_dim, conv_dim, dropout, lstm_hidden, nbr_lstm,
                 num_attention_heads=4):
        super().__init__()

        # Normalization layer
        self.norm = nn.LayerNorm(input_dim)

        # --- Text-guided cross-attention (optional) ---
        # Query = audio frames, Key/Value = text frames (already stretched
        # onto the same frame grid by align_text_to_frames, so both have
        # shape (B, T, input_dim) -- same dimension, no extra projection
        # needed since both come from the same Pantagruel Speech-Text
        # encoder space).
        #
        # This module is only exercised when `text` is passed to forward();
        # if it is None (e.g. LeBenchmark embeddings, or older caches
        # without text_features), the model behaves exactly as before.
        self.text_norm = nn.LayerNorm(input_dim)
        self.cross_attention = nn.MultiheadAttention(
            embed_dim=input_dim,
            num_heads=num_attention_heads,
            dropout=dropout,
            batch_first=True,
        )
        # Zero-init the attention's output projection: attn_out is exactly
        # 0 at initialization, regardless of the seed, so training starts
        # out identical to the text-free baseline.
        #
        # IMPORTANT: do NOT also multiply by a separate learned gate
        # initialized at 0 on top of this. Combining the two creates a
        # mutual dead end: with attn_out == 0 (because out_proj == 0), the
        # gradient w.r.t. the gate is loss_grad * attn_out == 0, and with
        # gate == 0, the gradient w.r.t. out_proj is loss_grad * gate * ... == 0.
        # Neither can ever move first, so the branch stays permanently
        # inactive -- this was confirmed in practice: attn_gate remained
        # exactly 0.0 after a full training run, making the "with text"
        # model mathematically identical to the text-free one throughout
        # training. Zero-initializing out_proj alone is sufficient to get
        # the same safe start, without this deadlock.
        nn.init.zeros_(self.cross_attention.out_proj.weight)
        nn.init.zeros_(self.cross_attention.out_proj.bias)

        # Convolution layer
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=5, padding=2),
            nn.ReLU(), nn.Dropout(dropout),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1),
            nn.ReLU(),
        )

        # LSTM layer
        self.lstm = nn.LSTM(input_size=conv_dim, hidden_size=lstm_hidden, num_layers=nbr_lstm,
                             batch_first=True, dropout=dropout if nbr_lstm > 1 else 0,
                             bidirectional=True)

        # Classifier

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
            # Residual connection: attn_out is exactly 0 at the very start
            # of training (out_proj zero-init above), so this starts out
            # identical to the text-free baseline. As out_proj moves away
            # from 0 during training, this branch's contribution grows
            # smoothly on its own -- no separate gate needed.
            x = x + attn_out

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
