import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence


class ProsodicSegmentationHead(nn.Module):
    
    def __init__(self,input_dim,conv_dim, dropout, lstm_hidden, nbr_lstm): # TODO try to add some config to a config file 
        super().__init__()
        
        # Normalization layer    
        self.norm = nn.LayerNorm(input_dim)
        
        # Convolution layer 
        self.conv = nn.Sequential(
            nn.Conv1d(input_dim, conv_dim, kernel_size=5,padding=2),
            nn.ReLU(),nn.Dropout(dropout),
            nn.Conv1d(conv_dim, conv_dim, kernel_size=3, padding=1 ),
            nn.ReLU(),
        )
        
        # LSTM layer
        self.lstm = nn.LSTM(input_size=conv_dim, hidden_size=lstm_hidden , num_layers=nbr_lstm, batch_first= True, dropout=dropout if  nbr_lstm > 1 else 0, bidirectional=True )   

        # Classifier 
        
        lstm_out = lstm_hidden*2
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out, lstm_hidden ),
            nn.ReLU(),
            nn.Dropout(dropout),
            nn.Linear(lstm_hidden,1),
        )
        
    def forward(self, x, lengths):
        x = self.norm(x) 
        x = x.transpose(1,2) # (B,D,T)
        x = self.conv(x)
        x = x.transpose(1,2) # (B,T,D)
        
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
        
    
        