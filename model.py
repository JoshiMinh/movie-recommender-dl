import torch
import torch.nn as nn
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence

class SequenceRecommender(nn.Module):
    def __init__(self, num_users, num_movies, user_emb_dim=32, movie_emb_dim=64, 
                 hidden_dim=128, rnn_type='gru', num_layers=1, dropout=0.3):
        super(SequenceRecommender, self).__init__()
        
        # Embeddings
        self.user_embedding = nn.Embedding(num_users, user_emb_dim)
        self.movie_embedding = nn.Embedding(num_movies, movie_emb_dim, padding_idx=0)
        
        self.rnn_type = rnn_type.lower()
        input_dim = user_emb_dim + movie_emb_dim
        
        rnn_dropout = dropout if num_layers > 1 else 0.0
        if self.rnn_type == 'rnn':
            self.rnn = nn.RNN(input_size=input_dim, hidden_size=hidden_dim, 
                              num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
        elif self.rnn_type == 'lstm':
            self.rnn = nn.LSTM(input_size=input_dim, hidden_size=hidden_dim, 
                               num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
        elif self.rnn_type == 'gru':
            self.rnn = nn.GRU(input_size=input_dim, hidden_size=hidden_dim, 
                              num_layers=num_layers, batch_first=True, dropout=rnn_dropout)
        else:
            raise ValueError("rnn_type must be one of 'rnn', 'lstm', or 'gru'")
            
        self.dropout_layer = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim, num_movies)
        
    def forward(self, user_ids, padded_seqs, seq_lengths):
        batch_size, seq_len = padded_seqs.size()
        
        # User embedding
        user_emb = self.user_embedding(user_ids) # (batch_size, user_emb_dim)
        user_emb_seq = user_emb.unsqueeze(1).repeat(1, seq_len, 1) # (batch_size, seq_len, user_emb_dim)
        
        # Movie embedding
        movie_emb_seq = self.movie_embedding(padded_seqs) # (batch_size, seq_len, movie_emb_dim)
        
        # Concatenate movie and user embeddings per timestep
        concat_emb = torch.cat([movie_emb_seq, user_emb_seq], dim=-1)
        concat_emb = self.dropout_layer(concat_emb)
        
        # Packed sequences
        packed_input = pack_padded_sequence(concat_emb, seq_lengths.cpu(), batch_first=True, enforce_sorted=False)
        
        if self.rnn_type == 'lstm':
            packed_output, _ = self.rnn(packed_input)
        else:
            packed_output, _ = self.rnn(packed_input)
            
        # Unpack padded sequence
        output_padded, _ = pad_packed_sequence(packed_output, batch_first=True) # (batch_size, seq_len, hidden_dim)
        
        # Extract last valid hidden state ignoring padding based on seq_lengths
        # indices shape required for gather: (batch_size, 1, hidden_dim)
        idx = (seq_lengths - 1).view(-1, 1).expand(batch_size, output_padded.size(-1)).unsqueeze(1)
        idx = idx.to(output_padded.device)
        
        last_hidden_state = output_padded.gather(1, idx).squeeze(1) # (batch_size, hidden_dim)
        last_hidden_state = self.dropout_layer(last_hidden_state)
        
        # Raw logits output
        logits = self.fc(last_hidden_state)
        return logits
        
    def predict(self, user_ids, padded_seqs, seq_lengths):
        logits = self.forward(user_ids, padded_seqs, seq_lengths)
        probs = torch.nn.functional.softmax(logits, dim=-1)
        return probs
