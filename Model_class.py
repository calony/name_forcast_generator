import torch
import torch.nn as nn

class Model(nn.Module):
    def __init__(self, num_embeddings=29, embedding_dim=128, hidden_size=128, out_features=29):
        super(Model, self).__init__()
        self.embedding = nn.Embedding(num_embeddings, embedding_dim) 
        self.lstm = nn.LSTM(embedding_dim, hidden_size, batch_first=True) 
        self.out = nn.Linear(hidden_size * 2, out_features)

    def forward(self,x):
        x = self.embedding(x)
        x, _= self.lstm(x) # output x of lstm is a tensor of shape (batch_size, seq_len, hidden_size)
        x = torch.cat((x[:, 0, :], x[:, 1, :]), dim=1)  # Concatenate the hidden states of both time steps, shape (batch_size, hidden_size * 2)
        x = self.out(x)
        return x
