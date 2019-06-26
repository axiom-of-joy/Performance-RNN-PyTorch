import torch
import torch.nn as nn

input_dim = 10
hidden_dim = 12
num_layers = 4
dropout = 0.1

num_directions = 1
lstm = nn.LSTM(input_dim, hidden_dim, num_layers=num_layers, dropout=dropout)
seq_len = 10
batch_size = 60


# Define inputs.

input_ = torch.randn(seq_len, batch_size, input_dim)
hidden = torch.randn(num_layers * num_directions, batch_size, hidden_dim)
cell = torch.randn(num_layers * num_directions, batch_size, hidden_dim)
 

# Forward pass stuff.
output, (h_n, c_n) = lstm(input_, (hidden, cell))
print("Output has shape: ", output.shape)
print("Hidden has shape: ", hidden.shape)
print("Cell has shape: ", cell.shape)

assert False
