"""
A file containing configuration parameters.

This script contains parameter values for the Performance RNN model,
training, and quantization statistics collection.

Author: Yuankui Lee, Alexander Song
"""

import torch
from sequence import EventSeq, ControlSeq

#pylint: disable=E1101

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

model = {
    'init_dim': 32,
    'event_dim': EventSeq.dim(),
    'control_dim': ControlSeq.dim(),
    'hidden_dim': 512,
    'gru_layers': 3,
    'gru_dropout': 0.3,
}

train = {
    'learning_rate': 0.001,
    'batch_size': 64,
    'window_size': 200,
    'stride_size': 10,
    'use_transposition': False,
    'control_ratio': 1.0,
    'teacher_forcing_ratio': 1.0
}

collect_quant_stats = {
    'batch_size': 2,
    'num_batches': 1000,
    'window_size': 500,
    'stride_size': 10,
    'use_transposition': False,
    'control_ratio': 1.0,
    'teacher_forcing_ratio': 1.0
}

