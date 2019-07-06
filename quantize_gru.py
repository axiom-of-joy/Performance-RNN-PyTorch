from model import PerformanceRNN
import torch
from torch import nn
import distiller
from distiller.modules.gru import DistillerGRU as GRU
from distiller.modules.gru import convert_model_to_distiller_gru
from tqdm import tqdm
import numpy as np


assert torch.cuda.is_available()
device = 'cuda:0'
sess_path = "save/ecomp_w500.sess"
state = torch.load(sess_path)
rnn_model = PerformanceRNN(**state['model_config']).to(device)
rnn_model.load_state_dict(state['model_state'])

convert_model_to_distiller_gru(rnn_model)


model = rnn_model #quantizer.model.to(device)
model.eval()
batch_size = 1
init = torch.randn(batch_size, model.init_dim).to(device)
max_len = 1000
controls=None
greedy_ratio = 0.7
temperature = 1.0

import pudb

with torch.no_grad():
    #pudb.set_trace()
    outputs = model.generate(init, max_len,
                             controls=controls,
                             greedy=greedy_ratio,
                             temperature=temperature,
                             verbose=True)
                                                                                                                                

outputs = outputs.cpu().numpy().T # [batch, steps]

## Save


import utils
import os

output_dir = "quantized_output/"
os.makedirs(output_dir, exist_ok=True)

for i, output in enumerate(outputs):
    name = f'output-{i:03d}.mid'
    path = os.path.join(output_dir, name)
    n_notes = utils.event_indeces_to_midi_file(output, path)
    print(f'===> {path} ({n_notes} notes)')
