'''
This scripts tests the Quantizer class in quantize.py as well as the
changes made in generate.py. To run all tests, use pytest with the verbose
flag.
'''

# Add source folder to path.
import sys
sys.path.insert(0, '../')
import quantize
from quantize import Quantizer
from model import PerformanceRNN
import pudb
import pytest
import torch
from distiller.quantization.range_linear import PostTrainLinearQuantizer



def test_quantize():
    '''
    Tests Quantizer.quantize_method. This test uses specific 
    '''

    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sess_path = "../save/ecomp_w500.sess"
    stats_file = "../performance_rnn_pretrained_stats.yaml"

    # Load from device.
    state = torch.load(sess_path)
    rnn_model = PerformanceRNN(**state['model_config']).to(device)
    rnn_model.load_state_dict(state['model_state'])
    
    # Quantizer.model.
    Q = Quantizer(rnn_model)
    quantizer = Q.quantize(stats_file)
    
    assert isinstance(quantizer, PostTrainLinearQuantizer)


def test_quantizer_quant_stats_collect():
    '''
    Tests Quantizer.quantize_method. This test uses specific 
    '''
    
    device = 'cuda:0' if torch.cuda.is_available() else 'cpu'
    sess_path = "../save/ecomp_w500.sess"
    stats_file = "stats.yaml"

    # Load from device.
    state = torch.load(sess_path)
    rnn_model = PerformanceRNN(**state['model_config']).to(device)
    rnn_model.load_state_dict(state['model_state'])

    Q = Quantizer(rnn_model)
    num_batches = 10
    rnn_model.collect_stats(stats_file, batch_gen, num_batches):
    



