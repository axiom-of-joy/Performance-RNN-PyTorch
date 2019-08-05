'''
This script defines a Quantizer class that may be used to quantize
instances of PerformanceRNN. It requires that the forked Distiller repo
at https://github.com/axiom-of-joy/distiller is cloned inside of the
amadeus project folder and installed in development mode (see README.md
for installation and set-up details).  
This script borrows from the example at
https://github.com/NervanaSystems/distiller/blob/master/examples/word_language_model/quantize_lstm.ipynb.

Author: Alexander Song
'''

import os
import sys
import optparse
from copy import deepcopy
from tqdm import tqdm
import torch
import torch.nn as nn
import torch.functional as F
import numpy as np
import distiller
from distiller.quantization import PostTrainLinearQuantizer, LinearQuantMode
from distiller.modules.gru import convert_model_to_distiller_gru
from distiller.data_loggers import QuantCalibrationStatsCollector
from distiller.data_loggers import collector_context
import config
from model import PerformanceRNN
from sequence import EventSeq, Control, ControlSeq
from sequence import EventSeq
from data import Dataset

#-----------------------------------------------------------------------
# Quantizer class.
#-----------------------------------------------------------------------

class Quantizer:
    """
    A class that quantizes instances of PerformanceRNN.

    Attributes:
        model (nn.Module): An instance of PerformanceRNN.
        batch_size (int): Size of batches used to collect
            pre-quantization statistics.
        num_batches (int): Number of batches used to collect
            pre-quantization statistics.
        use_transposition (bool): True if data should be transposed.
        window_size (int): Length of window size.
        teacher_forcing_ratio (float): The teacher forcing ratio.
        control_ratio (float): The control ratio.
    """

    def __init__(self, model):
        """
        The constructor for the Quantizer class.

        Args:
            model (nn.Module): An instance of PerformanceRNN.

        Returns:
            None
        """

        # Convert the instances of nn.modules.GRU in the input model to
        # instances of DistillerGRU.
        self.model = deepcopy(model)
        convert_model_to_distiller_gru(self.model)
        
        # Load attributes from config.py.
        self.batch_size = config.collect_quant_stats['batch_size']
        self.num_batches = config.collect_quant_stats['num_batches']
        self.use_transposition = (
            config.collect_quant_stats['use_transposition'])
        self.window_size = config.collect_quant_stats['window_size']
        self.teacher_forcing_ratio = (
            config.collect_quant_stats['teacher_forcing_ratio'])
        self.control_ratio = config.collect_quant_stats['control_ratio']

    def collect_stats(self, stats_file, batch_gen):
        """
        Collects pre-quantization calibration statistics for self.model.

        Loads num_batches batches of data from batch_gen and monitors the
        distribution of model weights during the forward pass of the data
        through self.model. The resulting statistics are written to stats_file.

        Args:
            stats_file (str): Path to YAML file where statistics will be
                written.

            batch_gen (dataset.batches): An instance of the batch generator
                defined in data.py.

        Returns:
            None
        """

        # Set up pre-quantization calibration statistics collector.
        distiller.utils.assign_layer_fq_names(self.model)
        collector = QuantCalibrationStatsCollector(self.model)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        print("Collecting pre-quantization calibration statistics.")

        with collector_context(collector) as collector:
            for iteration, (events, controls) in enumerate(batch_gen):
                print("Iteration: {:d} / {:d}".format(iteration,
                                                      self.num_batches))

                # Break when desired number of batches are processed.
                if iteration == self.num_batches:
                    break

                if self.use_transposition:
                    offset = np.random.choice(np.arange(-6, 6))
                    events, controls = utils.transposition(events, controls,
                                                           offset)

                events = torch.LongTensor(events).to(device)
                assert events.shape[0] == self.window_size

                if np.random.random() < self.control_ratio:
                    controls = torch.FloatTensor(controls).to(device)
                    assert controls.shape[0] == self.window_size
                else:
                    controls = None

                init = torch.randn(self.batch_size,
                    self.model.init_dim).to(device)
                outputs = self.model.generate(
                    init,
                    self.window_size,
                    events=events[:-1],
                    controls=controls,
                    teacher_forcing_ratio=self.teacher_forcing_ratio,
                    output_type='softmax')

                assert outputs.shape[:2] == events.shape[:2]

            # Save statistics.
            collector.save(stats_file)


    def quantize(self, stats_file):
        """
        Quantizes self.model using the pre-quantization statistics in
        stats_file.

        Args:
            stats_file (str): Path to YAML file containing
                pre-quantization statistics.

        Returns:
            quantizer (PostTrainLinearQuantizer): Quantizer object with
                quantized model stored at quantizer.model.
        """

        # The following override ensures that element-wise addition and
        # multiplication operations inside of instances of DistillerGRU are not
        # quantized but rather are carried out in half-precision. This has a
        # significant impact on the performance of the quantized model.
#        output_fc.*:
#            fp16: true
        overrides_yaml = """
        inithid.*:
            fp16: true
        event_embedding:
            fp16: true
        concat_input.*:
            fp16: true
        .*eltwise.*:
            fp16: true
        output_fc.*:
            fp16: true
        """
#        overrides_yaml = """
#        .*eltwise.*:
#            fp16: true
#        output_fc_activation:
#            fp16: true
#        """
#        overrides_yaml = """
#        .*eltwise.*:
#            fp16: true
#        """
        overrides = distiller.utils.yaml_ordered_load(overrides_yaml)
        quantizer = PostTrainLinearQuantizer(
            self.model,
            model_activation_stats=stats_file,
            overrides=overrides,
            mode=LinearQuantMode.ASYMMETRIC_SIGNED,
            per_channel_wts=True,
            clip_acts='AVG'
        )
        quantizer.prepare_model()
        quantizer.model.eval()
        return quantizer

#-----------------------------------------------------------------------
# Command line argument parser.
#-----------------------------------------------------------------------

def getopt():
    parser = optparse.OptionParser()

    parser.add_option('-s', '--session',
                      dest='sess_path',
                      type='string',
                      default=None,
                      help='session file containing the trained model')

    parser.add_option('-d', '--dataset',
                      dest='data_path',
                      type='string',
                      default=None)

    parser.add_option('-q', '--stats-file',
                      dest='stats_file',
                      type='string',
                      default=None,
                      help='path to YAML file containing quantization stats')

    return parser.parse_args()[0]

#-----------------------------------------------------------------------
# Main function (for collection of pre-quantization calibration stats)
#-----------------------------------------------------------------------

def main():
    # Parse command line arguments.
    opt = getopt()
    sess_path = opt.sess_path
    stats_file = opt.stats_file
    data_path = opt.data_path
    assert all([sess_path, stats_file, data_path])

    # Load parameters from config.py.
    batch_size = config.collect_quant_stats['batch_size']
    window_size = config.collect_quant_stats['window_size']
    stride_size = config.collect_quant_stats['stride_size']

    # Load pre-trained model.
    assert torch.cuda.is_available()
    device = 'cuda:0'
    state = torch.load(sess_path)
    model = PerformanceRNN(**state['model_config']).to(device)
    model.load_state_dict(state['model_state'])

    # Load dataset.
    dataset = Dataset(data_path, verbose=True)
    dataset_size = len(dataset.samples)
    assert dataset_size > 0
    batch_gen = dataset.batches(batch_size, window_size, stride_size)

    # Create quantizer and collect pre-quantization calibration statistics.
    quantizer = Quantizer(model)
    quantizer.collect_stats(stats_file, batch_gen)


if __name__ == "__main__":
    main()

