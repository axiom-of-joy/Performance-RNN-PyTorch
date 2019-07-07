'''
This script quantizes the PerformanceRNN model defined in model.py. It
requires that the forked Distiller repo at
https://github.com/axiom-of-joy/distiller is cloned inside of the
amadeus project folder and installed in development mode (see README.md
for installation and set-up details).
'''

import os, sys, optparse
import config
from model import PerformanceRNN
import torch
from torch import nn
import distiller
from distiller.modules.gru import convert_model_to_distiller_gru
from tqdm import tqdm
import numpy as np
from sequence import EventSeq, Control, ControlSeq
from data import Dataset
from sequence import EventSeq
import torch.functional as F
from distiller.quantization import PostTrainLinearQuantizer, LinearQuantMode
from copy import deepcopy


##-----------------------------------------------------------------------
## Settings.
##-----------------------------------------------------------------------
#
#def getopt():
#    parser = optparse.OptionParser()
#
#    parser.add_option('-b', '--batch-size',
#                      dest='batch_size',
#                      type='int',
#                      default=8)
#
#    parser.add_option('-s', '--session',
#                      dest='sess_path',
#                      type='string',
#                      default='save/train.sess',
#                      help='session file containing the trained model')
#
#    parser.add_option('-z', '--init-zero',
#                      dest='init_zero',
#                      action='store_true',
#                      default=False)
#    
#    parser.add_option('-n', '--num-batches',
#                      dest='num_batches',
#                      type='int',
#                      default=10,
#                      help='number of batches for pre-quantization statistics')
#
#    parser.add_option('-q', '--stats-file',
#                      dest='stats_file',
#                      type='str',
#                      default=None,
#                      help='path to prequantization statistics file')
#
#    return parser.parse_args()[0]
#
#
##-----------------------------------------------------------------------
## Parse command line arguments.
##-----------------------------------------------------------------------
#
#opt = getopt()
#stats_file = opt.stats_file
#num_batches = opt.num_batches
#batch_size = opt.batch_size
#sess_path = opt.sess_path
#init_zero = opt.init_zero
#

#-----------------------------------------------------------------------
# Quantizer class.
#-----------------------------------------------------------------------

class Quantizer:
    def __init__(self, model: 'PerformanceRNN'):
        self.model = model
        convert_model_to_distiller_gru(self.model)

    def collect_stats(self, stats_file, batch_gen, num_batches):
        """
        Collects pre-quantization calibration statistics for the
        self.model.

        Loads num_batches batches of data from batch_gen and collects
        pre-quantization statistics for self.model. Writes the results
        to stats_file.

        Args:
            stats_file (str): Path to YAML file where statistics will be
                written.

            batch_gen (dataset.batches): Batch generator from dataset
                file.

            num_batches (int): Number of batches to use while computing
                statistics.

        Returns:
            None
        """

        with collector_context(collector) as collector:
            for iteration, (events, controls) in enumerate(batch_gen):
                print(iteration)

                if iteration == num_batches:
                    break

                if use_transposition:
                    offset = np.random.choice(np.arange(-6, 6))
                    events, controls = utils.transposition(events, controls,
                                                           offset)

                events = torch.LongTensor(events).to(device)
                assert events.shape[0] == window_size

                if np.random.random() < control_ratio:
                    controls = torch.FloatTensor(controls).to(device)
                    assert controls.shape[0] == window_size
                else:
                    controls = None

                init = torch.randn(batch_size, model.init_dim).to(device)
                outputs = model.generate(
                    init,
                    window_size,
                    events=events[:-1],
                    controls=controls,
                    teacher_forcing_ratio=teacher_forcing_ratio,
                    output_type='logit')


                assert outputs.shape[:2] == events.shape[:2]

            # Save stats.
            collector.save(stats_file)

    def quantize(self, stats_file):
        """
        Quantizes self.model using the pre-quantization statistics in
        stats_file.

        Args:
            stats_file (str): Path to YAML file containing
                pre-quantization statistics.

        Returns:
            quantizer: Distiller quantizer object.
        """

        overrides_yaml = """
        .*eltwise.*:
            fp16: true
        output_fc_activation:
            fp16: true
        """
        overrides = distiller.utils.yaml_ordered_load(overrides_yaml)
        quantizer = PostTrainLinearQuantizer(
            deepcopy(self.model),
            model_activation_stats=stats_file,
            overrides=overrides,
            mode=LinearQuantMode.ASYMMETRIC_SIGNED,
            per_channel_wts=True
        )

        quantizer.prepare_model()
        quantizer.model.eval()
        return quantizer

def main():
    print("hello world")

if __name__ == "__main__":
    main()

