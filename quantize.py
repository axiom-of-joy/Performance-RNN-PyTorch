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
from distiller.data_loggers import QuantCalibrationStatsCollector
from distiller.data_loggers import collector_context

#-----------------------------------------------------------------------
# Quantizer class.
#-----------------------------------------------------------------------

class Quantizer:
    def __init__(self, model: 'PerformanceRNN'):
        self.model = deepcopy(model)
        convert_model_to_distiller_gru(self.model)

        self.num_batches = config.collect_quant_stats['num_batches']
        self.use_transposition = (
            config.collect_quant_stats['use_transposition'])
        self.window_size = config.collect_quant_stats['window_size']
        self.teacher_forcing_ratio = (
            config.collect_quant_stats['teacher_forcing_ratio'])
        self.control_ratio = config.collect_quant_stats['control_ratio']
        self.batch_size = config.collect_quant_stats['batch_size']

    def collect_stats(self, stats_file, batch_gen):
        """
        Collects pre-quantization calibration statistics for the
        self.model.

        Loads num_batches batches of data from batch_gen and collects
        pre-quantization statistics for self.model. Writes the results
        to stats_file.

        Args:
            stats_file (str): Path to YAML file where statistics will be
                written.

            batch_gen (dataset.batches): Batch generator from data file.

        Returns:
            None
        """
        distiller.utils.assign_layer_fq_names(self.model)
        collector = QuantCalibrationStatsCollector(self.model)
        device = 'cuda:0' if torch.cuda.is_available() else 'cpu'

        with collector_context(collector) as collector:
            for iteration, (events, controls) in enumerate(batch_gen):
                print(iteration)

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
            quantizer(PostTrainLinearQuantizer): Quantizer object with
                quantized model stored at quantizer.model.
        """

        # The following overrides the default quantization routine.
        overrides_yaml = """
        .*eltwise.*:
            fp16: true
        output_fc_activation:
            fp16: true
        """
        overrides = distiller.utils.yaml_ordered_load(overrides_yaml)
        quantizer = PostTrainLinearQuantizer(
            self.model,
            model_activation_stats=stats_file,
            overrides=overrides,
            mode=LinearQuantMode.ASYMMETRIC_SIGNED,
            per_channel_wts=True
        )

        quantizer.prepare_model()
        quantizer.model.eval()  # Disables dropout.
        return quantizer

