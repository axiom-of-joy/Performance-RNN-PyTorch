# Amadeus

A music generation and model compression project using PyTorch and [Distiller](https://nervanasystems.github.io/distiller/).

## Musical Autocomplete

Amadeus is a tool for musicians to break through creative blocks. Suppose you have a musical idea in mind, but you don't how to complete it. With Amadeus, you can connect a MIDI device (e.g., an electric keyboard) to your laptop or mobile device, play an input phrase of music, and hear the musical response of an AI trained on classical repertoire.

Amadeus uses Performance RNN, a model for real-time music generation described in [this blog post](https://magenta.tensorflow.org/performance-rnn). Performance RNN was originally implemented in TensorFlow, but Amadeus is built upon a [PyTorch re-implementation](https://github.com/djosix/Performance-RNN-PyTorch). Amadeus uses a model compression technique known as [post-training weight quantization](https://nervanasystems.github.io/distiller/quantization.html) to shrink the size of the neural network with only a marginal loss in output quality.


## Installation and Setup

The code in this repository currently requires a CUDA device to run. Clone this repository with

```
git clone https://github.com/axiom-of-joy/amadeus.git
```

Navigate into the Amadeus project folder with
```
cd amadeus
```
I recommend creating and activating a virtual environment `env`.

```
virtualenv env
source env/bin/activate
```

With the virtual environment activated, install the requisite Python packages with
```
pip install -r requirements.txt
```
Next, clone my [forked Distiller repository]() _inside_ of the `amadeus` folder with
```
git clone https://github.com/axiom-of-joy/distiller.git
```
Distiller is an "open-source Python package for neural network compression research" used here for post-training weight quantization. My forked Distiller repository contains a custom sub-module for quantizing the gated recurrent units (GRUs) used in Performance RNN. To gain access to this sub-module, install the forked Distiller repo in development with the commands:

```
cd distiller
pip install -e .
```

## Usage

This usage documentation is copied with modifications from the original repo.

#### Download datasets

```shell
cd dataset/
bash scripts/NAME_scraper.sh midi/NAME
```

#### Preprocessing

```shell
# Preprocess all MIDI files under dataset/midi/NAME
python3 preprocess.py dataset/midi/NAME dataset/processed/NAME
```

#### Training

```shell
# Train on .data files in dataset/processed/MYDATA, and save to save/myModel.sess every 10s
python3 train.py -s save/myModel.sess -d dataset/processed/MYDATA -i 10

# Or...
python3 train.py -s save/myModel.sess -d dataset/processed/MYDATA -p hidden_dim=1024
python3 train.py -s save/myModel.sess -d dataset/processed/MYDATA -b 128 -c 0.3
python3 train.py -s save/myModel.sess -d dataset/processed/MYDATA -w 100 -S 10
```

#### Quantizing

The code for collecting pre-quantization calibration statistics is currently contained in a Jupyter notebook, but will soon be incorporated into `quantize.py`. For now, if you wish to generate output using the quantized model, use my previously computed statistics at `stats/performance_rnn_pretrained_stats.yaml`.

#### Generating

```shell
# Generate with pitch histogram and note density for unquantized model.
python3 generate.py -s save/test.sess -l 1000 -c '1,0,1,0,1,1,0,1,0,1,0,1;3'

# Generate with pitch histogram and note density for quantized model.
python3 generate.py -s save/test.sess \
  -l 1000 -c '1,0,1,0,1,1,0,1,0,1,0,1;3' \
  -q 'stats/pre_quant.yaml'

# Generation conditioned on an input midi file.
# (currently supports a batch size of 1)
python3 generate.py -s save/test.sess -l 1000 -b 1 -i "input/input.midi"
```

Pre-trained weights for this model are stored in `.sess` files and can be found in the [original repository](https://github.com/djosix/Performance-RNN-PyTorch#pretrained-model).

Using the quantized model for generation assumes you have computed and stored pre-quantization calibration statistics in `stats/pre_quant.yaml` (the current statistics are stored in `stats/performance_rnn_pretrained_stats.yaml`).


## Contribution

This repository is built upon a [pre-existing implementation of Performance RNN](https://github.com/djosix/Performance-RNN-PyTorch). I modified the original codebase to produce output conditioned on a user's musical input. I also built a `Quantizer` class in `quantize.py` to compress the existing Performance RNN model.

At this time, the model compression framework [Distiller](https://nervanasystems.github.io/distiller/) does not support post-training weight quantization of gated recurrent units (GRUs), a recurrent neural network (RNN) architecture used in Performance RNN. To overcome this challenge, I built a sub-module within my [forked Distiller repository](https://github.com/axiom-of-joy/distiller) for quantizing GRUs (see the sub-module [here](https://github.com/axiom-of-joy/distiller/blob/master/distiller/modules/gru.py) and the tests [here](https://github.com/axiom-of-joy/distiller/blob/master/tests/test_gru.py)).

## Acknowledgment
I completed this project as an Artificial Intelligence Fellow at Insight Data Science. Many individuals contributed to the success of this project, but I owe a special thanks to program directors Amber Roberts and Matt Rubashkin, and especially to Ben Hammel for his generous help and advice.
