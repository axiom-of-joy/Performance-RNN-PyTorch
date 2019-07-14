# Amadeus

[![License: MIT](https://img.shields.io/badge/License-MIT-yellow.svg)](https://opensource.org/licenses/MIT)

A music generation and model compression project using PyTorch and [Distiller](https://nervanasystems.github.io/distiller/).

## Repo Structure

```
.
├── dataset/
│   ├── midi/
│   │   ├── dataset1/
│   │   │   └── *.mid
│   │   └── dataset2/
│   │       └── *.mid
│   ├── processed/
│   │   └── dataset1/
│   │       └── *.data (preprocess.py)
│   └── scripts/
│       └── *.sh (dataset download scripts)
├── input/
│   └── *.mid (generate.py)
├── output/
│   └── *.mid (generate.py)
├── save/
│   └── *.sess (train.py)
├── stats/
│   └── *.yaml (quantize.py)
├── tests/
│   └── *.py
├── runs/ (tensorboard logdir)

```


## Amadeus

Amadeus is a tool for musicians to break through creative blocks. Suppose you have a musical idea in mind, but you don't how to complete it. With Amadeus, you can connect a MIDI device (e.g., an electric keyboard) to your laptop or mobile device, play an input phrase of music, and hear the musical response of an AI trained on classical repertoire.

Amadeus creates music with Performance RNN, a model for real-time music generation. I use a model compression technique known as [post-training weight quantization](https://nervanasystems.github.io/distiller/quantization.html) to shrink the size of the original model by a factor of four with only a marginal loss in output quality.


## Performance RNN

Performance RNN is designed to capture the nuances of human musical performance and to generate musical output in real time. It uses a recurrent neural network (RNN) architecture known as gated recurrent units (GRUs) to model sequences of so-called "note events", of which there are four kinds:

- Note-on events, which represent the start of a pitch,
- Note-off events, which represent the end of a pitch,
- Velocity events, which control the volume at which a pitch is played,
- Time events, which move forward in time to the next note event.




This representation of music has several advantages. First, it allows Performance RNN to model "polyphonic" music -- music in which more than one pitch is played at a single time. Second, it captures the nuances of musical performances, e.g., subtle changes in dynamics or rhythm.

Performance RNN is trained on musical data in MIDI format. MIDI, which stands for "musical instrument digital interface", provides a lightweight representation of music and easily interfaces with digital instruments such as electric keyboards and electric guitars. Notable MIDI datasets for classical piano music include the [e-Piano Competition Dataset](http://www.piano-e-competition.com/) and the [MAESTRO Dataset](https://magenta.tensorflow.org/datasets/maestro) (use the scripts in `dataset/scripts/` for downloading these and other MIDI datasets).

 Performance RNN was originally implemented in TensorFlow as a [Google Magenta project](https://magenta.tensorflow.org/performance-rnn), but Amadeus is built upon a [PyTorch re-implementation](https://github.com/djosix/Performance-RNN-PyTorch).

## Post-Training Quantization



## Installation

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
