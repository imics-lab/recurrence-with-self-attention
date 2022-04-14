# Recurrence with Self Attention versus Time Series Transformer

Code from the paper titled "Recurrence and Self-Attention vs the
Transformer for Time-Series Classification: A Comparative Study,"
published in the 20th International Conference on Artificial Intelligence in
Medicine, AIME 2022, by Springer Nature."

Abstract: Recently the transformer has established itself as the state-
of-the-art in text processing and has demonstrated impressive results
in image processing, leading to the decline in the use of recurrence in
neural network models. As established in the seminal paper, Attention
Is All You Need, recurrence can be removed in favor of a simpler model
using only self-attention. While transformers have shown themselves to
be robust in a variety of text and image processing tasks, these tasks all
have one thing in common; they are inherently non-temporal. Although
transformers are also finding success in modeling time-series data, they
also have their limitations as compared to recurrent models. We explore a
class of problems involving classification and prediction from time-series
data and show that recurrence combined with self-attention can meet or
exceed the transformer architecture performance. This particular class of
problem, temporal classification, and prediction of labels through time
from time-series data is of particular importance to medical data sets
which are often time-series based.

## Requirements

Python 3.7
Tensor Flow 2.4


## Configuration

All configurable parameters are controlled from cfg.py


## Usage

Usage: main.py [1|2] [-vg]
       (assuming python3 in /usr/bin/)

1: LSTM (default)
2: Transformer

v: verbose mode (optional)
g: graphing mode (optional)
