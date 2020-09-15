#@title Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
# https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import tensorflow_datasets as tfds
import tensorflow as tf

import time
import numpy as np
import matplotlib.pyplot as plt
import itertools
import os.path as path
import sys

from transformer import *

tokenizer = getTokenizer(None)
langs = languages(tokenizer.vocab_size)

num_layers = 8
d_model = 256
dff = 1024
num_heads = 16

input_vocab_size = tokenizer.vocab_size + 1 + len(langs)
target_vocab_size = tokenizer.vocab_size + 1 + len(langs)
dropout_rate = 0.1

transformer = Transformer(num_layers, d_model, num_heads, dff,
                          input_vocab_size, target_vocab_size,
                          pe_input=input_vocab_size,
                          pe_target=target_vocab_size,
                          rate=dropout_rate)

checkpoint_path = "./checkpoints/train"

ckpt = tf.train.Checkpoint(transformer=transformer)

ckpt_manager = tf.train.CheckpointManager(ckpt, checkpoint_path, max_to_keep=5)

# if a checkpoint exists, restore the latest checkpoint.
if ckpt_manager.latest_checkpoint:
  ckpt.restore(ckpt_manager.latest_checkpoint).expect_partial()
  print ('Latest checkpoint restored!!')

if sys.argv[1] == 'help':
  print(langs)
elif sys.argv[1] == 'demo':
  translate(langs['pt'], langs['en'], "este é um problema que temos que resolver.", transformer, tokenizer)
  translate(langs['en'], langs['pt'], "we need to solve the problem.", transformer, tokenizer)
else:
  translate(langs[sys.argv[1]], langs[sys.argv[2]], sys.argv[3], transformer, tokenizer)
