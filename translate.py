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

from transformer import *

examples, metadata = tfds.load('ted_hrlr_translate/pt_to_en', with_info=True,
                               as_supervised=True)
train_examples, val_examples = examples['train'], examples['validation']

tokenizer = None
if path.exists('tokenizer.subwords'):
  print ('loading tokenizer....')
  tokenizer = tfds.features.text.SubwordTextEncoder.load_from_file('tokenizer')
else:
  tokenizer = tfds.features.text.SubwordTextEncoder.build_from_corpus(
    itertools.chain((en.numpy() for pt, en in train_examples), (pt.numpy() for pt, en in train_examples)), target_vocab_size=4**13)
  tokenizer.save_to_file('tokenizer')

num_layers = 4
d_model = 128
dff = 512
num_heads = 8

input_vocab_size = tokenizer.vocab_size + 4
target_vocab_size = tokenizer.vocab_size + 4
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

translate(tokenizer.vocab_size+2, "este é um problema que temos que resolver.", transformer, tokenizer, tokenizer)
translate(tokenizer.vocab_size+3, "so this is a problem that we have to solve ourselves.", transformer, tokenizer, tokenizer)
