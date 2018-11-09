# Copyright 2018 Google LLC
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================

"""Training the network."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import datetime
import functools
import glob
import itertools
import json
import os
import random
import sys
import time

import numpy as np
import tensorflow as tf
from tensorflow.core.framework import summary_pb2

import model.network as network
import cognitive.task_bank as task_bank
from cognitive import constants
from cognitive.constants import config
import cognitive.train_utils as tu

tf.app.flags.DEFINE_string('hparams', '',
                           'Comma separated list of name=value hyperparameter '
                           'pairs. These values will override the defaults')

# task_family flag inherited from task_bank.py
tf.app.flags.DEFINE_integer('batch_size', 128, 'batch size')

# Logistics
tf.app.flags.DEFINE_string('data_dir', None,
                           'Directory with training and validation data. '
                           'The directory should contain subdirectories '
                           'starting with train_, val_, and test_ . If no directory '
                           'is given, data will be generated on the fly.')
tf.app.flags.DEFINE_string('model_dir', '/tmp/cog',
                           'Directory containing saved model files.')

FLAGS = tf.app.flags.FLAGS


def get_default_hparams_dict():
  return dict(
      # exclude one task during training
      exclude_task_train='none',
      # learning rate decay: lr multiplier per 1M examples
      # value of 0.85 tranlates to 0.1 per 14M examples
      # value of 0.90 tranlates to 0.23 per 14M examples
      # value of 0.95 tranlates to 0.5 per 14M examples
      lr_decay=0.95,
      # learning rate
      learning_rate=0.0005,

      # gradient clipping
      grad_clip=10.,
      # clipping value for rnn state norm
      rnn_state_norm_clip=10000.,
      # number of core recurrent units
      n_rnn=768,
      # type of core rnn
      rnn_type='gru',
      # whether to use 128 as input image size
      use_img_size_128=False,
      # number of vision network output
      n_out_vis=128,
      # type of visual network
      use_vgg_pretrain=False,
      # type of visual network
      vis_type='myconv',
      # number of units for question network
      n_rnn_rule=128,
      # type of rule rnn
      rnn_rule_type='lstm',
      # embedding size for question words
      embedding_size=64,
      # train initial state or not
      train_init=True,
      # beta1 for AdamOptimizer
      adam_beta1=0.1,
      # beta2 for AdamOptimizer
      adam_beta2=0.0001,
      # epsilon for AdamOptimizer
      adam_epsilon=1e-8,
      # rule network bidirectional or not
      rnn_rule_bidir=True,
      # number of time point to repeat for each epoch
      n_time_repeat=5,
      # build network with visual attention or not
      feature_attention=True,
      # state-dependent attention?
      state_dep_feature_attention=False,
      # whether use a MLP to generation feature attention
      feature_attention_use_mlp=False,
      # whether apply feature attention to the second-to-last conv layer
      feature_attend_to_2conv=False,
      # whether to feed a spatially-summed visual input to core
      feed_space_sum_to_core=True,
      # build network with visual spatial attention or not
      spatial_attention=True,
      # whether spatial attention depends on retrieved memory
      memory_dep_spatial_attention=False,
      # whether spatial attention is fed back to controller
      feed_spatial_attn_back=True,
      # how are rule outputs used as memory
      verbal_attention=True,
      # size of the query for rule memory
      memory_query_size=128,
      # number of maps in visual spatial memory
      vis_memory_maps=4,
      # only use visual memory to point short-cut
      only_vis_to_pnt=True,
      # optimizer to use
      optimizer='adam',
      # momentum value to use with "momentum" optimizer
      momentum=0.9,
      # Whether to use Nesterov Accelerated Gradient with "momentum" optimizer
      nesterov=True,
      # signal new epoch
      signal_new_epoch=False,
      # final readout using a MLP
      final_mlp=False,
      # L2 regularization, consider a value between 1e-4 and 1e-5
      l2_weight=2*1e-5,

      # number of epochs each trial
      n_epoch=4,
      # average number of epochs an object needs to be held in memory
      average_memory_span=2,
      # maximum number of distractors
      max_distractors=1,

      # value 'factor' param to variance_scaling_initializer used as
      # controller GRU kernel initializer
      controller_gru_init_factor=0.3,

      # normalize images mean 0/std 1
      normalize_images=False,
  )


def get_dataparams(hparams):
  # Pick a random number of distractors between 1 and 10
  return dict(
    n_distractor=random.randint(1, hparams.max_distractors),
    average_memory_span=hparams.average_memory_span,
  )


def test_input_generator(task_families, hparams):
  def getter():
    next_family = 0
    n_task_family = len(task_families)
    while True:
      tasks = []
      family = task_families[next_family]
      next_family = (next_family + 1) % n_task_family
      for i in range(FLAGS.batch_size):
        tasks.append(task_bank.random_task(family))
      #print("Yielding batch of " + tasks[0].__class__.__name__)
      feeds = tu.generate_feeds(tasks, hparams, get_dataparams(hparams))
      yield feeds + (np.array([family]), )

  return getter


def evaluate(sess, task_families, model, task_family_tensor):
  print('Evaluating over the test dataset. You can interrupt the script at any '
        'time to see accuracy up to that point.')

  task_family_dict = dict([(task_family, i) for i, task_family in
                           enumerate(task_families)])

  acc_list = [0] * len(task_families)
  loss_list = [0] * len(task_families)
  family_list = [0] * len(task_families)
  start = time.time()
  for i in itertools.count():

    # Log progress
    if i and i % 100 == 0:
      rate = (i * FLAGS.batch_size) / (time.time() - start)
      done_examples = (i * FLAGS.batch_size)
      remaining_secs = int((500000 - done_examples) / rate)
      print("Running batch {}. Rate: {} examples/sec. Reaching 0.5M examples "
            "will take {} more".format(
              i, rate, str(datetime.timedelta(seconds=remaining_secs))))

    try:
      tf_val, acc_tmp, loss_tmp = sess.run([task_family_tensor, model.acc, model.loss])
      assert (tf_val == tf_val[0]).all(), ('Not all task families are the '
          'same in an evaluation batch. Does your batch_size evenly divide '
          'total number of evaluation examples? Family values %s' % tf_val)
      family = tf_val[0].decode('utf-8')
      # TF pads string tensors with zero bytes for some reason. Remove them.
      family = family.strip('\x00')
      family_ind = task_family_dict[family]

      acc_list[family_ind] += acc_tmp
      loss_list[family_ind] += loss_tmp
      family_list[family_ind] += 1
    except (tf.errors.OutOfRangeError, KeyboardInterrupt) as e:
      print("Run the testing over {} examples".format(FLAGS.batch_size * i))
      break

  acc_list = [x/float(count) if count else 0.0 for x, count in zip(acc_list, family_list)]
  loss_list = [x/float(count) if count else 0.0 for x, count in zip(loss_list, family_list)]

  for i, task_family_test in enumerate(task_families):
    print('{:25s}: Acc {:0.3f}  Loss {:0.3f}'.format(
        task_family_test, acc_list[i], loss_list[i]))

  acc_avg = np.mean([x for x in acc_list if x])
  loss_avg = np.mean([x for x in loss_list if x])

  print('Overall accuracy {:0.3f}'.format(acc_avg))
  print('------------------------------------')
  print('Validation took : {:0.2f}s'.format(time.time() - start))
  sys.stdout.flush()

  return acc_list


def _fname_to_ds(input_files, batch_size):
  feed_types = (tf.float32, tf.int64, tf.int64, tf.float32, tf.float32,
                tf.int64, tf.float32, tf.float32, tf.string)

  ds = tf.data.TextLineDataset(filenames=input_files, compression_type='GZIP')
  ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  ds = ds.map(
        lambda examples: tuple(tf.py_func(
          tu.json_to_feeds, [examples], feed_types, stateful=False)))
  return ds


def get_ds_from_files(data_dir, batch_size):
  """Returns inputs tensors for data in data_dir."""
  data_type = 'test'
  input_dir_glob = os.path.join(data_dir, data_type + '_*')
  input_dir = glob.glob(input_dir_glob)
  assert len(input_dir) == 1, ('Expected to find one ' + data_type +
      ' directory in ' + data_dir + ' but got ' + input_dir)
  input_dir = input_dir[0]
  input_files = glob.glob(os.path.join(input_dir, '*'))

  feed_shapes = (tf.TensorShape([None, None, 112, 112, 3]),
                 tf.TensorShape([config['maxseqlength'], None]),
                 tf.TensorShape([None]),
                 tf.TensorShape([None, None]),
                 tf.TensorShape([None, None]),
                 tf.TensorShape([None]),
                 tf.TensorShape([None]),
                 tf.TensorShape([None]),
                 tf.TensorShape([None]))

  input_ds = (tf.data.Dataset
      .from_tensor_slices(input_files)
      .interleave(functools.partial(_fname_to_ds, batch_size=batch_size),
                  cycle_length=len(input_files),
                  block_length=1))


  input_ds = input_ds.prefetch(buffer_size=2)
  input_feeds = input_ds.make_one_shot_iterator().get_next()
  for inp, shape in zip(input_feeds, feed_shapes):
    inp.set_shape(shape)
  input_feeds_dict = {'image': input_feeds[0],
                      'question': input_feeds[1],
                      'seq_len': input_feeds[2],
                      'point': input_feeds[3],
                      'point_xy': input_feeds[4],
                      'answer': input_feeds[5],
                      'mask_point': input_feeds[6],
                      'mask_answer': input_feeds[7],
                      }

  task_family_tensor = input_feeds[8]
  return input_feeds_dict, task_family_tensor


def get_inputs(train_task_families, task_families, hparams):
  if FLAGS.data_dir:
    print("Reading test examples from ", FLAGS.data_dir)
    return get_ds_from_files(FLAGS.data_dir, FLAGS.batch_size)

  print("Generating test examples on the fly")
  # Set up input pipelines
  feed_types = (tf.float32, tf.int64, tf.int64, tf.float32, tf.float32,
                tf.int64, tf.float32, tf.float32)
  feed_shapes = (tf.TensorShape([None, None, 112, 112, 3]),
                 tf.TensorShape([config['maxseqlength'], None]),
                 tf.TensorShape([None]),
                 tf.TensorShape([None, None]),
                 tf.TensorShape([None, None]),
                 tf.TensorShape([None]),
                 tf.TensorShape([None]),
                 tf.TensorShape([None]))


  test_ds = tf.data.Dataset.from_generator(
      test_input_generator(task_families[:], hparams),
      feed_types + (tf.string,),
      feed_shapes + (tf.TensorShape([None]),))
  test_ds = test_ds.prefetch(buffer_size=2)
  test_feeds = test_ds.make_one_shot_iterator().get_next()
  test_feeds_dict = {'image': test_feeds[0],
                    'question': test_feeds[1],
                    'seq_len': test_feeds[2],
                    'point': test_feeds[3],
                    'point_xy': test_feeds[4],
                    'answer': test_feeds[5],
                    'mask_point': test_feeds[6],
                    'mask_answer': test_feeds[7],
                    }
  task_family_tensor = test_feeds[8]

  return test_feeds_dict, task_family_tensor


def run_test(hparams, model_dir):
  task_families = list(task_bank.task_family_dict.keys())
  train_task_families = task_families

  if not tf.gfile.Exists(model_dir):
    print("model directory", model_dir, "does not exist")
    sys.exit(1)

  ######################### Build the model ##################################
  feeds_dict, task_family_tensor = get_inputs(
      train_task_families, task_families, hparams)

  tf.train.get_or_create_global_step()

  model = network.Model(hparams, config)
  model.build(feeds_dict, FLAGS.batch_size, is_training=True)

  saver = tf.train.Saver()

  ########################## Restore and Test ##########################
  checkpoint_path = model_dir
  with tf.Session() as sess:
    cpkt_path = tf.train.latest_checkpoint(checkpoint_path)
    if cpkt_path is not None:
      print("Restoring model from: " + cpkt_path)
      saver.restore(sess, cpkt_path)
    else:
      print("Did not find checkpoint at: " + checkpoint_path)
      sys.exit(1)

    evaluate(sess, task_families, model, task_family_tensor)

def main(_):
  hparams_dict = get_default_hparams_dict()
  hparams = tf.contrib.training.HParams(**hparams_dict)
  hparams = hparams.parse(FLAGS.hparams)  # Overwritten by FLAGS.hparams
  run_test(hparams, FLAGS.model_dir)


if __name__ == '__main__':
  tf.app.run(main)
