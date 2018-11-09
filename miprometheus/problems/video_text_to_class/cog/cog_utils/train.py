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
import functools
import glob
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
from cognitive.constants import config
import cognitive.train_utils as tu

tf.app.flags.DEFINE_string('hparams', '',
                           'Comma separated list of name=value hyperparameter '
                           'pairs. These values will override the defaults')

tf.app.flags.DEFINE_boolean('exclude_task', False,
                            'If true, vary the task to exclude')

# Training parameters
# task_family flag inherited from task_bank.py
tf.app.flags.DEFINE_integer('num_steps', 300000, 'number of training steps')
tf.app.flags.DEFINE_integer('display_step', 3000, 'display every # steps')
tf.app.flags.DEFINE_integer('summary_step', 500, 'log summaries every # steps')
tf.app.flags.DEFINE_integer('batch_size', 48, 'batch size for training')
tf.app.flags.DEFINE_integer('show_sample_tasks', 0,
                            'number of sample tasks to show')

# Logistics
tf.app.flags.DEFINE_string('data_dir', None,
                           'Directory with training and validation data. '
                           'The directory should contain subdirectories '
                           'starting with train_ and val_ . If no directory '
                           'is given, data will be generated on the fly.')
tf.app.flags.DEFINE_string('train_dir', '/tmp/cog',
                           'Directory to put the training logs.')
tf.app.flags.DEFINE_boolean('report_param_stat', False,
                            'If true, report parameter statistics')

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
      n_time_repeat=4,
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


def jit_input_generator(train_task_families, hparams):
  def random_getter():
    while True:
      tasks = []
      for i in range(FLAGS.batch_size):
        tasks.append(task_bank.random_task(
            random.choice(train_task_families)))
      #print("Yielding batch: " + str([
      #    t.__class__.__name__ for t in tasks]))
      yield tu.generate_feeds(tasks, hparams,
                              get_dataparams(hparams))

  return random_getter


def val_input_generator(task_families, hparams):
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


def evaluate(trial, sess,
             task_families, hparams, test_writer,
             val_model, task_family_tensor, num_batches):
  print('Starting validation')
  val_start = time.time()

  task_family_dict = dict([(task_family, i) for i, task_family in
                           enumerate(task_families)])

  acc_list = [0] * len(task_families)
  loss_list = [0] * len(task_families)
  for _ in range(num_batches * len(task_families)):
    tf_val, acc_tmp, loss_tmp, = sess.run(
        [task_family_tensor, val_model.acc, val_model.loss])
    assert (tf_val == tf_val[0]).all(), ('Not all task families are the '
        'same in an evaluation batch. Does your batch_size evenly divide '
        'total number of evaluation examples? Family values %s' % tf_val)
    family = tf_val[0].decode('utf-8')
    # TF pads string tensors with zero bytes for some reason. Remove them.
    family = family.strip('\x00')
    family_ind = task_family_dict[family]

    acc_list[family_ind] += acc_tmp
    loss_list[family_ind] += loss_tmp
  acc_list = [x/float(num_batches) for x in acc_list]
  loss_list = [x/float(num_batches) for x in loss_list]

  for i, task_family_test in enumerate(task_families):
    print('{:25s}: Acc {:0.3f}  Loss {:0.3f}'.format(
        task_family_test, acc_list[i], loss_list[i]))

  acc_avg = np.mean(acc_list)
  loss_avg = np.mean(loss_list)

  vals = []
  for task_family, accuracy, loss in zip(task_families + ['avg'],
                                         acc_list + [acc_avg],
                                         loss_list + [loss_avg]):
    vals.append(summary_pb2.Summary.Value(
        tag='summary/test/accuracy_' + task_family,
        simple_value=accuracy))
    vals.append(summary_pb2.Summary.Value(
        tag='summary/test/loss_' + task_family,
        simple_value=loss))
  test_writer.add_summary(summary_pb2.Summary(value=vals), trial)

  print('Overall accuracy {:0.3f}'.format(acc_avg))
  print('------------------------------------')
  sys.stdout.flush()

  print('Validation took : {:0.2f}s'.format(time.time() - val_start))
  return acc_list


def _fname_to_ds(input_files, batch_size):
  feed_types = (tf.float32, tf.int64, tf.int64, tf.float32, tf.float32,
                tf.int64, tf.float32, tf.float32, tf.string)

  ds = tf.data.TextLineDataset(filenames=input_files, compression_type='GZIP')
  ds = ds.apply(tf.contrib.data.batch_and_drop_remainder(batch_size))
  ds = ds.repeat()
  ds = ds.map(
        lambda examples: tuple(tf.py_func(
          tu.json_to_feeds, [examples], feed_types, stateful=False)))
  return ds


def get_ds_from_files(data_dir, data_type, batch_size):
  """Returns inputs tensors for data in data_dir.

  Args:
    data_type: one of 'train' or 'val'.
  """
  input_dir_glob = os.path.join(data_dir, data_type + '_*')
  input_dir = glob.glob(input_dir_glob)
  assert len(input_dir) == 1, ('Expected to find one ' + data_type +
      ' directory in ' + data_dir + ' but got ' + input_dir)
  input_dir = input_dir[0]
  input_files = glob.glob(os.path.join(input_dir, '*'))
  tf.logging.info('Found %s files: %s', data_type, input_files)

  feed_shapes = (tf.TensorShape([None, None, 112, 112, 3]),
                 tf.TensorShape([config['maxseqlength'], None]),
                 tf.TensorShape([None]),
                 tf.TensorShape([None, None]),
                 tf.TensorShape([None, None]),
                 tf.TensorShape([None]),
                 tf.TensorShape([None]),
                 tf.TensorShape([None]),
                 tf.TensorShape([None]))

  if data_type == 'val':
    input_ds = (tf.data.Dataset
        .from_tensor_slices(input_files)
        .interleave(functools.partial(_fname_to_ds, batch_size=batch_size),
                    cycle_length=len(input_files),
                    block_length=1))
  elif data_type == 'train':
    input_ds = _fname_to_ds(input_files, batch_size)
  else:
    raise RuntimeError('Unknown data_type: ' + str(data_type))


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


def get_inputs_from_files(data_dir):
  train_feeds_dict, _ = get_ds_from_files(data_dir, 'train', FLAGS.batch_size)
  val_feeds_dict, task_family_tensor = get_ds_from_files(data_dir, 'val',
      FLAGS.batch_size)
  return train_feeds_dict, val_feeds_dict, task_family_tensor


def get_inputs(train_task_families, task_families, hparams):
  if FLAGS.data_dir:
    return get_inputs_from_files(FLAGS.data_dir)

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


  train_ds = tf.data.Dataset.from_generator(
      jit_input_generator(train_task_families, hparams),
      feed_types,
      feed_shapes)
  train_ds = train_ds.prefetch(buffer_size=2)
  train_feeds = train_ds.make_one_shot_iterator().get_next()

  for i, v in enumerate(train_feeds):
    tf.logging.error("type of feeds %d: %s", i, v.dtype)

  train_feeds_dict = {'image': train_feeds[0],
                      'question': train_feeds[1],
                      'seq_len': train_feeds[2],
                      'point': train_feeds[3],
                      'point_xy': train_feeds[4],
                      'answer': train_feeds[5],
                      'mask_point': train_feeds[6],
                      'mask_answer': train_feeds[7],
                      }


  val_ds = tf.data.Dataset.from_generator(
      val_input_generator(task_families[:], hparams),
      feed_types + (tf.string,),
      feed_shapes + (tf.TensorShape([None]),))
  val_ds = val_ds.prefetch(buffer_size=2)
  val_feeds = val_ds.make_one_shot_iterator().get_next()
  val_feeds_dict = {'image': val_feeds[0],
                    'question': val_feeds[1],
                    'seq_len': val_feeds[2],
                    'point': val_feeds[3],
                    'point_xy': val_feeds[4],
                    'answer': val_feeds[5],
                    'mask_point': val_feeds[6],
                    'mask_answer': val_feeds[7],
                    }
  task_family_tensor = val_feeds[8]

  return train_feeds_dict, val_feeds_dict, task_family_tensor


def run_training(hparams, train_dir):
  """Train.

  Args:
    hparams: A HParam object with the hyperparameters to use.
    train_dir: Path of a directory where to log training events.
  """

  print('Hyperparameters:')
  for key, val in sorted(hparams.values().items()):
    print(key, val)

  ######################### Tasks to train ###################################
  if FLAGS.task_family == 'all':
    task_families = list(task_bank.task_family_dict.keys())
  else:
    task_families = [FLAGS.task_family]

  if hparams.exclude_task_train in task_families:
    train_task_families = task_families[:]
    print("All tasks: " + str(train_task_families))
    train_task_families.remove(hparams.exclude_task_train)
    print("Excluding task: " + hparams.exclude_task_train)
    print("Remaining tasks: " + str(train_task_families))
    # Add exclude_task to train_dir for easier mldash experience
    train_dir = os.path.join(train_dir, hparams.exclude_task_train)
  elif hparams.exclude_task_train == 'none':
    train_task_families = task_families
  else:
    raise RuntimeError("Invalid exclude_task_train value: " +
                       hparams.exclude_task_train)

  ######################### Create dir/files ##################################
  if not tf.gfile.Exists(train_dir):
    tf.gfile.MakeDirs(train_dir)
    #tf.gfile.DeleteRecursively(train_dir)

  with tf.gfile.FastGFile(os.path.join(train_dir, 'hparams'), 'w') as f:
    json.dump(hparams.to_json(), f)

  ######################### Build the model ##################################
  tf.reset_default_graph()

  train_feeds_dict, val_feeds_dict, task_family_tensor = get_inputs(
      train_task_families, task_families, hparams)

  tf.train.get_or_create_global_step()

  model = network.Model(hparams, config)
  model.build(train_feeds_dict, FLAGS.batch_size, is_training=True)

  if FLAGS.report_param_stat:
    ma = tf.contrib.tfprof.model_analyzer
    param_stats = tf.contrib.tfprof.model_analyzer.print_model_analysis(
        tf.get_default_graph(),
        tfprof_options=ma.TRAINABLE_VARS_PARAMS_STAT_OPTIONS)
    print('total_params: {:d}'.format(param_stats.total_parameters))

    # num params in best network for canonical COG
    PARAM_LIMIT = 5372889
    if param_stats.total_parameters > 1.1 * PARAM_LIMIT:
      raise tf.errors.ResourceExhaustedError(None, None,
          "Hyperparams resulted in too many params: %d" %
          param_stats.total_parameters)

  # Construct merged summaries before constructing the validation model
  # so that saving summaries does not run validataion dataset
  merged = tf.summary.merge_all()
  saver = tf.train.Saver()

  val_model = network.Model(hparams, config)
  val_model.build(val_feeds_dict, FLAGS.batch_size, is_training=True)


  ########################## For Tensorboard #################################

  test_writer = tf.summary.FileWriter(os.path.join(train_dir, 'tb'),
                                      flush_secs=120)

  ########################## Train the network ###############################
  print("Train dir: " + train_dir)
  checkpoint_path = train_dir + '/checkpoints'
  with tf.Session() as sess:
    # Initialize
    sess.run(tf.global_variables_initializer())  # initialize all variables

    cpkt_path = tf.train.latest_checkpoint(checkpoint_path)
    if cpkt_path is not None:
      print("Restoring model from: " + cpkt_path)
      saver.restore(sess, cpkt_path)
      print("Done restoring model")
    else:
      print("Did not find checkpoint at: " + checkpoint_path)

    t_start = time.time()
    next_family = 0
    global_step = sess.run(tf.train.get_global_step())
    print("Initial global step value: " + str(global_step))
    while global_step < FLAGS.num_steps:
      trial = global_step * FLAGS.batch_size

      if global_step > 0 and global_step % FLAGS.display_step == 0:

        print('Step {:d} Trial {:d}'.format(global_step, trial))
        print('Time taken: {:0.2f}s'.format(time.time() - t_start))
        save_path = saver.save(sess, checkpoint_path + '/model.ckpt',
                       global_step=tf.train.get_global_step())
        print('Model saved in file {:s}'.format(save_path))

        if global_step % (10 * FLAGS.display_step) == 0:
          # Do a high quality evaluation every 960k examples
          acc_list = evaluate(trial, sess, task_families, hparams,
                              test_writer, val_model,
                              task_family_tensor, num_batches=20)
        else:
          acc_list = evaluate(trial, sess, task_families, hparams,
                              test_writer, val_model,
                              task_family_tensor, num_batches=1)

      if global_step > 0 and global_step % FLAGS.summary_step == 0:
        global_step, summary, _ = sess.run([tf.train.get_global_step(),
                                            merged,
                                            model.train_step])
        test_writer.add_summary(summary, trial)
      else:
        global_step, _ = sess.run([tf.train.get_global_step(),
                                   model.train_step])

    print("Stopping at global step value: " + str(global_step))

    # Test the final accuracy and record it as the last summary point
    num_batches = 200
    print("Running final test over " +
          str(len(task_families) * num_batches * FLAGS.batch_size) +
          " examples")
    evaluate(trial + FLAGS.display_step * FLAGS.batch_size,
             sess, task_families, hparams, test_writer,
             val_model, task_family_tensor, num_batches=num_batches)
    test_writer.close()


def main(_):
  hparams_dict = get_default_hparams_dict()
  hparams = tf.contrib.training.HParams(**hparams_dict)
  hparams = hparams.parse(FLAGS.hparams)  # Overwritten by FLAGS.hparams

  if FLAGS.exclude_task:
    # Experiment to exclude one task family during training
    job_id = int(FLAGS.job_id)
    task_families = task_bank.task_family_dict.keys()
    hparams.exclude_task_train = task_families[job_id % len(task_families)]
    train_dir = os.path.join(FLAGS.train_dir, FLAGS.job_id)
  else:
    # Default experiment
    train_dir = FLAGS.train_dir

  run_training(hparams, train_dir)


if __name__ == '__main__':
  tf.app.run(main)
