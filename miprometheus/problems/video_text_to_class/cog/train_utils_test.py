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
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import itertools
import json
import numpy as np
import os
import tensorflow as tf
import traceback
import unittest

from cognitive import stim_generator as sg
from cognitive import generate_dataset as gd
from cognitive import train_utils as tu
from cognitive import task_bank


class TrainUtilsTest(unittest.TestCase):

  def testEquivalent(self):
    memory = 3
    distractors = 6
    epochs = 6
    batch_size = 7
    iters = 20

    np.set_printoptions(threshold=np.nan, linewidth=120)
    tf.logging.set_verbosity(tf.logging.ERROR)

    # Generate tasks and write them to file.
    # Also, remember the tasks for feed generation.
    objsets = []
    tasks = []
    f = gd.FileWriter(base_name='/tmp/cog_test', per_file=100)
    families = task_bank.task_family_dict.keys()
    families.sort()
    for i in range(iters):
      for task_family in families:
        example, objset, task = gd.generate_example(memory, distractors,
                                                    task_family, epochs)
        tasks.append(task)
        objsets.append(objset)

        # Write the example to file
        dump_str = json.dumps(example, sort_keys=True, separators=(',', ': '))
        assert '\n' not in dump_str, 'dump_str has new line %s' % (dump_str,)
        f.write(dump_str)
    f.close()

    # Read the examples using tf.data
    ds = tf.data.TextLineDataset(filenames=f.file_names,
                                 compression_type=None)
    ds = ds.batch(batch_size)

    feed_types = [tf.float32, tf.int64, tf.int64, tf.float32, tf.float32,
        tf.int64, tf.float32, tf.float32, tf.string]
    ds = ds.map(
        lambda examples: tuple(tf.py_func(
          tu.json_to_feeds, [examples], feed_types, stateful=False)))
    iterator = ds.make_one_shot_iterator()
    next_element = iterator.get_next()

    with tf.Session() as sess:
      for i in itertools.count():
        if i % 100 == 0:
          print("Iteration:", i)
        try:
          (in_imgs_s, in_rule_s, seq_length_s, out_pnt_s, out_pnt_xy_s,
           out_word_s, mask_pnt_s, mask_word_s, task_families) = sess.run(
               next_element)
        except tf.errors.OutOfRangeError:
          return

        # Mimic how we generate feed data from tasks.
        # The only difference from real training is that we pass objset
        # explicitly because their generation is non-deterministic.
        (in_imgs, in_rule, seq_length, out_pnt, out_pnt_xy,
            out_word, mask_pnt, mask_word) = tu.generate_batch(
                tasks[(batch_size * i):(batch_size * (i + 1))],
                n_epoch=epochs,
                img_size=112,
                objsets=objsets[(batch_size * i):(batch_size * (i + 1))],
                n_distractor=distractors,
                average_memory_span=memory)

        try:
          np.testing.assert_array_almost_equal(in_imgs, in_imgs_s)
          np.testing.assert_array_equal(in_imgs, in_imgs_s)
          np.testing.assert_array_almost_equal(in_rule, in_rule_s)
          np.testing.assert_array_equal(in_rule, in_rule_s)
          np.testing.assert_array_almost_equal(seq_length, seq_length_s)
          np.testing.assert_array_equal(seq_length, seq_length_s)
          np.testing.assert_array_almost_equal(out_pnt, out_pnt_s)
          # out_pnt precision is not perfect
          # np.testing.assert_array_equal(out_pnt, out_pnt_s)
          np.testing.assert_array_almost_equal(out_pnt_xy, out_pnt_xy_s)
          np.testing.assert_array_equal(out_pnt_xy, out_pnt_xy_s)
          np.testing.assert_array_almost_equal(out_word, out_word_s)
          np.testing.assert_array_equal(out_word, out_word_s)
          np.testing.assert_array_almost_equal(mask_pnt, mask_pnt_s)
          np.testing.assert_array_equal(mask_pnt, mask_pnt_s)
          np.testing.assert_array_almost_equal(mask_word, mask_word_s)
          np.testing.assert_array_equal(mask_pnt, mask_pnt_s)
        except AssertionError:
          traceback.print_exc()

          movie_fname = '/tmp/render_movie'
          in_imgs_r = np.reshape(in_imgs,
                                 [batch_size * epochs, 112, 112, 3])
          in_imgs_s_r = np.reshape(in_imgs_s,
                                   [batch_size * epochs, 112, 112, 3])
          sg.save_movie(
              in_imgs, movie_fname + "_obj.avi",
              float(in_imgs.shape[0]))
          sg.save_movie(
              in_imgs_s, movie_fname + "_static.avi",
              float(in_imgs_s.shape[0]))
          raise

  def testTasksToRules(self):
    memory = 3
    distractors = 1
    epochs = 2

    # Generate a task
    for task_family in task_bank.task_family_dict.keys():
      task1 = task_bank.random_task(task_family)
      task2 = task_bank.random_task(task_family)
      rule1 = str(task1)
      rule2 = str(task2)
      from_tasks = tu.tasks_to_rules([task1, task2])
      from_rules = tu.tasks_to_rules([rule1, rule2])
      np.testing.assert_array_equal(from_tasks[0], from_rules[0])
      np.testing.assert_array_equal(from_tasks[1], from_rules[1])


if __name__ == '__main__':
  unittest.main()
