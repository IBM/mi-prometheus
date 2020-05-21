# Copyright (C) IBM Corporation 2020
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#      http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
# Code was originally taken from https://github.com/google/cog :
#
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

"""Script for generating a COG dataset"""

import errno
import gzip
import json
import multiprocessing
import os
import random
import shutil

import numpy as np

import miprometheus.problems.seq_to_seq.video_text_to_class.cog.cog_utils.task_bank as task_bank


try:
  range_fn = xrange  # py 2
except NameError:
  range_fn = range  # py 3


def get_target_value(t):
  # Convert target t to string and convert True/False target values
  # to lower case strings for consistency with other uses of true/false
  # in vocabularies.
  t = t.value if hasattr(t, 'value') else str(t)
  if t is True or t == 'True':
    return 'true'
  if t is False or t == 'False':
    return 'false'
  return t


def generate_example(memory_length, max_distractors, task_family, epochs):
  #random.seed(1)
  task = task_bank.random_task(task_family)

  # To get maximum memory duration, we need to specify the following average
  # memory value
  avg_mem = round(memory_length/3.0 + 0.01, 2)
  objset = task.generate_objset(n_epoch=epochs,
                                n_distractor=random.randint(1, max_distractors),
                                average_memory_span=avg_mem)
  # Getting targets can remove some objects from objset.
  # Create example fields after this call.
  targets = task.get_target(objset)

  example = {
      'family': task_family,
      'epochs': epochs,  # saving an epoch explicitly is needed because
                         # there might be no objects in the last epoch.
      'question': str(task),
      'objects': [o.dump() for o in objset],
      'answers': [get_target_value(t) for t in targets]
  }
  return example, objset, task


class FileWriter(object):
  """Writes per_file examples in a file. Then, picks a new file."""
  def __init__(self, base_name, per_file=100, start_index=0, compress=True):
    self.per_file = per_file
    self.base_name = base_name
    self.compress = compress
    self.cur_file_index = start_index - 1
    self.cur_file = None
    self.written = 0
    self.file_names = []

    self._new_file()

  def _file_name(self):
    return '%s_%d.json' % (self.base_name, self.cur_file_index)

  def _new_file(self):
    if self.cur_file:
      self.close()

    self.written = 0
    self.cur_file_index += 1
    # 'b' is needed because we want to seek from the end. Text files
    # don't allow seeking from the end (because width of one char is
    # not fixed)
    self.cur_file = open(self._file_name(), 'wb')
    self.file_names.append(self._file_name())

  def write(self, data):
    if self.written >= self.per_file:
      self._new_file()
    self.cur_file.write(data)
    self.cur_file.write(b'\n')
    self.written += 1

  def close(self):
    self.cur_file.seek(-1, os.SEEK_END)
    # Remove last new line
    self.cur_file.truncate()
    self.cur_file.close()

    if self.compress:
      # Compress the file and delete the original. We can write to compressed
      # file immediately because truncate() does not work on compressed files.
      with open(self._file_name(), 'rb') as f_in, \
          gzip.open(self._file_name() + '.gz', 'wb') as f_out:
        shutil.copyfileobj(f_in, f_out)

      os.remove(self._file_name())


def mkdir(path):
  try:
    os.makedirs(path)
  except OSError as exc:
    if exc.errno == errno.EEXIST and os.path.isdir(path):
      pass
    else:
      raise


def generate_dataset(epochs, max_distractors, memory_length,
                     examples_per_task, output_dir,
                     random_families, start_index=0,
                     per_file=10000, compress=True):
  print("Generating dataset into %s:\n  examples per family=%d\n  epochs=%d"
        "\n  start_index=%d\n  per_file=%d"
        % (output_dir, examples_per_task, epochs, start_index, per_file))
  if not os.path.exists(output_dir):
    mkdir(output_dir)

  families = list(task_bank.task_family_dict.keys())
  n_families = len(families)
  total_examples = n_families * examples_per_task

  base_fname = os.path.join(output_dir, 'cog')

  if random_families:
    # Write tasks from random families to files.
    f = FileWriter(base_name=base_fname, per_file=per_file,
                   start_index=start_index, compress=compress)
    p = np.random.permutation(total_examples)
    for i, task_ind in enumerate(p):
      if i % 10000 == 0 and i > 0:
        print("Generated ", i, " examples")
      task_family = families[task_ind % n_families]
      example, _, _ = generate_example(memory_length, max_distractors,
                                       task_family, epochs)
      # Write the example to file
      dump_str = json.dumps(example, sort_keys=True, separators=(',', ': '))
      assert '\n' not in dump_str, 'dumps_str has new line %s' % (dump_str,)
      f.write(dump_str.encode())
    f.close()
    file_names = f.file_names
  else:
    # Write tasks for each task family into a separate file.
    file_names = []
    for family in families:
      fname = base_fname + '_' + family + '.json' + ('.gz' if compress else '')
      file_names.append(fname)
      open_fn = gzip.open if compress else open
      with open_fn(fname, 'wb') as f:
        for i in range_fn(examples_per_task):
          if i % 10000 == 0 and i > 0:
            print("Generated ", i, " examples")
          example, _, _ = generate_example(memory_length, max_distractors,
                                           family, epochs)
          # Write the example to file
          dump_str = json.dumps(example, sort_keys=True, separators=(',', ': '))
          assert '\n' not in dump_str, 'dumps_str has new line %s' % (dump_str,)
          f.write(dump_str.encode())
          if i != examples_per_task - 1:
            f.write(b'\n')

  print("Wrote dataset into:", file_names)


def generate_train(path,examples_per_task,sequence_length,memory_length,max_distractors,cog_variant,start_index=0,per_file=10000):

  output_dir = os.path.join(path, 'train_' + cog_variant)
  generate_dataset(sequence_length, max_distractors, memory_length,
                   examples_per_task, output_dir, random_families=True,
                   start_index=start_index, per_file=per_file,
                   compress=True)

def generate_val_or_test(path,examples_per_task,sequence_length,memory_length,max_distractors,nr_processors,data_type, cog_variant):
  # 20x smaller than training.
  output_dir = os.path.join(path, '%s_%s' % (data_type, cog_variant))
	
  generate_dataset(sequence_length, max_distractors, memory_length,
                   max(examples_per_task // 20, 50),
                   output_dir, random_families=False,
                   compress=True)


def main(path,examples_per_task,sequence_length,memory_length,max_distractors,nr_processors):
  cog_variant = '%d_%d_%d' % (sequence_length, memory_length, max_distractors)

  if nr_processors > 2:
    train_parallel = nr_processors - 2
    assert (examples_per_task % (2 * train_parallel)) == 0, (
      "examples_per_task must be a multiple of 2*(nr_processors - 2)")
    examples_per_task_per_job = examples_per_task // train_parallel

    pool = multiprocessing.Pool(processes=nr_processors)
    jobs = []
    n_tasks = len(task_bank.task_family_dict)
    for i in range_fn(train_parallel):
      jobs.append(pool.apply_async(generate_train, (
        path, examples_per_task_per_job, sequence_length,memory_length,max_distractors, cog_variant,
        2 * i, n_tasks * examples_per_task_per_job // 2)))
    jobs.append(pool.apply_async(generate_val_or_test, (path,examples_per_task,sequence_length,memory_length,max_distractors,nr_processors,'val', cog_variant)))
    jobs.append(pool.apply_async(generate_val_or_test, (path,examples_per_task,sequence_length,memory_length,max_distractors,nr_processors,'test', cog_variant)))

    # wait for all jobs to complete
    [j.get() for j in jobs]
  else:
    generate_train(path,examples_per_task,sequence_length,memory_length,max_distractors,cog_variant)
    generate_val_or_test(path,examples_per_task,sequence_length,memory_length,max_distractors,nr_processors,'val', cog_variant)
    generate_val_or_test(path,examples_per_task,sequence_length,memory_length,max_distractors,nr_processors,'test', cog_variant)

