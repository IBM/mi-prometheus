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

"""Training utility functions."""

from six import string_types
import re
import numpy as np

from miprometheus.problems.seq_to_seq.video_text_to_class.cog.cog_utils import stim_generator as sg
from miprometheus.problems.seq_to_seq.video_text_to_class.cog.cog_utils import constants as const

_R_MEAN = 123.68
_G_MEAN = 116.78
_B_MEAN = 103.94


def convert_to_grid(xy_coord, prefs):
  """Given a x-y coordinate, return the target activity for a grid of neurons.

  Args:
    xy_coord : numpy 2-D array (batch_size, 2)
    prefs: numpy 2-D array (n_out_pnt, 2). x and y preferences.

  Returns:
    activity: numpy array (batch_size, GRID_SIZE**2)
  """
  sigma2 = 0.02  # 2*sigma-squared
  activity = np.exp(-((xy_coord[:, 0:1] - prefs[:, 0])**2 +
                      (xy_coord[:, 1:2] - prefs[:, 1])**2) / sigma2)
  activity = (activity.T / np.sum(activity, axis=1)).T
  return activity

def tasks_to_rules(tasks):
  """Generate in_rule and seq_length arrays.

  Args:
    tasks: a list of tg.Task instances or string rules, length is batch_size.
  """
  batch_size = len(tasks)
  in_rule = np.zeros((const.MAXSEQLENGTH, batch_size), dtype=np.int64)
  seq_length = np.zeros((batch_size,), dtype=np.int64)
  for i_task, task in enumerate(tasks):
    word_list = re.findall(r"[\w']+|[.,!?;]", str(task))
    seq_length[i_task] = len(word_list)
    for i_word, word in enumerate(word_list):
      in_rule[i_word, i_task] = const.INPUTVOCABULARY.index(word)
  return in_rule, seq_length


def set_outputs_from_tasks(n_epoch, tasks, objsets,
                           out_pnt_xy, out_word,
                           mask_pnt, mask_word):
  j = 0
  for epoch_now in range(n_epoch):
    for task, objset in zip(tasks, objsets):
      target = task(objset, epoch_now)
      if target == const.INVALID:
        # For invalid target, no loss is used. Everything remains zero.
        pass
      elif isinstance(target, sg.Loc):
        # minimize point loss
        out_pnt_xy[j, :] = target.value
        mask_pnt[j] = 1.
      elif isinstance(target, bool) or isinstance(target, sg.Attribute):
        if isinstance(target, bool):
          target = 'true' if target else 'false'
        else:
          target = target.value
        # For boolean target, only minimize word loss
        out_word[j] = const.OUTPUTVOCABULARY.index(target)
        mask_word[j] = 1.
      else:
        raise TypeError('Unknown target type.')
      j += 1


def set_outputs_from_targets(n_epoch, objsets,
                             out_pnt_xy, out_word,
                             mask_pnt, mask_word):
  j = 0
  for epoch_now in range(n_epoch):
    for objset in objsets:
      target = objset.targets[epoch_now]
      if target == 'invalid':
        # For invalid target, no loss is used. Everything remains zero.
        pass
      elif isinstance(target, (list, tuple)):
        assert len(target) == 2, "Expected 2-D target. Got " + str(target)
        # minimize point loss
        out_pnt_xy[j, :] = target
        mask_pnt[j] = 1.
      elif isinstance(target, string_types):
        out_word[j] = const.OUTPUTVOCABULARY.index(target)
        mask_word[j] = 1.
      else:
        raise TypeError('Unknown target type: %s %s' % (type(target), target))
      j += 1


def generate_batch(tasks,
                   n_epoch=30,
                   img_size=224,
                   objsets=None,
                   n_distractor=1,
                   average_memory_span=2):
  """Generate a batch of trials.

  Return numpy arrays to feed the tensorflow placeholders.

  Args:
    tasks: a list of tg.Task instances, length is batch_size.
    n_epoch: int, number of epochs
    img_size: int, image size
    objsets: None or list of ObjectSet/StaticObjectSet instances
    n_distractor: int, number of distractors to add
    average_memory_span: int, the average number of epochs by which an object
      need to be held in working memory, if needed at all

  Returns:
    All variables are numpy array of float32
    in_imgs: (n_epoch*batch_size, img_size, img_size, 3)
    in_rule: (max_seq_length, batch_size) the rule language input, type int32
    seq_length: (batch_size,) the length of each task instruction
    out_pnt: (n_epoch*batch_size, n_out_pnt)
    out_pnt_xy: (n_epoch*batch_size, 2)
    out_word: (n_epoch*batch_size, n_out_word)
    mask_pnt: (n_epoch*batch_size)
    mask_word: (n_epoch*batch_size)

  Raises:
    TypeError: when target type is incorrect.
  """
  batch_size = len(tasks)

  if objsets is None:
    objsets = list()
    for task in tasks:
      objsets.append(
          task.generate_objset(n_epoch,
                               n_distractor=n_distractor,
                               average_memory_span=average_memory_span))

  max_objset_epoch = max([objset.n_epoch for objset in objsets])
  assert max_objset_epoch == n_epoch, '%d != %d' % (max_objset_epoch, n_epoch)

  in_imgs = sg.render(objsets, img_size)
  # The rendered images are batch major
  in_imgs = np.reshape(in_imgs, [batch_size, n_epoch, img_size, img_size, 3])
  # Swap to time major
  in_imgs = np.swapaxes(in_imgs, 0, 1)

  # Outputs and masks
  out_pnt_xy = np.zeros((n_epoch * batch_size, 2), dtype=np.float32)
  out_word = np.zeros((n_epoch * batch_size), dtype=np.int64)
  mask_pnt = np.zeros((n_epoch * batch_size), dtype=np.float32)
  mask_word = np.zeros((n_epoch * batch_size), dtype=np.float32)

  if isinstance(objsets[0], sg.StaticObjectSet):
    set_outputs_from_targets(n_epoch, objsets,
                             out_pnt_xy, out_word,
                             mask_pnt, mask_word)
  else:
    set_outputs_from_tasks(n_epoch, tasks, objsets,
                           out_pnt_xy, out_word,
                           mask_pnt, mask_word)

  # Process outputs
  out_pnt = convert_to_grid(out_pnt_xy, const.PREFS)

  # Generate rule inputs, padded to maximum number of words in a sentence
  in_rule, seq_length = tasks_to_rules(tasks)

  return (in_imgs, in_rule, seq_length, out_pnt, out_pnt_xy, out_word, mask_pnt,
          mask_word)


def static_objsets_from_examples(examples):
  """Returns a list of StaticObjectSet objects.

  Args:
    examples: an iterable of dictionaries decoded from json examples.
  """
  static_objsets = []
  for e in examples:
    static_objs = [o for multi_epoch_obj in e['objects']
                   for o in sg.static_objects_from_dict(multi_epoch_obj)]
    static_objset = sg.StaticObjectSet(n_epoch=e['epochs'],
                                       static_objects=static_objs,
                                       targets=e['answers'])
    static_objsets.append(static_objset)
  return static_objsets


def json_to_feeds(json_examples):
  if isinstance(json_examples, string_types):
    json_examples = [json_examples]

  examples = []
  families = []
  rules = []

  for entry in json_examples:
    rules.append(entry['question'])
    examples.append(entry)
    families.append(entry['family'])

  epochs = examples[0]['epochs']
  static_objsets = static_objsets_from_examples(examples)

  values = generate_batch(rules, n_epoch=epochs,
                        img_size=112, objsets=static_objsets,
                        # not used when objsets are given
                        n_distractor=0,
                        # not used when objsets are given
                        average_memory_span=0)

  values = values + (families,)
  return values

