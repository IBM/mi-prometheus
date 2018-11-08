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

"""Tests for cognitive.stim_generator."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import unittest

import numpy as np

from cognitive import stim_generator as sg
from cognitive import task_bank


class StimGeneratorTest(unittest.TestCase):

  def testAttributeEqual(self):
    color1 = sg.Color('red')
    color2 = sg.Color('green')
    color3 = sg.Color('red')

    self.assertFalse(color1 == color2)
    self.assertTrue(color1 == color3)

  def testMergeObject(self):
    obj1 = sg.Object([sg.Shape('circle')])
    obj2 = sg.Object([sg.Color('red')])

    merged = obj1.merge(obj2)
    self.assertTrue(merged)

    obj1 = sg.Object([sg.Shape('circle')])
    obj2 = sg.Object([sg.Shape('a')])

    merged = obj1.merge(obj2)
    self.assertFalse(merged)

  def testHasValue(self):
    loc = sg.Loc(None)
    shape = sg.Shape(None)
    color = sg.Color(None)
    space = sg.Space(None)

    self.assertFalse(loc.has_value)
    self.assertFalse(shape.has_value)
    self.assertFalse(color.has_value)
    self.assertFalse(space.has_value)

  def testSelectNow(self):
    objset = sg.ObjectSet(n_epoch=1)
    epoch_now = 0
    objset.add(sg.Object(when='now'), epoch_now)
    subset = objset.select(epoch_now, when='now')

    self.assertEqual(len(subset), 1)

  def testSelectNowLoc(self):
    objset = sg.ObjectSet(n_epoch=1)
    epoch_now = 0
    loc = sg.Loc([0.3, 0.3])
    objset.add(sg.Object([loc], when='now'), epoch_now)
    space1 = sg.Space([(0.2, 0.4), (0.1, 0.5)])
    space2 = sg.Space([(0.5, 0.7), (0.1, 0.5)])
    subset1 = objset.select(epoch_now, space=space1, when='now')
    subset2 = objset.select(epoch_now, space=space2, when='now')

    self.assertEqual(len(subset1), 1)
    self.assertEqual(len(subset2), 0)

  def testSelectLast(self):
    objset = sg.ObjectSet(n_epoch=2)
    objset.add(sg.Object(when='now'), epoch_now=0)
    objset.add(sg.Object(when='now'), epoch_now=1)
    objset.add(sg.Object(when='now'), epoch_now=1, add_if_exist=True)

    epoch_now = 1
    self.assertEqual(2, len(objset.select(epoch_now, when='latest')))
    self.assertEqual(1, len(objset.select(epoch_now, when='last1')))

  def testSelectBackTrack(self):
    objset = sg.ObjectSet(n_epoch=100)
    objset.add(sg.Object(when='now'), epoch_now=5)
    epoch_now = 10
    l1 = len(objset.select(epoch_now, when='latest', n_backtrack=4))
    self.assertEqual(l1, 0)

    objset = sg.ObjectSet(n_epoch=100)
    objset.add(sg.Object(when='now'), epoch_now=5)
    epoch_now = 10
    l1 = len(objset.select(epoch_now, when='latest', n_backtrack=5))
    self.assertEqual(l1, 1)

  def testLastAdded(self):
    objset = sg.ObjectSet(n_epoch=100)
    objset.add(sg.Object(when='now'), epoch_now=5)
    self.assertEqual(objset.last_added_obj.epoch, [5, 6])

  def testAddWhenNone(self):
    objset = sg.ObjectSet(n_epoch=100, n_max_backtrack=5)
    objset.add(sg.Object(when=None), epoch_now=0)
    self.assertEqual(objset.last_added_obj.epoch, [0, 100])

    l1 = len(objset.select(epoch_now=10, when='now'))
    l2 = len(objset.select(epoch_now=10, when='latest'))
    l3 = len(objset.select(epoch_now=10, when='last1'))
    self.assertEqual(l1, 1)
    self.assertEqual(l2, 1)
    self.assertEqual(l3, 1)

  def testAddIfExist(self):
    objset = sg.ObjectSet(n_epoch=1)
    epoch_now = 0
    objset.add(sg.Object(when='now'), epoch_now, add_if_exist=False)
    objset.add(sg.Object(when='now'), epoch_now, add_if_exist=False)

    self.assertEqual(len(objset), 1)

  def testAddFixLoc(self):
    loc = [0.2, 0.8]
    objset = sg.ObjectSet(n_epoch=1)
    obj = sg.Object([sg.Loc(loc)], when='now')
    objset.add(obj, epoch_now=0)
    self.assertEqual(list(obj.loc.value), loc)

  def testDeletable(self):
    objset = sg.ObjectSet(n_epoch=1)
    epoch_now = 0
    objset.add(sg.Object(when='now', deletable=True), epoch_now)
    objset.add(sg.Object(when='now'), epoch_now, add_if_exist=True)

    # The first object should have been deleted
    self.assertEqual(len(objset), 1)

  def testAddLast1(self):
    objset = sg.ObjectSet(n_epoch=100)
    objset.add(sg.Object(when='now'), epoch_now=0)
    objset.add(sg.Object(when='now', deletable=True), epoch_now=1)
    objset.add(sg.Object(when='last1'), epoch_now=2, add_if_exist=False)

    self.assertEqual(len(objset), 1)

  def testAddDistractor(self):
    n_epoch = 30
    objset = sg.ObjectSet(n_epoch=n_epoch)

    # Guess objects
    for epoch_now in range(n_epoch):
      objset.add_distractor(epoch_now)  # distractor

    self.assertEqual(len(objset), n_epoch)

    # Make sure all distractors are deleted if they enter the select process
    for epoch_now in range(n_epoch):
      _ = objset.select(epoch_now, when='now')

    self.assertEqual(len(objset), 0)

  def testShiftObjset(self):
    objset = sg.ObjectSet(n_epoch=2)
    epoch_now = 0
    objset.add(sg.Object(when='now'), epoch_now)

    objset.shift(1)
    self.assertEqual(objset.n_epoch, 3)
    self.assertEqual(objset.set[0].epoch, [1, 2])
    subset = objset.select(1, when='now')
    self.assertEqual(len(subset), 1)

    objset.shift(-2)
    self.assertEqual(len(objset), 0)

  def testAddObjectInSpace(self):
    objset = sg.ObjectSet(n_epoch=1)
    space1 = sg.Space([(0, 1), (0, 0.5)])
    space2 = sg.Space([(0, 1), (0.5, 1)])
    space3 = sg.Space([(0, 1), (0, 0.5)])
    epoch_now = 0
    objset.add(sg.Object([space1], when='now'), epoch_now)
    self.assertEqual(len(objset), 1)
    objset.add(sg.Object([space2], when='now'), epoch_now)
    self.assertEqual(len(objset), 2)
    objset.add(sg.Object([space3], when='now'), epoch_now)
    self.assertEqual(len(objset), 2)

  def testAnotherAttr(self):
    color = sg.Color('red')
    shape = sg.Shape('circle')
    space = sg.Space([(0.3, 0.7), (0.4, 0.7)])
    for i in range(1000):
      self.assertNotEqual(sg.another_attr(color), color)

    for i in range(1000):
      self.assertNotEqual(sg.another_attr(shape), shape)

    for i in range(1000):
      self.assertFalse(space.include(sg.another_attr(space)))

  def testGetSpaceTo(self):
    space1 = sg.Space([(0.3, 0.7), (0.4, 0.7)])
    space2 = space1.get_space_to('left')
    space3 = space1.get_space_to('top')
    self.assertTrue(space1.value[0][0] >= space2.value[0][1])
    self.assertTrue(space1.value[1][0] >= space3.value[1][1])

  def testCopyAttributes(self):
    attrs1 = [sg.Space(None), sg.Color('red'), sg.Shape('blue')]
    attrs2 = copy.copy(attrs1)
    attrs2[1] = sg.Color('blue')
    self.assertEqual(attrs1[1], sg.Color('red'))
    self.assertEqual(attrs2[1], sg.Color('blue'))

  def testRenderEquivalency(self):
    memory = 3
    distractors = 5
    epochs = 6
    iters = 10
    batch_size = 2

    #np.set_printoptions(threshold=np.nan, linewidth=120)
    #movie_fname = '/tmp/render_movie'
    movie_fname = None

    for task_family in task_bank.task_family_dict.keys():
      print("Testing family: ", task_family)
      for i in range(iters):
        objsets = []
        for batch in range(batch_size):
          # Generate a task and objset
          task = task_bank.random_task(task_family)
          objset = task.generate_objset(n_epoch=epochs,
                                        n_distractor=distractors,
                                        average_memory_span=memory)
          # This call is necessary because objects can be removed from
          # objsets when we calculate targets.
          task.get_target(objset)
          objsets.append(objset)

        # Make movie from objects
        from_obj_movie = sg.render(objsets, img_size=112,
            save_name=movie_fname + "_obj.avi" if movie_fname else None)

        static_objsets = []
        for objset in objsets:
          json_objs = [o.dump() for o in objset]
          static_objs = [o for d in json_objs
                         for o in sg.static_objects_from_dict(d)]
          static_objset = sg.StaticObjectSet(n_epoch=epochs,
                                             static_objects=static_objs)
          static_objsets.append(static_objset)

        # Make movie from static objects
        from_static_movie = sg.render(static_objsets, img_size=112,
            save_name=movie_fname + "_static.avi" if movie_fname else None)

        om = from_obj_movie
        sm = from_static_movie

        # Check that both movies are the same
        np.testing.assert_array_almost_equal(from_obj_movie,
                                             from_static_movie)



if __name__ == '__main__':
  unittest.main()
