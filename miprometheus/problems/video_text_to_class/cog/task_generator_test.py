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

"""Tests for cognitive/task_generator.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import random
import numpy as np
from collections import defaultdict, OrderedDict
import unittest

from cognitive import constants as const
from cognitive import stim_generator as sg
from cognitive import task_generator as tg
from cognitive import task_bank as tb


def targets_to_str(targets):
  return [t.value if hasattr(t, 'value') else str(t) for t in targets]


class TaskGeneratorTest(unittest.TestCase):

  def testHashing(self):
    should_be_dict = dict()
    op = tg.Select(shape=sg.Shape('circle'), when='last1')
    should_be_dict[op] = 1
    self.assertEqual(should_be_dict[op], 1)

  def testGetAllNodes(self):
    objs = tg.Select()
    color = tg.GetColor(objs)
    task = tg.Task(color)

    all_nodes = task._all_nodes
    for op in [objs, color]:
      self.assertIn(op, all_nodes)

  def testTopoSort(self):
    objs = tg.Select()
    color = tg.GetColor(objs)
    task = tg.Task(color)

    sorted_nodes = task.topological_sort()
    self.assertListEqual(sorted_nodes, [color, objs])

  def testIsFinal(self):
    objs1 = tg.Select(
        color=sg.Color('red'), shape=sg.Shape('square'), when='now')
    objs2 = tg.Select(
        color=sg.Color('blue'), shape=sg.Shape('square'), when='now')
    objs3 = tg.Select(shape=sg.Shape('circle'), when='last1')
    bool1 = tg.Exist(objs1)
    go1 = tg.Go(objs2)
    go2 = tg.Go(objs3)
    op = tg.Switch(bool1, go1, go2, both_options_avail=False)

    self.assertFalse(op.parent)
    self.assertTrue(bool1.parent)
    self.assertTrue(go1.parent)

  def testSelectCall(self):
    objset = sg.ObjectSet(n_epoch=10)
    attrs = [sg.Loc([0.5, 0.5]), sg.Shape('circle'), sg.Color('red')]
    objset.add(sg.Object(attrs, when='now'), epoch_now=1)

    select = tg.Select(color=sg.Color('red'), when='now')
    self.assertTrue(select(objset, epoch_now=1))
    self.assertFalse(select(objset, epoch_now=2))
    select = tg.Select(color=sg.Color('blue'), when='now')
    self.assertFalse(select(objset, epoch_now=1))

    select = tg.Select(shape=sg.Shape('circle'), when='now')
    self.assertTrue(select(objset, epoch_now=1))
    self.assertFalse(select(objset, epoch_now=2))

    select = tg.Select(loc=sg.Loc([0.6, 0.6]), when='now', space_type='left')
    self.assertTrue(select(objset, epoch_now=1))
    select = tg.Select(loc=sg.Loc([0.6, 0.6]), when='now', space_type='top')
    self.assertTrue(select(objset, epoch_now=1))

    select = tg.Select(color=sg.Color('red'), when='last1')
    self.assertFalse(select(objset, epoch_now=1))
    self.assertTrue(select(objset, epoch_now=2))

    select = tg.Select(color=sg.Color('red'), when='latest')
    self.assertTrue(select(objset, epoch_now=1))
    self.assertTrue(select(objset, epoch_now=2))

    attrs = [sg.Loc([0.7, 0.7]), sg.Shape('square'), sg.Color('red')]
    objset.add(sg.Object(attrs, when='now'), epoch_now=1)
    select = tg.Select(color=sg.Color('red'), when='latest')
    self.assertEqual(len(select(objset, epoch_now=1)), 2)

  def testSelectGetExpectedInputShouldBeTrue1(self):
    objset = sg.ObjectSet(n_epoch=10)
    select = tg.Select(color=sg.Color('red'), when='now')
    should_be = [sg.Object([sg.Loc([0.5, 0.5])])]
    objset, loc, color, space = select.get_expected_input(
      should_be, objset, epoch_now=1)
    objs = select(objset, epoch_now=1)
    self.assertTupleEqual(objs[0].loc.value, (0.5, 0.5))

  def testSelectGetExpectedInputShouldBeTrue2(self):
    objset = sg.ObjectSet(n_epoch=10)
    select = tg.Select(color=sg.Color('red'), when='now')
    should_be = [sg.Object([sg.Shape('circle')])]
    objset, loc, color, space = select.get_expected_input(
      should_be, objset, epoch_now=1)
    objs = select(objset, epoch_now=1)
    self.assertEqual(objs[0].shape, sg.Shape('circle'))

  def testSelectGetExpectedInputShouldBeTrue3(self):
    objset = sg.ObjectSet(n_epoch=10)
    select = tg.Select(color=sg.Color('red'), when='now')
    should_be = [sg.Object(when='now')]
    objset, loc, color, space = select.get_expected_input(
      should_be, objset, epoch_now=1)
    objset, loc, color, space = select.get_expected_input(
      should_be, objset, epoch_now=1)
    objs = select(objset, epoch_now=1)
    self.assertEqual(len(objs), 1)

  def testSelectGetExpectedInputShouldBeTrue4(self):
    objset = sg.ObjectSet(n_epoch=10)
    select = tg.Select(loc=sg.Loc([0.5, 0.5]), when='now', space_type='left')
    should_be = [sg.Object(when='now')]
    objset, loc, color, space = select.get_expected_input(
      should_be, objset, epoch_now=1)
    objs = select(objset, epoch_now=1)
    self.assertLess(objs[0].loc.value[0], 0.5)
    self.assertTrue(isinstance(loc, tg.Skip))

  def testSelectGetExpectedInputShouldBeEmpty1(self):
    objset = sg.ObjectSet(n_epoch=10)
    select = tg.Select(color=sg.Color('red'), when='now')
    should_be = []
    objset, loc, color, space = select.get_expected_input(
      should_be, objset, epoch_now=1)
    objs = select(objset, epoch_now=1)
    self.assertFalse(objs)

  def testSelectGetExpectedInputShouldBeEmpty2(self):
    objset = sg.ObjectSet(n_epoch=10)
    select = tg.Select(loc=sg.Loc([0.5, 0.5]), when='now', space_type='left')
    should_be = []
    objset, loc, color, space = select.get_expected_input(
      should_be, objset, epoch_now=1)
    objs = select(objset, epoch_now=1)
    self.assertFalse(objs)
    self.assertEqual(len(objset), 1)

  def testGetCall(self):
    objset = sg.ObjectSet(n_epoch=10)
    obj1 = tg.Select(color=sg.Color('red'), when='now')
    color1 = tg.GetColor(obj1)

    epoch_now = 1
    color1_eval = color1(objset, epoch_now)
    self.assertEqual(color1_eval, const.INVALID)

    objset.add(sg.Object([sg.Color('red')], when='now'), epoch_now)
    color1_eval = color1(objset, epoch_now)
    self.assertEqual(color1_eval, sg.Color('red'))

  def testGetGuessObjset(self):
    objset = sg.ObjectSet(n_epoch=10)
    objs = tg.Select()
    task = tg.Task(tg.GetColor(objs))
    epoch_now = 1
    objset = task.guess_objset(objset, epoch_now)
    l = len(objset.select(epoch_now, when='now'))
    self.assertEqual(1, l)

  def testGetGuessObjsetShouldBe(self):
    objset = sg.ObjectSet(n_epoch=10)
    objs = tg.Select(when='now')
    task = tg.Task(tg.GetColor(objs))

    for epoch_now in range(10):
      color = sg.random_color()
      objset = task.guess_objset(objset, epoch_now, should_be=color)
      o = objset.select(epoch_now, when='now')[0]
      self.assertEqual(color, o.color)

  def testGetGuessObjsetLast1(self):
    objs1 = tg.Select(color=sg.Color('red'), when='last1')
    task = tg.Task(tg.GetShape(objs1))

    n_epoch = 10
    objset = task.generate_objset(
      n_epoch, n_distractor=0, average_memory_span=2)
    target = [task(objset, epoch_now) for epoch_now in range(n_epoch)]

    n_invalid = sum([t == const.INVALID for t in target])

    self.assertLessEqual(n_invalid, 1)

  def testGoGuessObjset(self):
    objset = sg.ObjectSet(n_epoch=10)
    objs1 = tg.Select(shape=sg.Shape('square'), when='now')
    task = tg.Task(tg.Go(objs1))
    epoch_now = 1
    objset = task.guess_objset(objset, epoch_now)
    l1 = len(objset.select(epoch_now, shape=sg.Shape('square'), when='now'))
    l2 = len(objset.select(epoch_now, shape=sg.Shape('circle'), when='now'))
    self.assertEqual(1, l1)
    self.assertEqual(0, l2)

  def testGetTimeCall(self):

    obj1 = tg.Select(color=sg.Color('red'), when='latest')
    time1 = tg.GetTime(obj1)

    n_epoch = 10
    objset = sg.ObjectSet(n_epoch=n_epoch, n_max_backtrack=100)

    epoch_add = 1
    time1_eval = time1(objset, epoch_add)
    self.assertEqual(time1_eval, const.INVALID)

    objset.add(sg.Object([sg.Color('red')], when='now'), epoch_add)
    for epoch_now in range(epoch_add, n_epoch-1):
      time1_eval = time1(objset, epoch_now)
      self.assertEqual(time1_eval, epoch_add)

  def testExistGuessObjset(self):
    objset = sg.ObjectSet(n_epoch=10)
    objs1 = tg.Select(color=sg.Color('red'), when='now')
    task = tg.Task(tg.Exist(objs1))
    epoch_now = 1
    objset = task.guess_objset(objset, epoch_now, should_be=True)
    l1 = len(objset.select(epoch_now, color=sg.Color('red'), when='now'))
    l2 = len(objset.select(epoch_now, color=sg.Color('blue'), when='now'))
    self.assertEqual(1, l1)
    self.assertEqual(0, l2)
    self.assertTrue(task(objset, epoch_now))

    epoch_now = 2
    objset = task.guess_objset(objset, epoch_now, should_be=False)
    l1 = len(objset.select(epoch_now, color=sg.Color('red'), when='now'))
    self.assertEqual(0, l1)
    self.assertFalse(task(objset, epoch_now))

  def testExistSpaceGuessObjset(self):
    # When should_be is False, there is a
    objset = sg.ObjectSet(n_epoch=10)
    objs1 = tg.Select(color=sg.Color('red'), when='now')
    loc = tg.GetLoc(objs1)
    objs2 = tg.Select(loc=loc, color=sg.Color('blue'),
                      when='now', space_type='left')
    task = tg.Task(tg.Exist(objs2))

    for epoch_now in range(1, 10)[::-1]:
      should_be = random.random() > 0.5
      objset = task.guess_objset(objset, epoch_now, should_be=should_be)
      self.assertEqual(task(objset, epoch_now), should_be)

  def testBasicIsSameGuessObject(self):
    objset = sg.ObjectSet(n_epoch=10)
    objs1 = tg.Select(shape=sg.Shape('square'), when='now')
    attr1 = tg.GetColor(objs1)
    task = tg.Task(tg.IsSame(attr1, sg.Color('red')))
    epoch_now = 1
    objset = task.guess_objset(objset, epoch_now, should_be=True)
    c1 = objset.select(epoch_now, shape=sg.Shape('square'), when='now')[0].color
    self.assertEqual(c1, sg.Color('red'))

  def testIsSameGuessObjset(self):
    objset = sg.ObjectSet(n_epoch=10)
    objs1 = tg.Select(shape=sg.Shape('square'), when='now')
    objs2 = tg.Select(shape=sg.Shape('circle'), when='now')
    attr1 = tg.GetColor(objs1)
    attr2 = tg.GetColor(objs2)
    task = tg.Task(tg.IsSame(attr1, attr2))

    epoch_now = 1
    objset = task.guess_objset(objset, epoch_now, should_be=True)
    self.assertTrue(task(objset, epoch_now))
    c1 = objset.select(epoch_now, shape=sg.Shape('square'), when='now')[0].color
    c2 = objset.select(epoch_now, shape=sg.Shape('circle'), when='now')[0].color
    self.assertEqual(c1, c2)

  def testIsSameGuessObjsetLast1(self):
    objset = sg.ObjectSet(n_epoch=10)
    objs1 = tg.Select(shape=sg.Shape('square'), when='last1')
    objs2 = tg.Select(shape=sg.Shape('circle'), when='last1')
    attr1 = tg.GetColor(objs1)
    attr2 = tg.GetColor(objs2)
    task = tg.Task(tg.IsSame(attr1, attr2))

    objset = task.guess_objset(objset, epoch_now=1, should_be=True)
    self.assertEqual(2, len(objset.select(epoch_now=0, when='now')))
    c1 = objset.select(
      epoch_now=0, shape=sg.Shape('square'), when='now')[0].color
    c2 = objset.select(
      epoch_now=0, shape=sg.Shape('circle'), when='now')[0].color
    self.assertEqual(c1, c2)

  def testAndGuessObjset(self):
    objs1 = tg.Select(when='last1')
    objs2 = tg.Select(when='now')
    s1 = tg.GetShape(objs1)
    s2 = tg.GetShape(objs2)
    c1 = tg.GetColor(objs1)
    c2 = tg.GetColor(objs2)
    task = tg.Task(tg.And(tg.IsSame(s1, s2), tg.IsSame(c1, c2)))

    objset = sg.ObjectSet(n_epoch=10)
    obj0 = sg.Object([sg.Color('green'), sg.Shape('square')], when='now')
    obj1 = sg.Object([sg.Color('red'), sg.Shape('circle')], when='now')
    objset.add(obj0, epoch_now=0)
    objset.add(obj1, epoch_now=1)
    objset = task.guess_objset(objset, epoch_now=2, should_be=True)
    obj2 = objset.last_added_obj
    self.assertEqual(obj1.color.value, obj2.color.value)
    self.assertEqual(obj1.shape.value, obj2.shape.value)

  def testAndCompareColorGuessObjset(self):
    objs1 = tg.Select(shape=sg.Shape('circle'), when='last1')
    objs2 = tg.Select(shape=sg.Shape('square'), when='last1')
    objs3 = tg.Select(shape=sg.Shape('triangle'), when='last1')
    objs4 = tg.Select(shape=sg.Shape('hbar'), when='last1')
    a1 = tg.Get('color', objs1)
    a2 = tg.Get('color', objs2)
    a3 = tg.Get('color', objs3)
    a4 = tg.Get('color', objs4)
    task = tg.Task(tg.And(tg.IsSame(a1, a2), tg.IsSame(a3, a4)))

    n_epoch = 10
    for i in range(100):
      epoch_now = random.randint(1, n_epoch-1)
      objset = sg.ObjectSet(n_epoch=n_epoch)
      should_be = random.random() > 0.5
      objset = task.guess_objset(objset, epoch_now, should_be=should_be)
      self.assertEqual(task(objset, epoch_now), should_be)

  def testAndOperatorSize(self):
    objs1 = tg.Select(when='last1')
    objs2 = tg.Select(when='now')
    s1 = tg.GetShape(objs1)
    s2 = tg.GetShape(objs2)
    c1 = tg.GetColor(objs1)
    c2 = tg.GetColor(objs2)
    and1 = tg.And(tg.IsSame(s1, s2), tg.IsSame(c1, c2))

    task = tg.Task(and1)
    self.assertEqual(task.operator_size, 9)

  def testSetChild(self):
    objs1 = tg.Select(when='last1')
    objs2 = tg.Select(when='now')
    s1 = tg.GetShape(objs1)
    s2 = tg.GetShape(objs2)
    c1 = tg.GetColor(objs1)
    c2 = tg.GetColor(objs2)
    and1 = tg.And(tg.IsSame(s1, s2), tg.IsSame(c1, c2))

    self.assertEqual(len(and1.child), 2)

  def testOperatorSize(self):
    objs1 = tg.Select(shape=sg.Shape('circle'), when='now')
    color1 = tg.GetColor(objs1)
    objs2 = tg.Select(color=color1, shape=sg.Shape('square'), when='now')
    exist = tg.Exist(objs2)
    task = tg.Task(exist)
    self.assertEqual(task.operator_size, 4)

    objs1 = tg.Select(when='last1')
    objs2 = tg.Select(when='now')
    s1 = tg.GetShape(objs1)
    s2 = tg.GetShape(objs2)
    c1 = tg.GetColor(objs1)
    c2 = tg.GetColor(objs2)
    bool1 = tg.And(tg.IsSame(s1, s2), tg.IsSame(c1, c2))
    task = tg.Task(bool1)
    self.assertEqual(task.operator_size, 9)

  def testCompareChild(self):
    objs1 = tg.Select(color=sg.Color('red'), when='now')
    bool1 = tg.Exist(objs1)
    go1 = tg.Go(objs1)

    self.assertEqual(bool1.child[0], go1.child[0])

  def testGetShapeOf(self):
    objs1 = tg.Select(color=sg.Color('blue'), when='last1')
    shape = tg.GetShape(objs1)
    objs2 = tg.Select(shape=shape, color=sg.Color('red'), when='now')
    task = tg.Task(tg.Exist(objs2))

    n_epoch = 5
    objset = sg.ObjectSet(n_epoch=n_epoch)
    for i_epoch in range(n_epoch):
      objset = task.guess_objset(objset, i_epoch)

  def testGetSpaceCall(self):
    objs0 = tg.Select(color=sg.Color('red'), when='last1')
    objs1 = tg.Select(loc=tg.GetLoc(objs0), when='now', space_type='left')
    task1 = tg.Task(tg.Exist(objs1))
    objs2 = tg.Select(loc=tg.GetLoc(objs0), when='now', space_type='right')
    task2 = tg.Task(tg.Exist(objs2))
    objs3 = tg.Select(loc=tg.GetLoc(objs0), when='now', space_type='top')
    task3 = tg.Task(tg.Exist(objs3))
    objs4 = tg.Select(loc=tg.GetLoc(objs0), when='now', space_type='bottom')
    task4 = tg.Task(tg.Exist(objs4))

    objset = sg.ObjectSet(n_epoch=2)
    obj1 = sg.Object([sg.Loc([0.5, 0.5]), sg.Color('red')], when='now')
    objset.add(obj1, epoch_now=0)
    obj1 = sg.Object([sg.Loc([0.2, 0.3])], when='now')
    objset.add(obj1, epoch_now=1)

    self.assertTrue(task1(objset, epoch_now=1))
    self.assertFalse(task2(objset, epoch_now=1))
    self.assertTrue(task3(objset, epoch_now=1))
    self.assertFalse(task4(objset, epoch_now=1))

  def testGetSpaceGuessObjset(self):
    objs1 = tg.Select(color=sg.Color('red'), when='last1')
    loc = tg.GetLoc(objs1)
    objs2 = tg.Select(
      loc=loc, color=sg.Color('blue'),when='now', space_type='left')
    task = tg.Task(tg.GetShape(objs2))

    n_epoch = 2
    objset = sg.ObjectSet(n_epoch=n_epoch)
    for i_epoch in range(n_epoch)[::-1]:
      objset = task.guess_objset(objset, i_epoch)
    o1 = objset.select(0, color=sg.Color('red'), when='now')[0]
    o2 = objset.select(1, color=sg.Color('blue'), when='now')[0]
    self.assertLess(o2.loc.value[0], o1.loc.value[0])

  def testExistColorOfGuessObjset(self):
    objs1 = tg.Select(shape=sg.Shape('circle'), when='last1')
    color = tg.GetColor(objs1)
    objs2 = tg.Select(color=color, shape=sg.Shape('square'), when='now')
    task = tg.Task(tg.Exist(objs2))

    n_epoch = 10
    objset = sg.ObjectSet(n_epoch=n_epoch)
    objset = task.guess_objset(objset, 1, should_be=True)
    objset = task.guess_objset(objset, 3, should_be=False)
    self.assertTrue(task(objset, 1))
    self.assertFalse(task(objset, 3))

  def testExistColorSpaceGuessObjset(self):
    for space_type in ['left', 'right', 'top', 'bottom']:
      objs1 = tg.Select(color=sg.Color('red'), when='now')
      loc = tg.GetLoc(objs1)
      objs2 = tg.Select(
        loc=loc, color=sg.Color('blue'), when='now', space_type=space_type)
      task = tg.Task(tg.Exist(objs2))

      n_epoch = 100
      objset = sg.ObjectSet(n_epoch=n_epoch)
      for i in range(0, n_epoch)[::-1]:
        should_be = random.random() > 0.5
        objset = task.guess_objset(objset, i, should_be=should_be)
        self.assertEqual(task(objset, i), should_be)

  def testExistColorSpaceGuessObjsetManyEpochs(self):
    objs1 = tg.Select(color=sg.Color('red'), when='last1')
    loc = tg.GetLoc(objs1)
    objs2 = tg.Select(
      loc=loc, color=sg.Color('blue'), when='now', space_type='left')
    task = tg.Task(tg.Exist(objs2))

    n_epoch = 10
    objset = sg.ObjectSet(n_epoch=n_epoch)
    objset = task.guess_objset(objset, 2, should_be=False)
    objset = task.guess_objset(objset, 1, should_be=True)
    self.assertFalse(task(objset, 2))
    self.assertTrue(task(objset, 1))

  def testExistColorOfGuessObjsetManyEpochs(self):
    objs1 = tg.Select(shape=sg.Shape('circle'), when='last1')
    color = tg.GetColor(objs1)
    objs2 = tg.Select(color=color, shape=sg.Shape('square'), when='now')
    task = tg.Task(tg.Exist(objs2))

    n_epoch = 100
    objset = sg.ObjectSet(n_epoch=n_epoch)
    for i in range(1, n_epoch)[::-1]:
      should_be = random.random() > 0.5
      objset = task.guess_objset(objset, i, should_be=should_be)
      self.assertEqual(task(objset, i), should_be)

  def testIsSameGuessObjsetManyEpochs(self):
    objs1 = tg.Select(shape=sg.Shape('square'), when='last1')
    objs2 = tg.Select(shape=sg.Shape('circle'), when='now')
    attr1 = tg.GetColor(objs1)
    attr2 = tg.GetColor(objs2)
    task = tg.Task(tg.IsSame(attr1, attr2))

    n_epoch = 100
    objset = sg.ObjectSet(n_epoch=n_epoch)
    for i in range(1, n_epoch)[::-1]:
      should_be = random.random() > 0.5
      objset = task.guess_objset(objset, i, should_be=should_be)
      self.assertEqual(task(objset, i), should_be)

  def testIsSameGuessObjsetWithDistractors(self):
    objs1 = tg.Select(shape=sg.Shape('square'), when='last1')
    objs2 = tg.Select(shape=sg.Shape('circle'), when='last1')
    attr1 = tg.GetColor(objs1)
    attr2 = tg.GetColor(objs2)
    task = tg.Task(tg.IsSame(attr1, attr2))

    n_epoch = 10
    objset = sg.ObjectSet(n_epoch=n_epoch)
    obj1 = sg.Object([sg.Color('green'), sg.Shape('square')], when='now',
                     deletable=True)
    objset.add(obj1, 0, add_if_exist=True)
    obj1 = sg.Object([sg.Color('red'), sg.Shape('circle')], when='now',
                     deletable=True)
    objset.add(obj1, 0, add_if_exist=True)
    objset = task.guess_objset(objset, 0, should_be=True)
    objset.add_distractor(1)
    objset = task.guess_objset(objset, 1, should_be=True)
    self.assertTrue(task(objset, 1))

  def testGenerateObjset(self):
    objs1 = tg.Select(shape=sg.Shape('square'), when='last1')
    objs2 = tg.Select(shape=sg.Shape('circle'), when='last1')
    attr1 = tg.GetColor(objs1)
    attr2 = tg.GetColor(objs2)
    task = tg.Task(tg.IsSame(attr1, attr2))

    task.generate_objset(n_epoch=20, n_distractor=3, average_memory_span=3)

  def manualAverageMemorySpan(self):
    n_epoch = 6
    average_memory_span = 1.67

    memory_spans = list()
    for _ in range(3000):
      objs1 = tg.Select(color=sg.Color('red'), when='last1')
      task = tg.Task(tg.GetShape(objs1))
      time1 = tg.GetTime(objs1)

      objset = task.generate_objset(
        n_epoch, n_distractor=1, average_memory_span=average_memory_span)
      for epoch_now in range(1, n_epoch):
        time1_val = time1(objset, epoch_now)
        memory_span = epoch_now - time1_val
        memory_spans.append(memory_span)

    avg_mem_span_estimation = sum(memory_spans)*1.0/len(memory_spans)
    print("avg_mem_span_estimation:", avg_mem_span_estimation)

  def manualFrames(self):
    n_epoch = 8
    average_memory_span = 1.01

    c_inv = []
    for _ in range(100):
      for name, task_class in tb.task_family_dict.items():
        task = task_class()
        objset = task.generate_objset(
            n_epoch, n_distractor=10, average_memory_span=average_memory_span)

        targets = task.get_target(objset)
        s = targets_to_str(targets)
        c_inv.append(len([x for x in s if x == 'invalid']))
    print(np.mean(c_inv))

  def manualLastOp(self):
    d = defaultdict(list)
    for name, task_class in tb.task_family_dict.items():
      d[task_class()._operator.__class__.__name__].append(name)

    for op, tasks in d.items():
      print('  d[\'%s\'] = %s' % (op, tasks))

  def manualOpCount(self):
    d = {}
    for name, task_class in tb.task_family_dict.items():
      d[name] = task_class().operator_size

    print(sorted(d.items(), key=lambda x: x[1]))

    t24 = []
    t58 = []
    t911 = []
    for name, count in d.items():
      if count <= 4:
        t24.append(name)
      elif count <= 8:
        t58.append(name)
      else:
        t911.append(name)

    print(t24)
    print(t58)
    print(t911)

  def testAverageMemorySpan(self):
    n_epoch = 1000
    average_memory_span = 10

    objs1 = tg.Select(color=sg.Color('red'), when='last1')
    task = tg.Task(tg.GetShape(objs1))
    time1 = tg.GetTime(objs1)

    objset = task.generate_objset(
      n_epoch, n_distractor=1, average_memory_span=average_memory_span)
    memory_spans = list()
    for epoch_now in range(1, n_epoch):
      time1_val = time1(objset, epoch_now)
      memory_span = epoch_now - time1_val
      memory_spans.append(memory_span)

    avg_mem_span_estimation = sum(memory_spans)*1.0/len(memory_spans)
    self.assertLess(abs(avg_mem_span_estimation-average_memory_span), 2)


if __name__ == '__main__':
  unittest.main()
