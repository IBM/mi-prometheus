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

"""A bank of available tasks."""

from collections import OrderedDict
import random

from miprometheus.problems.seq_to_seq.video_text_to_class.cog.cog_utils import stim_generator as sg
from miprometheus.problems.seq_to_seq.video_text_to_class.cog.cog_utils import task_generator as tg
from miprometheus.problems.seq_to_seq.video_text_to_class.cog.cog_utils.task_generator import Task


class GoColor(Task):
  """Go to color X."""

  def __init__(self):
    color1 = sg.random_color()
    when1 = sg.random_when()
    objs1 = tg.Select(color=color1, when=when1)
    self._operator = tg.Go(objs1)

  @property
  def instance_size(self):
    return sg.n_random_color() * sg.n_random_when()


class GoShape(Task):
  """Go to shape X."""

  def __init__(self):
    shape1 = sg.random_shape()
    when1 = sg.random_when()
    objs1 = tg.Select(shape=shape1, when=when1)
    self._operator = tg.Go(objs1)

  @property
  def instance_size(self):
    return sg.n_random_shape() * sg.n_random_when()


class Go(Task):
  """Go to object X."""

  def __init__(self):
    color1 = sg.random_color()
    shape1 = sg.random_shape()
    when1 = sg.random_when()
    objs1 = tg.Select(color=color1, shape=shape1, when=when1)
    self._operator = tg.Go(objs1)

  @property
  def instance_size(self):
    return sg.n_random_color() * sg.n_random_shape() * sg.n_random_when()


class GoColorOf(Task):
  """Go to shape 1 with the same color as shape 2.

  In general, this task can be extremely difficult, requiring memory of
  locations of all latest shape 2 of different colors.

  To make this task reasonable, we use customized generate_objset.

  Returns:
    task: task
  """

  def __init__(self):
    shape1, shape2, shape3 = sg.sample_shape(3)
    objs1 = tg.Select(shape=shape1, when='latest')
    color1 = tg.GetColor(objs1)
    objs2 = tg.Select(color=color1, shape=shape2, when='now')
    self._operator = tg.Go(objs2)

    self._shape1, self._shape2, self._shape3 = shape1, shape2, shape3

  def generate_objset(self, n_epoch, n_distractor=1, average_memory_span=2):
    """Generate object set.

    The task has 4 epochs: Fixation, Sample, Delay, and Test.
    During sample, one sample object is shown.
    During test, two test objects are shown, one of them will match the color
    of the sample object

    Args:
      n_epoch: int

    Returns:
      objset: ObjectSet instance.

    Raises:
      ValueError: when n_epoch is less than 4,
          the minimum epoch number for this task
    """
    if n_epoch < 4:
      raise ValueError('Number of epoch {:d} is less than 4'.format(n_epoch))
    color1, color2 = sg.sample_color(2)
    color3 = sg.random_color()

    objset = sg.ObjectSet(n_epoch=n_epoch)

    sample1 = sg.Object([color1, self._shape1], when='now')
    distractor1 = sg.Object([color3, self._shape3], when='now')
    test1 = sg.Object([color1, self._shape2], when='now')
    test2 = sg.Object([color2, self._shape2], when='now')

    objset.add(sample1, epoch_now=1)  # sample epoch
    objset.add(distractor1, epoch_now=2)  # delay epoch
    objset.add(test1, epoch_now=3)  # test epoch
    objset.add(test2, epoch_now=3)  # test epoch
    return objset

  @property
  def instance_size(self):
    return sg.n_sample_shape(3)


class GoShapeOf(Task):
  """Go to color 1 with the same shape as color 2."""

  def __init__(self):
    color1, color2, color3 = sg.sample_color(3)
    objs1 = tg.Select(color=color1, when='latest')
    shape1 = tg.GetShape(objs1)
    objs2 = tg.Select(color=color2, shape=shape1, when='now')
    self._operator = tg.Go(objs2)
    self._color1, self._color2, self._color3 = color1, color2, color3

  def generate_objset(self, n_epoch, n_distractor=1, average_memory_span=2):
    """Generate object set."""
    if n_epoch < 4:
      raise ValueError('Number of epoch {:d} is less than 4'.format(n_epoch))
    shape1, shape2 = sg.sample_shape(2)
    shape3 = sg.random_shape()

    objset = sg.ObjectSet(n_epoch=n_epoch)

    sample1 = sg.Object([self._color1, shape1], when='now')
    distractor1 = sg.Object([self._color3, shape3], when='now')
    test1 = sg.Object([self._color2, shape1], when='now')
    test2 = sg.Object([self._color2, shape2], when='now')

    objset.add(sample1, epoch_now=1)  # sample epoch
    objset.add(distractor1, epoch_now=2)  # delay epoch
    objset.add(test1, epoch_now=3)  # test epoch
    objset.add(test2, epoch_now=3)  # test epoch
    return objset

  @property
  def instance_size(self):
    return sg.n_sample_color(3)


class GetColor(Task):

  def __init__(self):
    when = sg.random_when()
    shape = sg.random_shape()
    objs1 = tg.Select(shape=shape, when=when)
    self._operator = tg.GetColor(objs1)

  @property
  def instance_size(self):
    return sg.n_random_shape() * sg.n_random_when()


class GetShape(Task):

  def __init__(self):
    when = sg.random_when()
    color = sg.random_color()
    objs1 = tg.Select(color=color, when=when)
    self._operator = tg.GetShape(objs1)

  @property
  def instance_size(self):
    return sg.n_random_color() * sg.n_random_when()


class GetColorSpace(Task):
  """The get color space task."""
  # TODO(gryang): The problem with this task is that there is a short-cut
  # solution. For example, for question: color of now triangle on left of now
  # vbar, the network can achieve perfect accuracy by answering the question:
  # color of now triangle on left

  def __init__(self):
    shape1, shape2 = sg.sample_shape(2)
    objs1 = tg.Select(shape=shape1, when=sg.random_when())
    loc = tg.GetLoc(objs1)
    objs2 = tg.Select(loc=loc, shape=shape2, when=sg.random_when(),
                      space_type=sg.random_space())
    self._operator = tg.GetColor(objs2)

  @property
  def instance_size(self):
    return sg.n_sample_shape(2) * sg.n_random_space() * sg.n_random_when()**2


class GetShapeSpace(Task):
  """The get shape space task."""

  def __init__(self):
    color1, color2 = sg.sample_color(2)
    objs1 = tg.Select(color=color1, when=sg.random_when())
    loc = tg.GetLoc(objs1)
    objs2 = tg.Select(loc=loc, color=color2, when=sg.random_when(),
                      space_type=sg.random_space())
    self._operator = tg.GetShape(objs2)

  @property
  def instance_size(self):
    return sg.n_sample_color(2) * sg.n_random_space() * sg.n_random_when()**2


class ExistColor(Task):
  """The exist color task family."""

  def __init__(self):
    objs1 = tg.Select(color=sg.random_color(), when='now')
    self._operator = tg.Exist(objs1)

  @property
  def instance_size(self):
    return sg.n_random_color()


class ExistShape(Task):
  """The exist shape task family."""

  def __init__(self):
    objs1 = tg.Select(shape=sg.random_shape(), when='now')
    self._operator = tg.Exist(objs1)

  @property
  def instance_size(self):
    return sg.n_random_shape()


class Exist(Task):
  """The exist task family."""

  def __init__(self):
    attr1 = sg.random_colorshape()
    objs1 = tg.Select(color=attr1[0], shape=attr1[1], when='now')
    task = tg.Exist(objs1)
    self._operator = task

  @property
  def instance_size(self):
    return sg.n_random_color() * sg.n_random_shape()


class ExistColorSpace(Task):
  """The exist color space task family."""

  def __init__(self):
    color1, color2 = sg.sample_color(2)
    objs1 = tg.Select(color=color1, when=sg.random_when())
    loc = tg.GetLoc(objs1)
    objs2 = tg.Select(loc=loc, color=color2, when='now',
                      space_type=sg.random_space())
    self._operator = tg.Exist(objs2)

  @property
  def instance_size(self):
    return sg.n_sample_color(2) * sg.n_random_space() * sg.n_random_when()


class ExistShapeSpace(Task):
  """The exist shape space task family."""

  def __init__(self):
    shape1, shape2 = sg.sample_shape(2)
    objs1 = tg.Select(shape=shape1, when=sg.random_when())
    loc = tg.GetLoc(objs1)
    objs2 = tg.Select(loc=loc, shape=shape2, when='now',
                      space_type=sg.random_space())
    self._operator = tg.Exist(objs2)

  @property
  def instance_size(self):
    return sg.n_sample_shape(2) * sg.n_random_space() * sg.n_random_when()


class ExistSpace(Task):
  """The exist space task family."""

  def __init__(self):
    attr1, attr2 = sg.sample_colorshape(2)
    objs1 = tg.Select(color=attr1[0], shape=attr1[1], when=sg.random_when())
    loc = tg.GetLoc(objs1)
    objs2 = tg.Select(loc=loc, color=attr2[0], shape=attr2[1], when='now',
                      space_type=sg.random_space())
    self._operator = tg.Exist(objs2)

  @property
  def instance_size(self):
    return sg.n_sample_colorshape(2) * sg.n_random_space() * sg.n_random_when()


class ExistColorSpaceGo(Task):
  """The exist color space go task family."""

  def __init__(self):
    color1, color2, color3, color4 = sg.sample_color(4)
    objs1 = tg.Select(color=color1, when=sg.random_when())
    loc = tg.GetLoc(objs1)
    objs2 = tg.Select(loc=loc, color=color2, when='now',
                      space_type=sg.random_space())
    bool1 = tg.Exist(objs2)

    objs3 = tg.Select(color=color3)
    objs4 = tg.Select(color=color4)
    go1 = tg.Go(objs3)
    go2 = tg.Go(objs4)
    self._operator = tg.Switch(bool1, go1, go2, both_options_avail=True)

  @property
  def instance_size(self):
    return sg.n_sample_color(4) * sg.n_random_space() * sg.n_random_when()


class ExistShapeSpaceGo(Task):
  """The exist shape space go task family."""

  def __init__(self):
    shape1, shape2, shape3, shape4 = sg.sample_shape(4)
    objs1 = tg.Select(shape=shape1, when=sg.random_when())
    loc = tg.GetLoc(objs1)
    objs2 = tg.Select(loc=loc, shape=shape2, when='now',
                      space_type=sg.random_space())
    bool1 = tg.Exist(objs2)

    objs3 = tg.Select(shape=shape3)
    objs4 = tg.Select(shape=shape4)
    go1 = tg.Go(objs3)
    go2 = tg.Go(objs4)
    self._operator = tg.Switch(bool1, go1, go2, both_options_avail=True)

  @property
  def instance_size(self):
    return sg.n_sample_shape(4) * sg.n_random_space() * sg.n_random_when()


class ExistSpaceGo(Task):
  """The exist space go task family."""

  def __init__(self):
    attr1, attr2, attr3, attr4 = sg.sample_colorshape(4)
    objs1 = tg.Select(color=attr1[0], shape=attr1[1], when=sg.random_when())
    loc = tg.GetLoc(objs1)
    objs2 = tg.Select(loc=loc, color=attr2[0], shape=attr2[1], when='now',
                      space_type=sg.random_space())
    bool1 = tg.Exist(objs2)

    objs3 = tg.Select(color=attr3[0], shape=attr3[1])
    objs4 = tg.Select(color=attr4[0], shape=attr4[1])
    go1 = tg.Go(objs3)
    go2 = tg.Go(objs4)
    self._operator = tg.Switch(bool1, go1, go2, both_options_avail=True)

  @property
  def instance_size(self):
    return sg.n_sample_colorshape(4) * sg.n_random_space() * sg.n_random_when()


class SimpleExistColorGo(Task):
  """If exist color A then go color A, else go color B."""

  def __init__(self):
    color1, color2 = sg.sample_color(2)
    objs1 = tg.Select(color=color1, when='now')
    objs2 = tg.Select(color=color2, when=sg.random_when())
    self._operator = tg.Switch(
        tg.Exist(objs1), tg.Go(objs1), tg.Go(objs2), both_options_avail=False)

  @property
  def instance_size(self):
    return sg.n_sample_color(2) * sg.n_random_when()


class SimpleExistShapeGo(Task):
  """If exist shape A then go shape A, else go shape B."""

  def __init__(self):
    shape1, shape2 = sg.sample_shape(2)
    objs1 = tg.Select(shape=shape1, when='now')
    objs2 = tg.Select(shape=shape2, when=sg.random_when())
    self._operator = tg.Switch(
        tg.Exist(objs1), tg.Go(objs1), tg.Go(objs2), both_options_avail=False)

  @property
  def instance_size(self):
    return sg.n_sample_shape(2) * sg.n_random_when()


class SimpleExistGo(Task):
  """If exist A, then go A, else go B."""

  def __init__(self):
    attr1, attr2 = sg.sample_colorshape(2)
    when2 = sg.random_when()

    objs1 = tg.Select(color=attr1[0], shape=attr1[1], when='now')
    objs2 = tg.Select(color=attr2[0], shape=attr2[1], when=when2)
    self._operator = tg.Switch(
        tg.Exist(objs1), tg.Go(objs1), tg.Go(objs2), both_options_avail=False)

  @property
  def instance_size(self):
    return sg.n_sample_colorshape(2) * sg.n_random_when()


class AndSimpleExistColorGo(Task):
  """If exist color A and exist color B then go color A, else go color C."""

  def __init__(self):
    color1, color2, color3 = sg.sample_color(3)
    objs1 = tg.Select(color=color1, when='now')
    objs2 = tg.Select(color=color2, when='now')
    objs3 = tg.Select(color=color3, when=sg.random_when())
    bool1 = tg.And(tg.Exist(objs1), tg.Exist(objs2))
    self._operator = tg.Switch(
        bool1, tg.Go(objs1), tg.Go(objs3), both_options_avail=False)

  @property
  def instance_size(self):
    return sg.n_sample_color(3) * sg.n_random_when()


class AndSimpleExistShapeGo(Task):
  """If exist shape A and shape B then go shape A, else go shape C."""

  def __init__(self):
    shape1, shape2, shape3 = sg.sample_shape(3)
    objs1 = tg.Select(shape=shape1, when='now')
    objs2 = tg.Select(shape=shape2, when='now')
    objs3 = tg.Select(shape=shape3, when=sg.random_when())
    bool1 = tg.And(tg.Exist(objs1), tg.Exist(objs2))
    self._operator = tg.Switch(
        bool1, tg.Go(objs1), tg.Go(objs3), both_options_avail=False)

  @property
  def instance_size(self):
    return sg.n_sample_shape(3) * sg.n_random_when()


class AndSimpleExistGo(Task):
  """If exist A and B, then go A, else go C."""

  def __init__(self):
    attr1, attr2, attr3 = sg.sample_colorshape(3)

    objs1 = tg.Select(color=attr1[0], shape=attr1[1], when='now')
    objs2 = tg.Select(color=attr2[0], shape=attr2[1], when='now')
    objs3 = tg.Select(color=attr3[0], shape=attr3[1], when=sg.random_when())
    bool1 = tg.And(tg.Exist(objs1), tg.Exist(objs2))
    self._operator = tg.Switch(
        bool1, tg.Go(objs1), tg.Go(objs3), both_options_avail=False)

  @property
  def instance_size(self):
    return sg.n_sample_colorshape(3) * sg.n_random_when()


class ExistColorGo(Task):
  """Exist color go task."""

  def __init__(self):
    color1, color2, color3 = sg.sample_color(3)
    objs1 = tg.Select(color=color1, when='now')
    objs2 = tg.Select(color=color2, when=sg.random_when())
    objs3 = tg.Select(color=color3, when=sg.random_when())
    self._operator = tg.Switch(
        tg.Exist(objs1), tg.Go(objs2), tg.Go(objs3), both_options_avail=False)

  @property
  def instance_size(self):
    return sg.n_sample_color(3) * (sg.n_random_when())**2


class ExistShapeGo(Task):
  """Exist shape go task."""

  def __init__(self):
    shape1, shape2, shape3 = sg.sample_shape(3)
    objs1 = tg.Select(shape=shape1, when='now')
    objs2 = tg.Select(shape=shape2, when=sg.random_when())
    objs3 = tg.Select(shape=shape3, when=sg.random_when())
    self._operator = tg.Switch(
        tg.Exist(objs1), tg.Go(objs2), tg.Go(objs3), both_options_avail=False)

  @property
  def instance_size(self):
    return sg.n_sample_shape(3) * (sg.n_random_when())**2


class ExistGo(Task):
  """Exist go task."""

  def __init__(self):
    attr1, attr2, attr3 = sg.sample_colorshape(3)
    objs1 = tg.Select(color=attr1[0], shape=attr1[1], when='now')
    objs2 = tg.Select(color=attr2[0], shape=attr2[1], when=sg.random_when())
    objs3 = tg.Select(color=attr3[0], shape=attr3[1], when=sg.random_when())
    self._operator = tg.Switch(
        tg.Exist(objs1), tg.Go(objs2), tg.Go(objs3), both_options_avail=False)

  @property
  def instance_size(self):
    return sg.n_sample_colorshape(3) * (sg.n_random_when())**2


class SimpleCompareColor(Task):
  """Compare color."""

  def __init__(self):
    objs1 = tg.Select(shape=sg.random_shape(), when=sg.random_when())
    a2 = sg.random_color()
    a1 = tg.GetColor(objs1)
    if random.random() > 0.5:
      self._operator = tg.IsSame(a1, a2)
    else:
      self._operator = tg.IsSame(a2, a1)

  @property
  def instance_size(self):
    return sg.n_random_shape() * sg.n_random_when() * sg.n_random_color() * 2


class SimpleCompareShape(Task):
  """Compare shape."""

  def __init__(self):
    objs1 = tg.Select(color=sg.random_color(), when=sg.random_when())
    a2 = sg.random_shape()
    a1 = tg.GetShape(objs1)
    if random.random() > 0.5:
      self._operator = tg.IsSame(a1, a2)
    else:
      self._operator = tg.IsSame(a2, a1)

  @property
  def instance_size(self):
    return sg.n_random_color() * sg.n_random_when() * sg.n_random_shape() * 2


class AndSimpleCompareColor(Task):
  """Compare color and compare another color."""

  def __init__(self):
    shape1, shape2 = sg.sample_shape(2)
    objs1 = tg.Select(shape=shape1, when=sg.random_when())
    objs2 = tg.Select(shape=shape2, when=sg.random_when())
    color11 = tg.GetColor(objs1)
    color21 = tg.GetColor(objs2)
    color12, color22 = sg.random_color(), sg.random_color()
    if random.random() > 0.5:
      bool1 = tg.IsSame(color11, color12)
    else:
      bool1 = tg.IsSame(color11, color12)
    if random.random() > 0.5:
      bool2 = tg.IsSame(color21, color22)
    else:
      bool2 = tg.IsSame(color21, color22)

    self._operator = tg.And(bool1, bool2)

  @property
  def instance_size(self):
    return (sg.n_sample_shape(2) * sg.n_random_when()**2 *
            sg.n_random_color()**2 * 4)


class AndSimpleCompareShape(Task):
  """Compare shape and compare another shape."""

  def __init__(self):
    color1, color2 = sg.sample_color(2)
    objs1 = tg.Select(color=color1, when=sg.random_when())
    objs2 = tg.Select(color=color2, when=sg.random_when())
    shape11 = tg.GetShape(objs1)
    shape21 = tg.GetShape(objs2)
    shape12, shape22 = sg.random_shape(), sg.random_shape()
    if random.random() > 0.5:
      bool1 = tg.IsSame(shape11, shape12)
    else:
      bool1 = tg.IsSame(shape11, shape12)
    if random.random() > 0.5:
      bool2 = tg.IsSame(shape21, shape22)
    else:
      bool2 = tg.IsSame(shape21, shape22)

    self._operator = tg.And(bool1, bool2)

  @property
  def instance_size(self):
    return (sg.n_sample_color(2) * sg.n_random_when()**2 *
            sg.n_random_shape()**2 * 4)


class CompareColor(Task):
  """Compare color between two objects."""

  def __init__(self):
    shape1, shape2 = sg.sample_shape(2)
    objs1 = tg.Select(shape=shape1, when=sg.random_when())
    objs2 = tg.Select(shape=shape2, when=sg.random_when())
    a1 = tg.GetColor(objs1)
    a2 = tg.GetColor(objs2)
    self._operator = tg.IsSame(a1, a2)

  @property
  def instance_size(self):
    return sg.n_sample_shape(2) * (sg.n_random_when())**2


class CompareShape(Task):
  """Compare shape between two objects."""

  def __init__(self):
    color1, color2 = sg.sample_color(2)
    objs1 = tg.Select(color=color1, when=sg.random_when())
    objs2 = tg.Select(color=color2, when=sg.random_when())
    a1 = tg.GetShape(objs1)
    a2 = tg.GetShape(objs2)
    self._operator = tg.IsSame(a1, a2)

  @property
  def instance_size(self):
    return sg.n_sample_color(2) * (sg.n_random_when())**2


class AndCompareColor(Task):
  """Compare color between two objects and compare another pair."""

  def __init__(self):
    shape1, shape2, shape3, shape4 = sg.sample_shape(4)
    objs1 = tg.Select(shape=shape1, when=sg.random_when())
    objs2 = tg.Select(shape=shape2, when=sg.random_when())
    objs3 = tg.Select(shape=shape3, when=sg.random_when())
    objs4 = tg.Select(shape=shape4, when=sg.random_when())
    a1 = tg.GetColor(objs1)
    a2 = tg.GetColor(objs2)
    a3 = tg.GetColor(objs3)
    a4 = tg.GetColor(objs4)
    self._operator = tg.And(tg.IsSame(a1, a2), tg.IsSame(a3, a4))

  @property
  def instance_size(self):
    return sg.n_sample_shape(4) * (sg.n_random_when())**4


class AndCompareShape(Task):
  """Compare shape between two objects and compare another pair."""

  def __init__(self):
    color1, color2, color3, color4 = sg.sample_color(4)
    objs1 = tg.Select(color=color1, when=sg.random_when())
    objs2 = tg.Select(color=color2, when=sg.random_when())
    objs3 = tg.Select(color=color3, when=sg.random_when())
    objs4 = tg.Select(color=color4, when=sg.random_when())
    a1 = tg.GetShape(objs1)
    a2 = tg.GetShape(objs2)
    a3 = tg.GetShape(objs3)
    a4 = tg.GetShape(objs4)
    self._operator = tg.And(tg.IsSame(a1, a2), tg.IsSame(a3, a4))

  @property
  def instance_size(self):
    return sg.n_sample_color(4) * (sg.n_random_when())**4


class ExistColorOf(Task):
  """Check if exist object with color of a shape."""

  def __init__(self):
    shape1, shape2 = sg.sample_shape(2)
    objs1 = tg.Select(shape=shape1, when=sg.random_when())
    color1 = tg.GetColor(objs1)
    objs2 = tg.Select(color=color1, shape=shape2, when='now')
    self._operator = tg.Exist(objs2)

  @property
  def instance_size(self):
    return sg.n_sample_shape(2) * sg.n_random_when()


class ExistShapeOf(Task):
  """Check if exist object with shape of a colored object."""

  def __init__(self):
    color1, color2 = sg.sample_color(2)
    objs1 = tg.Select(color=color1, when=sg.random_when())
    shape1 = tg.GetShape(objs1)
    objs2 = tg.Select(color=color2, shape=shape1, when='now')
    self._operator = tg.Exist(objs2)

  @property
  def instance_size(self):
    return sg.n_sample_color(2) * sg.n_random_when()


class ExistLastShapeSameColor(Task):
  """Check if exist last shape with color same as current shape."""

  def __init__(self):
    self._shape1 = sg.random_shape()
    objs1 = tg.Select(shape=self._shape1, when='now')
    color1 = tg.GetColor(objs1)
    objs2 = tg.Select(color=color1, shape=self._shape1, when='last1')
    self._operator = tg.Exist(objs2)

  def generate_objset(self, n_epoch, n_distractor=1, average_memory_span=2):
    if n_epoch < 4:
      raise ValueError('Number of epoch {:d} is less than 4'.format(n_epoch))
    objset = sg.ObjectSet(n_epoch=n_epoch)

    n_sample = 2
    sample_colors = sg.sample_color(n_sample)
    for i in range(n_sample):
      obj = sg.Object([sample_colors[i], self._shape1], when='now')
      objset.add(obj, epoch_now=1)

    obj = sg.Object([sg.another_shape(self._shape1)], when='now')
    objset.add(obj, epoch_now=1)  # distractor

    if random.random() < 0.5:
      color3 = random.choice(sample_colors)
    else:
      color3 = sg.another_color(sample_colors)
    test1 = sg.Object([color3, self._shape1], when='now')
    objset.add(test1, epoch_now=3)  # test epoch
    return objset

  @property
  def instance_size(self):
    return sg.n_random_shape()


class ExistLastColorSameShape(Task):
  """Check if exist last color with shape same as current color object."""

  def __init__(self):
    self._color1 = sg.random_color()
    objs1 = tg.Select(color=self._color1, when='now')
    shape1 = tg.GetShape(objs1)
    objs2 = tg.Select(color=self._color1, shape=shape1, when='last1')
    self._operator = tg.Exist(objs2)

  def generate_objset(self, n_epoch, n_distractor=1, average_memory_span=2):
    if n_epoch < 4:
      raise ValueError('Number of epoch {:d} is less than 4'.format(n_epoch))
    objset = sg.ObjectSet(n_epoch=n_epoch)

    n_sample = 2
    sample_shapes = sg.sample_shape(n_sample)
    for i in range(n_sample):
      obj = sg.Object([self._color1, sample_shapes[i]], when='now')
      objset.add(obj, epoch_now=1)

    obj = sg.Object([sg.another_color(self._color1)], when='now')
    objset.add(obj, epoch_now=1)  # distractor

    if random.random() < 0.5:
      shape3 = random.choice(sample_shapes)
    else:
      shape3 = sg.another_shape(sample_shapes)
    test1 = sg.Object([self._color1, shape3], when='now')
    objset.add(test1, epoch_now=3)  # test epoch
    return objset

  @property
  def instance_size(self):
    return sg.n_random_color()

# TODO(gryang): Consider getting rid of this task
class ExistLastObjectSameObject(Task):
  """Check if exist last color with shape same as current color object."""

  def __init__(self):
    objs1 = tg.Select(when='now')
    color1 = tg.GetColor(objs1)
    shape1 = tg.GetShape(objs1)
    objs2 = tg.Select(color=color1, shape=shape1, when='last1')
    self._operator = tg.Exist(objs2)

  def generate_objset(self, n_epoch, n_distractor=1, average_memory_span=2):
    """Manual generate objset.

    By design this function will not be balanced because the network always
    answer False during the sample epoch.
    """
    if n_epoch < 4:
      raise ValueError('Number of epoch {:d} is less than 4'.format(n_epoch))
    objset = sg.ObjectSet(n_epoch=n_epoch)

    n_sample = random.choice([1, 2, 3, 4])
    sample_attrs = sg.sample_colorshape(n_sample + 1)
    for attrs in sample_attrs[:n_sample]:
      obj = sg.Object(attrs, when='now')
      objset.add(obj, epoch_now=1)

    if random.random() < 0.5:
      attr = random.choice(sample_attrs[:n_sample])
    else:
      attr = sample_attrs[-1]
    test1 = sg.Object(attr, when='now')
    objset.add(test1, epoch_now=3)  # test epoch
    return objset

  @property
  def instance_size(self):
    return 1


class SimpleCompareColorGo(Task):
  """Simple compare color go."""

  def __init__(self):
    shape1, shape2 = sg.sample_shape(2)
    when1 = sg.random_when()
    a2 = sg.random_color()
    color1, color2 = sg.sample_color(2)

    objs1 = tg.Select(shape=shape1, when=when1)
    a1 = tg.GetColor(objs1)

    if random.random() > 0.5:
      bool1 = tg.IsSame(a1, a2)
    else:
      bool1 = tg.IsSame(a2, a1)

    objs2 = tg.Select(color=color1, shape=shape2)
    objs3 = tg.Select(color=color2, shape=shape2)
    go1 = tg.Go(objs2)
    go2 = tg.Go(objs3)
    self._operator = tg.Switch(bool1, go1, go2, both_options_avail=True)

  @property
  def instance_size(self):
    return (sg.n_sample_shape(2)*sg.n_random_color()*
            sg.n_sample_color(2)*sg.n_random_when())*2


class SimpleCompareShapeGo(Task):
  """Simple compare shape go."""

  def __init__(self):
    color1, color2 = sg.sample_color(2)
    when1 = sg.random_when()
    a2 = sg.random_shape()
    shape1, shape2 = sg.sample_shape(2)

    objs1 = tg.Select(color=color1, when=when1)
    a1 = tg.GetShape(objs1)

    if random.random() > 0.5:
      bool1 = tg.IsSame(a1, a2)
    else:
      bool1 = tg.IsSame(a2, a1)

    objs2 = tg.Select(color=color2, shape=shape1)
    objs3 = tg.Select(color=color2, shape=shape2)
    go1 = tg.Go(objs2)
    go2 = tg.Go(objs3)
    self._operator = tg.Switch(bool1, go1, go2, both_options_avail=True)

  @property
  def instance_size(self):
    return (sg.n_sample_color(2)*sg.n_random_shape()*
            sg.n_sample_shape(2)*sg.n_random_when())*2


class CompareColorGo(Task):
  """Compare color go.

  Compare color of shape X and Y, if true go to color 1 shape Z,
  else color 2 shape Z. The two shape Z are shown throughout the trial.

  Returns:
    task: Operator instance.
  """

  def __init__(self):
    shape1, shape2, shape3 = sg.sample_shape(3)
    color1, color2 = sg.sample_color(2)
    when1 = sg.random_when()
    when2 = sg.random_when()

    objs1 = tg.Select(shape=shape1, when=when1)
    objs2 = tg.Select(shape=shape2, when=when2)
    a1 = tg.GetColor(objs1)
    a2 = tg.GetColor(objs2)
    bool1 = tg.IsSame(a1, a2)

    objs3 = tg.Select(color=color1, shape=shape3)
    objs4 = tg.Select(color=color2, shape=shape3)
    go1 = tg.Go(objs3)
    go2 = tg.Go(objs4)
    self._operator = tg.Switch(bool1, go1, go2, both_options_avail=True)

  @property
  def instance_size(self):
    return sg.n_sample_shape(3)*sg.n_sample_color(2)*(sg.n_random_when())**2


class CompareShapeGo(Task):
  """Compare shape go."""

  def __init__(self):
    color1, color2, color3 = sg.sample_color(3)
    shape3, shape4 = sg.sample_shape(2)
    when1 = sg.random_when()
    when2 = sg.random_when()

    objs1 = tg.Select(color=color1, when=when1)
    objs2 = tg.Select(color=color2, when=when2)
    s1 = tg.GetShape(objs1)
    s2 = tg.GetShape(objs2)
    bool1 = tg.IsSame(s1, s2)
    objs3 = tg.Select(color=color3, shape=shape3)
    objs4 = tg.Select(color=color3, shape=shape4)
    go1 = tg.Go(objs3)
    go2 = tg.Go(objs4)
    self._operator = tg.Switch(bool1, go1, go2, both_options_avail=True)

  @property
  def instance_size(self):
    return sg.n_sample_color(3)*sg.n_sample_shape(2)*(sg.n_random_when())**2


task_family_dict = OrderedDict([
    ('GoColor', GoColor),
    ('GoShape', GoShape),
    ('Go', Go),
    ('GoColorOf', GoColorOf),
    ('GoShapeOf', GoShapeOf),
    ('GetColor', GetColor),
    ('GetShape', GetShape),
    ('GetColorSpace', GetColorSpace),
    ('GetShapeSpace', GetShapeSpace),
    ('ExistColor', ExistColor),
    ('ExistShape', ExistShape),
    ('Exist', Exist),
    ('ExistColorSpace', ExistColorSpace),
    ('ExistShapeSpace', ExistShapeSpace),
    ('ExistSpace', ExistSpace),
    ('ExistColorSpaceGo', ExistColorSpaceGo),
    ('ExistShapeSpaceGo', ExistShapeSpaceGo),
    ('ExistSpaceGo', ExistSpaceGo),
    ('ExistShapeOf', ExistShapeOf),
    ('ExistColorOf', ExistColorOf),
    ('ExistLastShapeSameColor', ExistLastShapeSameColor),
    ('ExistLastColorSameShape', ExistLastColorSameShape),
    ('ExistLastObjectSameObject', ExistLastObjectSameObject),
    ('SimpleExistColorGo', SimpleExistColorGo),
    ('SimpleExistShapeGo', SimpleExistShapeGo),
    ('SimpleExistGo', SimpleExistGo),
    ('AndSimpleExistColorGo', AndSimpleExistColorGo),
    ('AndSimpleExistShapeGo', AndSimpleExistShapeGo),
    ('AndSimpleExistGo', SimpleExistGo),
    ('ExistColorGo', ExistColorGo),
    ('ExistShapeGo', ExistShapeGo),
    ('ExistGo', ExistGo),
    ('SimpleCompareColor', SimpleCompareColor),
    ('SimpleCompareShape', SimpleCompareShape),
    ('AndSimpleCompareColor', AndSimpleCompareColor),
    ('AndSimpleCompareShape', AndSimpleCompareShape),
    ('CompareColor', CompareColor),
    ('CompareShape', CompareShape),
    ('AndCompareColor', AndCompareColor),
    ('AndCompareShape', AndCompareShape),
    ('SimpleCompareColorGo', SimpleCompareColorGo),
    ('SimpleCompareShapeGo', SimpleCompareShapeGo),
    ('CompareColorGo', CompareColorGo),
    ('CompareShapeGo', CompareShapeGo),
])


def random_task(task_family):
  """Return a random question from the task family."""
  return task_family_dict[task_family]()
