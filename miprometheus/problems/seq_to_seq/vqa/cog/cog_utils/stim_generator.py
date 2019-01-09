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

"""High-level API for generating stimuli.

Objects are first generated abstractly, with high-level specifications
like loc='random'.
Abstract relationships between objects can also be specified.

All objects and relationships are then collected into a ObjectSet.
The ObjectSet object can interpret the abstract specifications and instantiate
the stimuli in each trial.

Rendering function generates movies based on the instantiated stimuli
"""

import itertools
from bisect import bisect_left
from collections import defaultdict
import random
import numpy as np
import string
from PIL import Image, ImageDraw, ImageFont
import os

#import cv2 as cv2

from miprometheus.problems.seq_to_seq.vqa.cog.cog_utils import constants as const


class Attribute(object):
  """Base class for attributes."""

  def __init__(self, value):
    self.value = value if not isinstance(value, list) else tuple(value)
    self.parent = list()

  def __call__(self, *args):
    """Including a call function to be consistent with Operator class."""
    return self

  def __str__(self):
    return str(self.value)

  def __eq__(self, other):
    """Override the default Equals behavior."""
    if isinstance(other, self.__class__):
      return self.value == other.value
    return False

  def __ne__(self, other):
    """Define a non-equality test."""
    if isinstance(other, self.__class__):
      return not self.__eq__(other)
    return True

  def __hash__(self):
    """Override the default hash behavior."""
    return hash(tuple(sorted(self.__dict__.items())))

  def resample(self):
    raise NotImplementedError('Abstract method.')

  @property
  def has_value(self):
    return self.value is not None


class Shape(Attribute):
  """Shape class."""

  def __init__(self, value):
    super(Shape, self).__init__(value)
    self.attr_type = 'shape'

  def sample(self):
    self.value = random_shape().value

  def resample(self):
    self.value = another_shape(self).value


class Color(Attribute):
  """Color class."""

  def __init__(self, value):
    super(Color, self).__init__(value)
    self.attr_type = 'color'

  def sample(self):
    self.value = random_color().value

  def resample(self):
    self.value = another_color(self).value


def _get_space_to(x0, x1, y0, y1, space_type):
  if space_type == 'right':
    space = [(x1, 0.95), (0.05, 0.95)]
  elif space_type == 'left':
    space = [(0.05, x0), (0.05, 0.95)]
  elif space_type == 'top':
    space = [(0.05, 0.95), (0.05, y0)]
  elif space_type == 'bottom':
    space = [(0.05, 0.95), (y1, 0.95)]
  else:
    raise ValueError('Unknown space type: ' + str(space_type))

  return Space(space)


class Loc(Attribute):
  """Location class."""

  def __init__(self, value=None):
    """Initialize location.

    Args:
      value: None or a tuple of floats
      space: None or a tuple of tuple of floats
      If tuple of floats, then the actual
    """
    super(Loc, self).__init__(value)
    self.attr_type = 'loc'

  def get_space_to(self, space_type):
    if self.value is None:
      return Space(None)
    else:
      x, y = self.value
      return _get_space_to(x, x, y, y, space_type)

  def get_opposite_space_to(self, space_type):
    opposite_space = {'left': 'right',
                      'right': 'left',
                      'top': 'bottom',
                      'bottom': 'top',
                      }[space_type]
    return self.get_space_to(opposite_space)


class Space(Attribute):
  """Space class."""

  def __init__(self, value):
    super(Space, self).__init__(value)
    if self.value is None:
      self._value = [(0, 1), (0, 1)]
    else:
      self._value = value

  def sample(self, avoid=None):
    """Sample a location.

    This function will attempt to find a location to place the object
    that doesn't overlap will other objects at locations avoid,
    but will place an object anyway if it didn't find a good place

    Args:
      avoid: a list of locations (tuples) to be avoided
    """
    if avoid is None:
      avoid = []

    n_max_try = 100
    avoid_radius2 = 0.04  # avoid radius squared
    dx = (self._value[0][1] - self._value[0][0]) * 0.1
    xrange = (self._value[0][0] + dx, self._value[0][1] - dx)
    dy = (self._value[1][1] - self._value[1][0]) * 0.1
    yrange = (self._value[1][0] + dy, self._value[1][1] - dy)
    for i_try in range(n_max_try):
      # Round to 3 decimal places to save space in json dump
      loc = (round(random.uniform(*xrange), 3),
             round(random.uniform(*yrange), 3))

      not_overlapping = True
      for loc_avoid in avoid:
        not_overlapping *= ((loc[0] - loc_avoid[0])**2 +
                            (loc[1] - loc_avoid[1])**2 > avoid_radius2)

      if not_overlapping:
        break

    return Loc(loc)

  def include(self, loc):
    """Check if an unsampled location (a space) includes a loc."""
    x, y = loc.value
    return ((self._value[0][0] < x < self._value[0][1]) and
            (self._value[1][0] < y < self._value[1][1]))

  def get_space_to(self, space_type):
    x0, x1 = self._value[0]
    y0, y1 = self._value[1]
    return _get_space_to(x0, x1, y0, y1, space_type)

  def get_opposite_space_to(self, space_type):
    opposite_space = {'left': 'right',
                      'right': 'left',
                      'top': 'bottom',
                      'bottom': 'top',
                      }[space_type]
    return self.get_space_to(opposite_space)


def static_objects_from_dict(d):
  epochs = d['epochs'] 
  epochs = epochs if isinstance(epochs, list) else [epochs]
  return [StaticObject(loc=tuple(d['location']),
                       color=d['color'],
                       shape=d['shape'],
                       epoch=e)
          for e in epochs]


class StaticObject(object):
  """Object that can be loaded from dataset and rendered."""
  def __init__(self, loc, color, shape, epoch):
    self.loc = loc  # 2-tuple of floats
    self.color = color  # string
    self.shape = shape  # string
    self.epoch = epoch  # int


class StaticObjectSet(object):
  """Provides a subset of functionality provided by ObjectSet.

  This functionality is just enough to create StaticObjectSets from
  json strings and use them to generate feeds for training.
  """
  def __init__(self, n_epoch, static_objects=None, targets=None):
    self.n_epoch = n_epoch

    # {epoch -> [objs]}
    self.dict = defaultdict(list)
    if static_objects:
      for o in static_objects:
        self.add(o)

    self.targets = targets

  def add(self, obj):
    self.dict[obj.epoch].append(obj)

  def select_now(self, epoch_now):
    subset = self.dict[epoch_now]
    # Order objects by location to have a deterministic ordering.
    # Ordering determines occlusion.
    subset.sort(key=lambda o: (o.loc, o.color, o.shape))
    return subset


class Object(object):
  """An object on the screen.

  An object is a collection of attributes.

  Args:
    loc: tuple (x, y)
    color: string ('red', 'green', 'blue', 'white')
    shape: string ('circle', 'square')
    when: string ('now', 'last1', 'latest')
    deletable: boolean. Whether or not this object is deletable. True if
      distractors.

  Raises:
    TypeError if loc, color, shape are neither None nor respective Attributes
  """

  def __init__(self,
               attrs=None,
               when=None,
               deletable=False):

    self.loc = Loc(None)
    self.color = Color(None)
    self.shape = Shape(None)
    self.space = Space(None)

    if attrs is not None:
      for a in attrs:
        if isinstance(a, Loc):
          self.loc = a
        elif isinstance(a, Color):
          self.color = a
        elif isinstance(a, Shape):
          self.shape = a
        elif isinstance(a, Space):
          self.space = a
        else:
          raise TypeError('Unknown type for attribute: ' +
                          str(a) + ' ' + str(type(a)))

    self.when = when
    self.epoch = None
    self.deletable = deletable

  def __str__(self):
    return ' '.join([
        'Object:', 'loc',
        str(self.loc), 'color',
        str(self.color), 'shape',
        str(self.shape), 'when',
        str(self.when), 'epoch',
        str(self.epoch), 'deletable',
        str(self.deletable)
    ])

  def dump(self):
    """Returns representation of self suitable for dumping as json."""
    return {
        'location': self.loc.value,
        'color': self.color.value,
        'shape': self.shape.value,
        'epochs': (self.epoch[0] if self.epoch[0] + 1 == self.epoch[1] else
                   list(range(*self.epoch))),
        'is_distractor': self.deletable
    }

  def to_static(self):
    """Convert self to a list of StaticObjects."""
    return [StaticObject(loc=self.loc.value,
                         color=self.color.value,
                         shape=self.shape.value,
                         epoch=epoch)
            for epoch in range(*self.epoch)]

  def merge(self, obj):
    """Attempt to merge with another object.

    Args:
      obj: an Object Instance

    Returns:
      bool: True if successfully merged, False otherwise
    """
    new_attr = dict()
    # TODO(gryang): What to do with self.when and self.loc?
    for attr_type in ['color', 'shape']:
      if not getattr(self, attr_type).has_value:
        new_attr[attr_type] = getattr(obj, attr_type)
      elif not getattr(obj, attr_type).has_value:
        new_attr[attr_type] = getattr(self, attr_type)
      else:
        return False

    for attr_type in ['color', 'shape']:
      setattr(self, attr_type, new_attr[attr_type])

    return True


class ObjectSet(object):
  """A collection of objects."""

  def __init__(self, n_epoch, n_max_backtrack=4):
    """Initialize the collection of objects.

    Args:
      n_epoch: int, the number of epochs or frames in the object set
      n_max_backtrack: int or None
        If int, maximum number of epoch to look back when searching, at least 1.
        If None, will search the entire history
    """
    self.n_epoch = n_epoch
    self.n_max_backtrack = n_max_backtrack
    self.set = list()
    self.end_epoch = list()
    self.dict = defaultdict(list)  # key: epoch, value: list of obj

    self.last_added_obj = None  # Last added object

  def __iter__(self):
    return self.set.__iter__()

  def __str__(self):
    return '\n'.join([str(o) for o in self])

  def __len__(self):
    return len(self.set)

  def add(self,
          obj,
          epoch_now,
          add_if_exist=False,
          delete_if_can=True):
    """Add an object.

    This function will attempt to add the obj if possible.
    It will not only add the object to the objset, but also instantiate the
    attributes such as color, shape, and loc if not already instantiated.

    Args:
      obj: an Object instance
      epoch_now: the current epoch when this object is added
      add_if_exist: if True, add object anyway. If False, do not add object if
        already exist
      delete_if_can: Boolean. If True, will delete object if it conflicts with
        current object to be added. Should be set to True for most situations.

    Returns:
      obj: the added object if object added. The existing object if not added.

    Raises:
      ValueError: if can't find place to put stimuli
    """

    if obj is None:
      return None

    # Check if already exists
    n_backtrack = self.n_max_backtrack
    obj_subset = self.select(
        epoch_now,
        space=obj.space,
        color=obj.color,
        shape=obj.shape,
        when=obj.when,
        n_backtrack=n_backtrack,
        delete_if_can=delete_if_can,
    )

    if obj_subset and not add_if_exist:  # True if more than zero satisfies
      self.last_added_obj = obj_subset[-1]
      return self.last_added_obj

    # Interpret the object
    if not obj.loc.has_value:
      # Randomly generate locations, but avoid objects already placed now
      avoid = [o.loc.value for o in self.select_now(epoch_now)]
      obj.loc = obj.space.sample(avoid=avoid)

    if not obj.shape.has_value:
      obj.shape.sample()

    if not obj.color.has_value:
      obj.color.sample()

    if obj.when is None:
      # If when is None, then object is always presented
      obj.epoch = [0, self.n_epoch]
    elif obj.when == 'now':
      obj.epoch = [epoch_now, epoch_now + 1]
    elif (obj.when == 'last1') or (obj.when == 'latest'):
      if obj.when == 'last1':
        if epoch_now == 0:
          return None
        epoch_range_right = epoch_now - 1
      else:
        epoch_range_right = epoch_now

      # Place the stimulus is a previous epoch or current epoch
      if self.n_max_backtrack is not None:
        epoch_range_left = max(epoch_now - self.n_max_backtrack, 0)
      else:
        epoch_range_left = 0
      # Note that random.randint is inclusive with right range
      epoch = random.randint(epoch_range_left, epoch_range_right)
      obj.epoch = [epoch, epoch + 1]
    else:
      raise NotImplementedError(
          'When value: {:s} is not implemented'.format(str(obj.when)))

    # Insert and maintain order
    i = bisect_left(self.end_epoch, obj.epoch[1])
    self.set.insert(i, obj)
    self.end_epoch.insert(i, obj.epoch[1])

    # Add to dict
    for epoch in range(obj.epoch[0], obj.epoch[1]):
      self.dict[epoch].append(obj)

    self.last_added_obj = obj
    return self.last_added_obj

  def add_distractor(self, epoch_now):
    """Add a distractor."""
    attr1 = random_colorshape()
    obj1 = Object(attr1, when='now', deletable=True)
    self.add(obj1, epoch_now, add_if_exist=True)

  def delete(self, obj):
    """Delete an object."""
    i = self.set.index(obj)
    self.set.pop(i)
    self.end_epoch.pop(i)

    for epoch in range(obj.epoch[0], obj.epoch[1]):
      self.dict[epoch].remove(obj)

  def shift(self, x):
    """Shift every object in the set.

    Args:
      x: int, shift every object by x-epoch.
          An object that originally stays between (a, b) now stays between
          (max(0,a+x), b+x).

    Raises:
      ValueError: if n_epoch + x <= 0
    """
    self.n_epoch += x
    if self.n_epoch < 1:
      raise ValueError('n_epoch + x <= 0')

    new_set = list()
    new_end_epoch = list()
    new_dict = defaultdict(list)

    for obj in self.set:
      obj.epoch[0] = max((0, obj.epoch[0] + x))
      obj.epoch[1] += x
      if obj.epoch[1] > 0:
        new_set.append(obj)
        new_end_epoch.append(obj.epoch[1])

        for epoch in range(obj.epoch[0], obj.epoch[1]):
          new_dict[epoch].append(obj)

    self.set = new_set
    self.end_epoch = new_end_epoch
    self.dict = new_dict

  def select(self,
             epoch_now,
             space=None,
             color=None,
             shape=None,
             when=None,
             n_backtrack=None,
             delete_if_can=True
            ):
    """Select an object satisfying properties.

    Args:
      epoch_now: int, the current epoch
      space: None or a Loc instance, the loc to be selected.
      color: None or a Color instance, the color to be selected.
      shape: None or a Shape instance, the shape to be selected.
      when: None or a string, the temporal window to be selected.
      n_backtrack: None or int, the number of epochs to backtrack
      delete_if_can: boolean, delete object found if can

    Returns:
      a list of Object instance that fit the pattern provided by arguments
    """
    space = space or Space(None)
    color = color or Color(None)
    shape = shape or Shape(None)

    if not isinstance(color, Color):
      raise TypeError('color has to be Color class, is instead of class ' +
                      str(type(color)))
    if not isinstance(shape, Shape):
      raise TypeError('shape has to be Shape class, is instead of class ' +
                      str(type(shape)))
    assert isinstance(space, Space)

    if when == 'now':
      # Use the fast implementation
      return self.select_now(epoch_now, space, color, shape, delete_if_can)

    if when == 'last1':
      epoch_now -= 1

    if n_backtrack is None:
      n_backtrack = self.n_max_backtrack

    epoch_stop = max(0, epoch_now - n_backtrack)

    while epoch_now >= epoch_stop:
      subset = self.select_now(epoch_now, space, color, shape, delete_if_can)
      if subset:
        return subset
      epoch_now -= 1
    return []

  def select_now(self,
                 epoch_now,
                 space=None,
                 color=None,
                 shape=None,
                 delete_if_can=False
                ):
    """Select all objects presented now that satisfy properties."""
    # Select only objects that have happened
    subset = self.dict[epoch_now]

    if color is not None and color.has_value:
      subset = [o for o in subset if o.color == color]

    if shape is not None and shape.has_value:
      subset = [o for o in subset if o.shape == shape]

    if space is not None and space.has_value:
      subset = [o for o in subset if space.include(o.loc)]

    if delete_if_can:
      for o in subset:
        if o.deletable:
          # delete obj from self
          self.delete(o)
      # Keep the not-deleted
      subset = [o for o in subset if not o.deletable]

    # Order objects by location to have a deterministic ordering
    subset.sort(key=lambda o: (o.loc.value, o.color.value, o.shape.value))

    return subset


def render_static_obj(canvas, obj, img_size):
  """Render a single object.

  Args:
    canvas: numpy array of type int8 (img_size, img_size, 3). Modified in place.
        Converted to PIL from OpenCV. Font is different from original COG
    obj: StaticObject instance
    img_size: int, image size.
  """
  # Fixed specifications
  radius = int(0.05 * img_size)

	# Convert to PIL image.
  image = Image.fromarray(np.uint8(canvas),'RGB')
  draw = ImageDraw.Draw(image)

  # Note that OpenCV color is (Blue, Green, Red)
	# Converted to PIL, so there is array rearrangements.
  color = const.WORD2COLOR[obj.color]
  shape = obj.shape
  center = (int(obj.loc[0] * img_size), int(obj.loc[1] * img_size))
  if shape == 'circle':
    draw.ellipse((center[0]-radius,center[1]-radius,center[0]+radius,center[1]+radius),fill=color)
    #cv2.circle(canvas, center, radius, color, -1)
  elif shape == 'square':
    draw.rectangle((center[0]-radius,center[1]-radius,center[0]+radius,center[1]+radius),fill=color)
    #cv2.rectangle(canvas, (center[0] - radius, center[1] - radius),
    #              (center[0] + radius, center[1] + radius), color, -1)
  elif shape == 'cross':
    thickness = int(0.02 * img_size)
    draw.line((center[0] - radius, center[1], center[0] + radius, center[1]), fill=color, width=thickness)
    draw.line((center[0], center[1] - radius, center[0], center[1] + radius), fill=color, width=thickness)
    #cv2.line(canvas, (center[0] - radius, center[1]),
    #         (center[0] + radius, center[1]), color, thickness)
    #cv2.line(canvas, (center[0], center[1] - radius),
    #        (center[0], center[1] + radius), color, thickness)
  elif shape == 'triangle':
    r1 = int(0.08 * img_size)
    r2 = int(0.04 * img_size)
    r3 = int(0.069 * img_size)
    pts = np.array([(center[0], center[1] - r1),
                    (center[0] - r3, center[1] + r2), (center[0] + r3,
                                                       center[1] + r2)])
    #cv2.fillConvexPoly(canvas, pts, color)
    draw.polygon(pts,fill=color)
  elif shape == 'vbar':
    r1 = int(0.5 * radius)
    r2 = int(1.2 * radius)
    #cv2.rectangle(canvas, (center[0] - r1, center[1] - r2),
    #              (center[0] + r1, center[1] + r2), color, -1)
    draw.rectangle((center[0] - r1, center[1] - r2, center[0] + r1, center[1] + r2),fill=color)
  elif shape == 'hbar':
    r1 = int(1.2 * radius)
    r2 = int(0.5 * radius)
    draw.rectangle((center[0] - r1, center[1] - r2, center[0] + r1, center[1] + r2),fill=color)
    #cv2.rectangle(canvas, (center[0] - r1, center[1] - r2),
    #              (center[0] + r1, center[1] + r2), color, -1)
  elif shape in string.ascii_letters:
    #font = cv2.FONT_HERSHEY_SIMPLEX
    # Shift x and y by -3 and 5 respectively to center the character
    #cv2.putText(canvas, shape, (center[0]-3, center[1]+5), font, 0.5, color, 2)
    font = ImageFont.truetype(os.path.join(os.path.dirname(__file__),'roboto.ttf' ),18)
    draw.fontmode = '1'
    draw.text((center[0]-3, center[1]-10), shape, color,font=font)
  else:
    raise NotImplementedError('Unknown shape ' + str(shape))

  canvas[:] = np.array(np.float32(image)) 

def render_obj(canvas, obj, img_size):
  """Render a single object.

  Args:
    canvas: numpy array of type int8 (img_size, img_size, 3). Modified in place.
        Importantly, opencv default is (B, G, R) instead of (R,G,B)
				Converted to PIL, subfunctions handle the switch
    obj: Object or StaticObject instance, containing object information
    img_size: int, image size.
  """
  if isinstance(obj, StaticObject):
    render_static_obj(canvas, obj, img_size)
  else:
    render_static_obj(canvas, obj.to_static()[0], img_size)


def render_static(objlists, img_size=224, save_name=None):
  """Render a movie by epoch.

  Args:
    objlists: a list of lists of StaticObject instances
    img_size: int, size of image (both x and y)
    save_name: if not None, save movie at save_name

  Returns:
    movie: numpy array (n_time, img_size, img_size, 3)
  """

  n_epoch_max = max([o.epoch for objlist in objlists for o in objlist]) + 1

  # list of lists of lists. Each inner-most list contains
  # objects in a given epoch from a certain objlist.
  by_epoch = []
  key = lambda o: o.epoch
  for objects in objlists:
    by_epoch.append([])
    # Sort objects by epoch
    objects.sort(key=key)
    epoch_obj_dict = defaultdict(list,
        [(epoch, list(group)) for epoch, group
          in itertools.groupby(objects, key)])
    for i in range(n_epoch_max):
      # Order objects by location so that ordering is deterministic.
      # It controls occlusion.
      os = epoch_obj_dict[i]
      os.sort(key=lambda o: o.loc)
      by_epoch[-1].append(os)

  # It's faster if use uint8 here, but later conversion to float32 seems slow
  movie = np.zeros((len(objlists) * n_epoch_max, img_size, img_size, 3),
      np.float32)

  i_frame = 0
  for objects in by_epoch:
    for epoch_objs in objects:
      canvas = movie[i_frame:i_frame + 1, ...]  # return a view
      canvas = np.squeeze(canvas, axis=0)
      for obj in epoch_objs:
        render_static_obj(canvas, obj, img_size)
      i_frame += 1
  assert i_frame == len(objlists) * n_epoch_max, '%d != %d' % (
      i_frame, len(objlists) * n_epoch_max)

  if save_name is not None:
    t_total = len(objlists) * n_epoch_max * 1.0  # need fps >= 1
    save_movie(movie, save_name, t_total)

  return movie


def render(objsets, img_size=224, save_name=None):
  """Render a movie by epoch.

  Args:
    objsets: an ObjsetSet instance or a list of them
    img_size: int, size of image (both x and y)
    save_name: if not None, save movie at save_name

  Returns:
    movie: numpy array (n_time, img_size, img_size, 3)
  """
  if not isinstance(objsets, list):
    objsets = [objsets]

  n_objset = len(objsets)
  n_epoch_max = max([objset.n_epoch for objset in objsets])

  # It's faster if use uint8 here, but later conversion to float32 seems slow
  movie = np.zeros((n_objset * n_epoch_max, img_size, img_size, 3), np.float32)

  i_frame = 0
  for objset in objsets:
    for epoch_now in range(n_epoch_max):
      canvas = movie[i_frame:i_frame + 1, ...]  # return a view
      canvas = np.squeeze(canvas, axis=0)

      subset = objset.select_now(epoch_now)
      for obj in subset:
        render_obj(canvas, obj, img_size)

      i_frame += 1

  if save_name is not None:
    t_total = n_objset * n_epoch_max * 1.0  # need fps >= 1
    save_movie(movie, save_name, t_total)

  return movie


def render_target(movie, target):
  """Specifically render the target response.

  Args:
    movie: numpy array (n_time, img_size, img_size, 3)
    target: list of tuples. List has to be length n_time

  Returns:
    movie_withtarget: same format as movie, but with target response shown

  Raises:
    TypeError: when target type is incorrect.
  """

  movie_withtarget = movie.copy()

  img_size = movie[0].shape[0]
  radius = int(0.02 * img_size)

  for frame, target_now in zip(movie_withtarget, target):
    if isinstance(target_now, Loc):
      loc = target_now.value
      center = (int(loc[0] * img_size), int(loc[1] * img_size))
      #cv2.circle(frame, center, radius, (255, 255, 255), -1)
      image = Image.fromarray(frame,'RGB')
      draw = ImageDraw.Draw(image)
      draw.ellipse((center[0]-radius,center[1]-radius,center[0]+radius,center[1]+radius),fill=(255,255,255))
      frame[:] = np.array(image)[:]
      
    else:
      if target_now == const.INVALID:
        string = 'invalid'
      elif isinstance(target_now, bool):
        string = 'true' if target_now else 'false'
      elif isinstance(target_now, Attribute):
        string = target_now.value
      elif isinstance(target_now, str):
        string = target_now
      else:
        raise TypeError('Unknown target type.')

      #font = cv2.FONT_HERSHEY_SIMPLEX
      #cv2.putText(frame, string, (int(0.1 * img_size), int(0.8 * img_size)),
      #            font, 0.5, (255, 255, 255))
      image = Image.fromarray(frame,'RGB')
      draw = ImageDraw.Draw(image)
      draw.text((int(0.1 * img_size), int(0.8 * img_size)), string, (255,255,255))
      frame[:] = np.array(image)[:]

  return movie_withtarget


def random_attr(attr_type):
  if attr_type == 'color':
    return random_color()
  elif attr_type == 'shape':
    return random_shape()
  elif attr_type == 'loc':
    return Loc([round(random.uniform(0.05, 0.95), 3),
                round(random.uniform(0.05, 0.95), 3)])
  else:
    raise NotImplementedError('Unknown attr_type :' + str(attr_type))


def random_space():
  return random.choice(const.ALLSPACES)


def n_random_space():
  return len(const.ALLSPACES)


def random_color():
  return Color(random.choice(const.ALLCOLORS))


def n_random_color():
  return len(const.ALLCOLORS)


def random_shape():
  return Shape(random.choice(const.ALLSHAPES))


def n_random_shape():
  return len(const.ALLSHAPES)


def random_colorshape():
  color, shape = random.choice(const.ALLCOLORSHAPES)
  return Color(color), Shape(shape)


def n_random_colorshape():
  return len(const.ALLCOLORSHAPES)


def random_when():
  """Random choose a when property.

  Here we use the numpy random generator to provide different probabilities.

  Returns:
    when: a string.
  """
  return np.random.choice(const.ALLWHENS, p=const.ALLWHENS_PROB)


def n_random_when():
  return len(const.ALLWHENS)


def sample_color(k):
  return [Color(c) for c in random.sample(const.ALLCOLORS, k)]


def n_sample_color(k):
  return np.prod(range(len(const.ALLCOLORS)-k+1, len(const.ALLCOLORS)+1))


def sample_shape(k):
  return [Shape(s) for s in random.sample(const.ALLSHAPES, k)]


def n_sample_shape(k):
  return np.prod(range(len(const.ALLSHAPES)-k+1, len(const.ALLSHAPES)+1))


def sample_colorshape(k):
  return [
      (Color(c), Shape(s)) for c, s in random.sample(const.ALLCOLORSHAPES, k)
  ]


def n_sample_colorshape(k):
  return np.prod(
      range(len(const.ALLCOLORSHAPES)-k+1, len(const.ALLCOLORSHAPES)+1))


def another_color(color):
  allcolors = list(const.ALLCOLORS)
  if isinstance(color,list):
    for c in color:
      allcolors.remove(c.value)
  else:
    allcolors.remove(color.value)

  return Color(random.choice(allcolors))


def another_shape(shape):
  allshapes = list(const.ALLSHAPES)
  if isinstance(shape,list):
    for s in shape:
      allshapes.remove(s.value)
  else:
    allshapes.remove(shape.value)

  return Shape(random.choice(allshapes))


def another_loc(space):
  n_max_try = 100
  for i_try in range(n_max_try):
    loc = Loc((round(random.uniform(0.05, 0.95), 3),
               round(random.uniform(0.05, 0.95), 3)))
    if not space.include(loc):
      break
  return loc


def another_attr(attr):
  if isinstance(attr, Color):
    return another_color(attr)
  elif isinstance(attr, Shape):
    return another_shape(attr)
  elif isinstance(attr, Space):
    return another_loc(attr)
  elif attr == const.INVALID:
    return attr
  else:
    raise TypeError(
      'Type {:s} of {:s} is not supported'.format(str(attr), str(type(attr))))


def another_colorshape(color_shape):
  allcolorshapes = list(const.ALLCOLORSHAPES)
  try:
    allcolorshapes.remove((color_shape[0].value, color_shape[1].value))
  except AttributeError:
    for c, s in color_shape:
      allcolorshapes.remove((c.value, s.value))
  c, s = random.choice(allcolorshapes)
  return Color(c), Shape(s)

