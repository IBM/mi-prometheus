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

"""Vocabulary of functional programs.

Contains the building blocks for permissible tasks.
"""

from collections import defaultdict
import random

from miprometheus.problems.seq_to_seq.vqa.cog.cog_utils import constants as const
from miprometheus.problems.seq_to_seq.vqa.cog.cog_utils import stim_generator as sg


def obj_str(loc=None, color=None, shape=None,
            when=None, space_type=None):
  """Get a string describing an object with attributes."""

  loc = loc or sg.Loc(None)
  color = color or sg.Color(None)
  shape = shape or sg.Shape(None)

  sentence = []
  if when is not None:
    sentence.append(when)
  if isinstance(color, sg.Attribute) and color.has_value:
    sentence.append(str(color))
  if isinstance(shape, sg.Attribute) and shape.has_value:
    sentence.append(str(shape))
  else:
    sentence.append('object')
  if isinstance(color, Operator):
    sentence += ['with', str(color)]
  if isinstance(shape, Operator):
    if isinstance(color, Operator):
      sentence.append('and')
    sentence += ['with', str(shape)]
  if isinstance(loc, Operator):
    sentence += ['on', space_type, 'of', str(loc)]
  return ' '.join(sentence)


class Skip(object):
  """Skip this operator."""
  def __init__(self):
    pass


class Task(object):
  """Base class for tasks."""

  def __init__(self, operator=None):
    if operator is None:
      self._operator = Operator()
    else:
      if not isinstance(operator, Operator):
        raise TypeError('operator is the wrong type ' + str(type(operator)))
      self._operator = operator

  def __call__(self, objset, epoch_now):
    return self._operator(objset, epoch_now)

  def __str__(self):
    return str(self._operator)

  def _get_all_nodes(self, op, visited):
    """Get the total number of operators in the graph starting with op."""
    visited[op] = True
    all_nodes = [op]
    for c in op.child:
      if isinstance(c, Operator) and not visited[c]:
        all_nodes.extend(self._get_all_nodes(c, visited))
    return all_nodes

  @property
  def _all_nodes(self):
    """Return all nodes in a list."""
    visited = defaultdict(lambda: False)
    return self._get_all_nodes(self._operator, visited)

  @property
  def operator_size(self):
    """Return the number of unique operators."""
    return len(self._all_nodes)

  def topological_sort_visit(self, node, visited, stack):
    """Recursive function that visits a node."""

    # Mark the current node as visited.
    visited[node] = True

    # Recur for all the vertices adjacent to this vertex
    for child in node.child:
      if isinstance(child, Operator) and not visited[child]:
        self.topological_sort_visit(child, visited, stack)

    # Push current vertex to stack which stores result
    stack.insert(0, node)

  def topological_sort(self):
    """Perform a topological sort."""
    nodes = self._all_nodes

    # Mark all the vertices as not visited
    visited = defaultdict(lambda: False)
    stack = []

    # Call the recursive helper function to store Topological
    # Sort starting from all vertices one by one
    for node in nodes:
      if not visited[node]:
        self.topological_sort_visit(node, visited, stack)

    # Print contents of stack
    return stack

  def guess_objset(self, objset, epoch_now, should_be=None):
    nodes = self.topological_sort()
    should_be_dict = defaultdict(lambda: None)

    if should_be is not None:
      should_be_dict[nodes[0]] = should_be

    for node in nodes:
      should_be = should_be_dict[node]
      if isinstance(should_be, Skip):
        inputs = Skip() if len(node.child)==1 else [Skip()] * len(node.child)
      elif isinstance(node, Select):
        inputs = node.get_expected_input(should_be, objset, epoch_now)
        objset = inputs[0]
        inputs = inputs[1:]
      elif isinstance(node, IsSame) or isinstance(node, And):
        inputs = node.get_expected_input(should_be, objset, epoch_now)
      else:
        inputs = node.get_expected_input(should_be)

      if len(node.child) == 1:
        outputs = [inputs]
      else:
        outputs = inputs

      # Update the should_be dictionary for the node children
      for c, output in zip(node.child, outputs):
        if not isinstance(c, Operator):
          continue
        if isinstance(output, Skip):
          should_be_dict[c] = Skip()
          continue
        if should_be_dict[c] is None:
          # If not assigned, assign
          should_be_dict[c] = output
        else:
          # If assigned, for each object, try to merge them
          if isinstance(c, Select):
            # Loop over new output
            for o in output:
              assert isinstance(o, sg.Object)
              merged = False
              # Loop over previously assigned outputs
              for s in should_be_dict[c]:
                # Try to merge
                merged = s.merge(o)
                if merged:
                  break
              if not merged:
                should_be_dict[c].append(o)
          else:
            raise NotImplementedError()

    return objset

  @property
  def instance_size(self):
    """Return the total number of possible instantiated tasks."""
    raise NotImplementedError('instance_size is not defined for this task.')

  def generate_objset(self, n_epoch, n_distractor=1, average_memory_span=2):
    """Guess objset for all n_epoch.

    Mathematically, the average_memory_span is n_max_backtrack/3

    Args:
      n_epoch: int, total number of epochs
      n_distractor: int, number of distractors to add
      average_memory_span: int, the average number of epochs by which an object
        need to be held in working memory, if needed at all

    Returns:
      objset: full objset for all n_epoch
    """
    n_max_backtrack = int(average_memory_span*3)
    objset = sg.ObjectSet(n_epoch=n_epoch, n_max_backtrack=n_max_backtrack)

    # Guess objects
    # Importantly, we generate objset backward in time
    epoch_now = n_epoch - 1
    while epoch_now >= 0:
      for _ in range(n_distractor):
        objset.add_distractor(epoch_now)  # distractor
      objset = self.guess_objset(objset, epoch_now)
      epoch_now -= 1

    return objset

  def get_target(self, objset):
    return [self(objset, epoch_now) for epoch_now in range(objset.n_epoch)]


class Operator(object):
  """Base class for task constructors."""

  def __init__(self):
    # Whether or not this operator is the final operator
    self.child = list()
    self.parent = list()

  def __str__(self):
    pass

  def __call__(self, objset, epoch_now):
    pass

  def set_child(self, child):
    """Set operators as children."""
    try:
      child.parent.append(self)
      self.child.append(child)
    except AttributeError:
      for c in child:
        self.set_child(c)

  def get_expected_input(self, should_be=None, objset=None,epoch_now=None):
    """Guess and update the objset at this epoch.

    Args:
      should_be: the supposed output
    """
    raise NotImplementedError('get_expected_input method is not defined.')


class Select(Operator):
  """Selecting the objects that satisfy properties."""

  def __init__(self,
               loc=None,
               color=None,
               shape=None,
               when=None,
               space_type=None):
    super(Select, self).__init__()

    loc = loc or sg.Loc(None)
    color = color or sg.Color(None)
    shape = shape or sg.Shape(None)

    if isinstance(loc, Operator) or loc.has_value:
      assert space_type is not None

    self.loc, self.color, self.shape = loc, color, shape
    self.set_child([loc, color, shape])

    self.when = when
    self.space_type = space_type

  def __str__(self):
    """Get the language for color, shape, when combination."""
    return obj_str(
      self.loc, self.color, self.shape, self.when, self.space_type)

  def __call__(self, objset, epoch_now):
    """Return subset of objset."""
    loc = self.loc(objset, epoch_now)
    color = self.color(objset, epoch_now)
    shape = self.shape(objset, epoch_now)

    if const.INVALID in (loc, color, shape):
      return const.INVALID

    space = loc.get_space_to(self.space_type)

    subset = objset.select(
        epoch_now,
        space=space,
        color=color,
        shape=shape,
        when=self.when,
    )

    return subset

  def get_expected_input(self, should_be, objset, epoch_now):
    """Guess objset for Select operator.

    Optionally modify the objset based on the target output, should_be,
    and pass down the supposed input attributes

    There are two sets of attributes to compute:
    (1) The attributes of the expected input, i.e. the attributes to select
    (2) The attributes of new object being added to the objset

    There are two main scenarios:
    (1) the target output is empty
    (2) the target output is not empty, then it has to be a list with a single
    Object instance in it

    When (1) the target output is empty, the goal is to make sure
    attr_newobject and attr_expectedinput differ by a single attribute

    When (2) the target output is not empty, then
    attr_newobject and attr_expectedinput are the same

    We first decide the attributes of the expected inputs.

    If target output is empty, then expected inputs is determined solely
    based on its actual inputs. If the actual inputs can already be computed,
    then use it, and skip the traversal of following nodes. Otherwise, use a
    random allowed value.

    If target output is not empty, again, if the actual inputs can be computed,
    use it. If the actual input can not be computed, but the attribute is
    specified by the target output, use that. Otherwise, use a random value.

    Then we determine the new object being added.

    If target output is empty, then randomly change one of the attributes that
    is not None.
    For self.space attribute, if it is an operator, then add the object at the
    opposite side of space

    If target output is not empty, use the attributes of the expected input

    Args:
      should_be: a list of Object instances
      objset: objset instance
      epoch_now: int, the current epoch

    Returns:
      objset: objset instance
      space: supposed input
      color: supposed input
      shape: supposed input

    Raises:
      TypeError: when should_be is not a list of Object instances
      NotImplementedError: when should_be is a list and has length > 1
    """
    # print(should_be[0])
    if should_be is None:
      raise NotImplementedError('This should not happen.')

    if should_be:
      # Making sure should_be is a length-1 list of Object instance
      for s in should_be:
        if not isinstance(s, sg.Object):
          raise TypeError('Wrong type for items in should_be: '
                          + str(type(s)))

      if len(should_be) > 1:
        for s in should_be:
          print(s)
        raise NotImplementedError()
      obj_should_be = should_be[0]

      attr_new_object = list()
      attr_type_not_fixed = list()
      # First evaluate the inputs that can be evaluated
      for attr_type in ['loc', 'color', 'shape']:
        a = getattr(self, attr_type)
        attr = a(objset, epoch_now)
        # If the input is successfully evaluated
        if attr != const.INVALID and attr.has_value:
          if attr_type == 'loc':
            attr = attr.get_space_to(self.space_type)
          attr_new_object.append(attr)
        else:
          attr_type_not_fixed.append(attr_type)

      # Add an object based on these attributes
      # Note that, objset.add() will implicitly select the objset
      obj = sg.Object(attr_new_object, when=self.when)
      obj = objset.add(obj, epoch_now, add_if_exist=False)

      if obj is None:
        return objset, Skip(), Skip(), Skip()

      # If some attributes of the object are given by should_be, use them
      for attr_type in attr_type_not_fixed:
        a = getattr(obj_should_be, attr_type)
        if a.has_value:
          setattr(obj, attr_type, a)

      # If an attribute of select is an operator, then the expected input is
      # the value of obj
      attr_expected_in = list()
      for attr_type in ['loc', 'color', 'shape']:
        a = getattr(self, attr_type)
        if isinstance(a, Operator):
          # Only operators need expected_in
          if attr_type == 'loc':
            space = obj.loc.get_opposite_space_to(self.space_type)
            attr = space.sample()
          else:
            attr = getattr(obj, attr_type)
          attr_expected_in.append(attr)
        else:
          attr_expected_in.append(Skip())

    if not should_be:
      # First determine the attributes to flip later
      attr_type_to_flip = list()
      for attr_type in ['loc', 'color', 'shape']:
        a = getattr(self, attr_type)
        # If attribute is operator or is a specified value
        if isinstance(a, Operator) or a.has_value:
          attr_type_to_flip.append(attr_type)

      # Now generate expected input attributes
      attr_expected_in = list()
      attr_new_object = list()
      for attr_type in ['loc', 'color', 'shape']:
        a = getattr(self, attr_type)
        attr = a(objset, epoch_now)
        if isinstance(a, Operator):
          if attr == const.INVALID:
            # Can not be evaluated yet, then randomly choose one
            attr = sg.random_attr(attr_type)
          attr_expected_in.append(attr)
        else:
          attr_expected_in.append(Skip())

        if attr_type in attr_type_to_flip:
          # Candidate attribute values for the new object
          attr_new_object.append(attr)

      # Randomly pick one attribute to flip
      attr_type = random.choice(attr_type_to_flip)
      i = attr_type_to_flip.index(attr_type)
      if attr_type == 'loc':
        # If flip location, place it in the opposite direction
        loc = attr_new_object[i]
        attr_new_object[i] = loc.get_opposite_space_to(self.space_type)
      else:
        # Select a different attribute
        attr_new_object[i] = sg.another_attr(attr_new_object[i])
        # Not flipping loc, so place it in the correct direction
        if 'loc' in attr_type_to_flip:
          j = attr_type_to_flip.index('loc')
          loc = attr_new_object[j]
          attr_new_object[j] = loc.get_space_to(self.space_type)

      # Add an object based on these attributes
      obj = sg.Object(attr_new_object, when=self.when)
      objset.add(obj, epoch_now, add_if_exist=False)

    # Return the expected inputs
    return [objset] + attr_expected_in


class Get(Operator):
  """Get attribute of an object."""

  def __init__(self, attr_type, objs):
    """Get attribute of an object.

    Args:
      attr_type: string, color, shape, or loc. The type of attribute to get.
      objs: Operator instance or Object instance
    """
    super(Get, self).__init__()
    self.attr_type = attr_type
    self.objs = objs
    assert isinstance(objs, Operator)
    self.set_child(objs)

  def __str__(self):
    words = [self.attr_type, 'of', str(self.objs)]
    if not self.parent:
      words += ['?']
    return ' '.join(words)

  def __call__(self, objset, epoch_now):
    """Get the attribute.

    By default, get the attribute of the unique object. If there are
    multiple objects, then return INVALID.

    Args:
      objset: objset
      epoch_now: epoch now

    Returns:
      attr: Attribute instance or INVALID
    """
    if isinstance(self.objs, Operator):
      objs = self.objs(objset, epoch_now)
    else:
      objs = self.objs

    if objs == const.INVALID:
      return const.INVALID
    elif len(objs) != 1:
      # Ambiguous or non-existent
      return const.INVALID
    else:
      return getattr(objs[0], self.attr_type)

  def get_expected_input(self, should_be):
    if should_be is None:
      should_be = sg.random_attr(self.attr_type)
    objs = sg.Object([should_be])
    return [objs]


class GetColor(Get):
  """Go color of object."""

  def __init__(self, objs):
    super(GetColor, self).__init__('color', objs)


class GetShape(Get):
  """Go shape of object."""

  def __init__(self, objs):
    super(GetShape, self).__init__('shape', objs)


class GetLoc(Get):
  """Get location of object."""

  def __init__(self, objs):
    super(GetLoc, self).__init__('loc', objs)

  def __str__(self):
    return str(self.objs)


class Go(Get):
  """Go to location of object."""

  def __init__(self, objs):
    super(Go, self).__init__('loc', objs)

  def __str__(self):
    return ' '.join(['point', str(self.objs)])


class GetTime(Operator):
  """Get time of an object.

  This operator is not tested and not finished.
  """

  def __init__(self, objs):
    """Get attribute of an object.

    Args:
      attr_type: string, color, shape, or loc. The type of attribute to get.
      objs: Operator instance or Object instance
    """
    super(GetTime, self).__init__()
    self.attr_type = 'time'
    self.objs = objs
    assert isinstance(objs, Operator)
    self.set_child(objs)

  def __str__(self):
    words = [self.attr_type, 'of', str(self.objs)]
    if not self.parent:
      words += ['?']
    return ' '.join(words)

  def __call__(self, objset, epoch_now):
    """Get the attribute.

    By default, get the attribute of the unique object. If there are
    multiple objects, then return INVALID.

    Args:
      objset: objset
      epoch_now: epoch now

    Returns:
      attr: Attribute instance or INVALID
    """
    if isinstance(self.objs, Operator):
      objs = self.objs(objset, epoch_now)
    else:
      objs = self.objs
    if objs == const.INVALID:
      return const.INVALID
    elif len(objs) != 1:
      # Ambiguous or non-existent
      return const.INVALID
    else:
      # TODO(gryang): this only works when object is shown for a single epoch
      return objs[0].epoch[0]

  def get_expected_input(self, should_be):
    raise NotImplementedError()
    #if should_be is None:
    #  should_be = sg.random_attr(self.attr_type)
    #objs = sg.Object([should_be])
    #return [objs]


class Exist(Operator):
  """Check if object with property exists."""

  def __init__(self, objs):
    super(Exist, self).__init__()
    self.objs = objs
    assert isinstance(objs, Operator)
    self.set_child(objs)

  def __str__(self):
    words = [str(self.objs), 'exist']
    if not self.parent:
      words += ['?']
    return ' '.join(words)

  def __call__(self, objset, epoch_now):
    subset = self.objs(objset, epoch_now)
    if subset == const.INVALID:
      return const.INVALID
    elif subset:
      # If subset is not empty
      return True
    else:
      return False

  def get_expected_input(self, should_be):
    if self.objs.when != 'now':
      raise ValueError("""
      Guess objset is not supported for the Exist class
      for when other than now""")

    if should_be is None:
      should_be = random.random() > 0.5

    if should_be:
      should_be = [sg.Object()]
    else:
      should_be = []

    return should_be


class Switch(Operator):
  """Switch behaviors based on trueness of statement.

  Args:
    statement: boolean type operator
    do_if_true: operator performed if the evaluated statement is True
    do_if_false: operator performed if the evaluated statement is False
    invalid_as_false: When True, invalid statement is treated as False
    both_options_avail: When True, both do_if_true and do_if_false will be
      called the get_expected_input function
  """

  def __init__(self,
               statement,
               do_if_true,
               do_if_false,
               invalid_as_false=False,
               both_options_avail=True):
    super(Switch, self).__init__()

    self.statement = statement
    self.do_if_true = do_if_true
    self.do_if_false = do_if_false

    self.set_child([statement, do_if_true, do_if_false])

    self.invalid_as_false = invalid_as_false
    self.both_options_avail = both_options_avail

  def __str__(self):
    assert not self.parent
    words = [
        'if',
        str(self.statement), ',', 'then',
        str(self.do_if_true), ',', 'else',
        str(self.do_if_false)
    ]
    if not self.parent:
      words += ['.']
    return ' '.join(words)

  def __call__(self, objset, epoch_now):
    statement_true = self.statement(objset, epoch_now)
    if statement_true == const.INVALID:
      if self.invalid_as_false:
        statement_true = False
      else:
        return const.INVALID

    if statement_true:
      return self.do_if_true(objset, epoch_now)
    else:
      return self.do_if_false(objset, epoch_now)

  def __getattr__(self, key):
    """Get attributes.

    The Switch operator should also have the shared attributes of do_if_true and
    do_if_false. Because regardless of the statement truthness, the common
    attributes will be true.

    Args:
      key: attribute

    Returns:
      attr1: common attribute of do_if_true and do_if_false

    Raises:
      ValueError: if the attribute is not common. Temporary.
    """
    attr1 = getattr(self.do_if_true, key)
    attr2 = getattr(self.do_if_false, key)
    if attr1 == attr2:
      return attr1
    else:
      raise ValueError()

  def get_expected_input(self, should_be):
    """Guess objset for Switch operator.

    Here the objset is guessed based on whether or not the statement should be
    true.
    Both do_if_true and do_if_false should be executable regardless of the
    statement truthfulness. In this way, it doesn't provide additional
    information.

    Args:
      objset: objset
      epoch_now: current epoch

    Returns:
      objset: updated objset
    """
    if should_be is None:
      should_be = random.random() > 0.5

    return should_be, None, None


class IsSame(Operator):
  """Check if two attributes are the same."""

  def __init__(self, attr1, attr2):
    """Compare to attributes.

    Args:
      attr1: Instance of Attribute or Get
      attr2: Instance of Attribute or Get

    Raises:
      ValueError: when both attr1 and attr2 are instances of Attribute
    """
    super(IsSame, self).__init__()

    self.attr1 = attr1
    self.attr2 = attr2

    self.attr1_is_op = isinstance(self.attr1, Operator)
    self.attr2_is_op = isinstance(self.attr2, Operator)

    self.set_child([self.attr1, self.attr2])

    if (not self.attr1_is_op) and (not self.attr2_is_op):
      raise ValueError('attr1 and attr2 can not both be Attribute instances.')

    self.attr_type = self.attr1.attr_type
    assert self.attr_type == self.attr2.attr_type

  def __str__(self):
    words = [self.attr1.__str__(), 'equal', self.attr2.__str__()]
    if not self.parent:
      words += ['?']
    return ' '.join(words)

  def __call__(self, objset, epoch_now):
    attr1 = self.attr1(objset, epoch_now)
    attr2 = self.attr2(objset, epoch_now)

    if (attr1 == const.INVALID) or (attr2 == const.INVALID):
      return const.INVALID
    else:
      return attr1 == attr2

  def get_expected_input(self, should_be, objset, epoch_now):
    if should_be is None:
      should_be = random.random() > 0.5

    # Determine which attribute should be fixed and which shouldn't
    attr1_value = self.attr1(objset, epoch_now)
    attr2_value = self.attr2(objset, epoch_now)

    attr1_fixed = attr1_value != const.INVALID
    attr2_fixed = attr2_value != const.INVALID

    if attr1_fixed:
      assert attr1_value.has_value

    if attr1_fixed and attr2_fixed:
      # do nothing
      attr1_assign, attr2_assign = Skip(), Skip()
    elif attr1_fixed and not attr2_fixed:
      attr1_assign = Skip()
      attr2_assign = attr1_value if should_be else sg.another_attr(attr1_value)
    elif not attr1_fixed and attr2_fixed:
      attr1_assign = attr2_value if should_be else sg.another_attr(attr2_value)
      attr2_assign = Skip()
    else:
      attr = sg.random_attr(self.attr_type)
      attr1_assign = attr
      attr2_assign = attr if should_be else sg.another_attr(attr)

    return attr1_assign, attr2_assign


class And(Operator):
  """And operator."""

  def __init__(self, op1, op2):
    super(And, self).__init__()
    self.op1, self.op2 = op1, op2
    self.set_child([op1, op2])

  def __str__(self):
    words = [str(self.op1), 'and', str(self.op2)]
    if not self.parent:
      words += ['?']
    return ' '.join(words)

  def __call__(self, objset, epoch_now):
    return self.op1(objset, epoch_now) and self.op2(objset, epoch_now)

  def get_expected_input(self, should_be, objset, epoch_now):
    if should_be is None:
      should_be = random.random() > 0.5

    if should_be:
      op1_assign, op2_assign = True, True
    else:
      r = random.random()
      # Assume the two operators are independent with P[op=True]=sqrt(0.5)
      # Here show the conditional probability given the output is False
      if r < 0.414:  # 2*sqrt(0.5) - 1
        op1_assign, op2_assign = False, True
      elif r < 0.828:
        op1_assign, op2_assign = True, False
      else:
        op1_assign, op2_assign = False, False

    return op1_assign, op2_assign


def generate_objset(task, n_epoch=30, distractor=True):
  objset = sg.ObjectSet(n_epoch=n_epoch)

  # Guess objects
  for epoch_now in range(n_epoch):
    if distractor:
      objset.add(sg.Object(when='now', deletable=True), epoch_now)
    objset = task.get_expected_input(objset, epoch_now)

  return objset
