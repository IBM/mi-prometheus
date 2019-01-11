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

"""Store all the constants."""

import itertools
import string
import numpy as np

# RGB, from https://sashat.me/2017/01/11/list-of-20-simple-distinct-colors/
WORD2COLOR = {
    'red': (230, 25, 75),
    'green': (60, 180, 75),
    'blue': (0, 130, 200),
    'yellow': (255, 225, 25),
    'purple': (145, 30, 180),
    'orange': (245, 130, 48),
    'cyan': (70, 240, 240),
    'magenta': (240, 50, 230),
    'lime': (210, 245, 60),
    'pink': (250, 190, 190),
    'teal': (0, 128, 128),
    'lavender': (230, 190, 255),
    'brown': (170, 110, 40),
    'beige': (255, 250, 200),
    'maroon': (128, 0, 0),
    'mint': (170, 255, 195),
    'olive': (128, 128, 0),
    'coral': (255, 215, 180),
    'navy': (0, 0, 128),
    'grey': (128, 128, 128),
    'white': (255, 255, 255)
}

ALLSPACES = ['left', 'right', 'top', 'bottom']
ALLCOLORS = ['red', 'green', 'blue', 'yellow', 'purple', 'orange']
ALLSHAPES = ['circle', 'square', 'cross', 'triangle', 'vbar', 'hbar']

# Comment out the following to use a smaller set of colors and shapes
ALLCOLORS += [
    'cyan', 'magenta', 'lime', 'pink', 'teal', 'lavender', 'brown', 'beige',
    'maroon', 'mint', 'olive', 'coral', 'navy', 'grey', 'white']
ALLSHAPES += list(string.ascii_lowercase)

ALLCOLORSHAPES = [x for x in itertools.product(ALLCOLORS, ALLSHAPES)]
ALLWHENS = ['now', 'latest', 'last1']
ALLWHENS_PROB = [0.5, 0.25, 0.25]

# When the stimuli are invalid for a task
INVALID = 'invalid'

# Allowed vocabulary, the first word is invalid
INPUTVOCABULARY = [
    'invalid',
    '.', ',', '?',
    'object', 'color', 'shape',
    'loc', 'on',
    'if', 'then', 'else',
    'exist',
    'equal', 'and',
    'the', 'of', 'with',
    'point',
] + ALLSPACES + ALLCOLORS + ALLSHAPES + ALLWHENS
# For faster str -> index lookups
INPUTVOCABULARY_DICT = dict([(k, i) for i, k in enumerate(INPUTVOCABULARY)])

INPUTVOCABULARY_SIZE = len(INPUTVOCABULARY)

OUTPUTVOCABULARY = ['true', 'false'] + ALLCOLORS + ALLSHAPES

# Maximum number of words in a sentence
MAXSEQLENGTH = 25

# If use popvec out_type
#def get_prefs(grid_size):
#  """Helper function.
#
#  Args:
#    grid_size : int
#
#  Returns:
#    prefs: numpy 2-D array (n_out_pnt, 2). x and y preferences.
#  """
#  prefs_y, prefs_x = (np.mgrid[0:grid_size, 0:grid_size]) / (grid_size - 1.)
#  prefs_x = prefs_x.flatten().astype('float32')
#  prefs_y = prefs_y.flatten().astype('float32')
#
  # numpy array (Grid_size**2, 2)
#  prefs = (np.array([prefs_x, prefs_y]).astype('float32')).T
#  return prefs

GRID_SIZE = 7
prefs_y, prefs_x = (np.mgrid[0:GRID_SIZE, 0:GRID_SIZE]) / (GRID_SIZE - 1.)
prefs_x = prefs_x.flatten().astype('float32')
prefs_y = prefs_y.flatten().astype('float32')

  # numpy array (Grid_size**2, 2)
PREFS = (np.array([prefs_x, prefs_y]).astype('float32')).T
#PREFS = get_prefs(GRID_SIZE)

