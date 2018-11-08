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

"""cognitive/constants.py"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
import tensorflow as tf
import unittest

from cognitive import constants
from cognitive import stim_generator as sg
from cognitive import train_utils


class ConstantsTest(unittest.TestCase):

  def testDecodeConvertToGrid(self):
    img_size = 112
    prefs = constants.get_prefs(img_size)

    loc_xy = [0.2, 0.8]

    # Target activity given this location
    loc_xy_ = np.array([loc_xy], dtype=np.float32)
    out_pnt = train_utils.convert_to_grid(loc_xy_, prefs)

    # Population vector decoding
    out_pnt = out_pnt[0]
    out_pnt /= out_pnt.sum()
    loc_decoded = np.dot(out_pnt, prefs)

    dist = (loc_decoded[0] - loc_xy[0])**2 + (loc_decoded[1] - loc_xy[1])**2
    self.assertLess(dist, 0.01)

  def testDecodeRender(self):
    img_size = 112
    prefs = constants.get_prefs(img_size)

    loc_xy = sg.Loc([0.2, 0.8])

    objset = sg.ObjectSet(n_epoch=1)
    obj = sg.Object([loc_xy, sg.Shape('square'), sg.Color('blue')],
                    when='now')
    objset.add(obj, epoch_now=0)

    movie = sg.render(objset, img_size=img_size)
    self.assertEqual(list(movie.shape), [1, img_size, img_size, 3])

    movie = movie.sum(axis=-1)  # sum across color
    movie /= movie.sum()
    movie = np.reshape(movie, (1, -1))

    loc_decoded = np.dot(movie, prefs)[0]
    dist = ((loc_decoded[0] - loc_xy.value[0])**2 +
            (loc_decoded[1] - loc_xy.value[1])**2)
    self.assertLess(dist, 0.01)

  def testDecodeRenderPool(self):
    img_size = 112
    grid_size = 7
    prefs = constants.get_prefs(grid_size)

    loc_xy = sg.Loc([0.8, 0.3])

    n_epoch = 5
    objset = sg.ObjectSet(n_epoch=n_epoch)
    obj = sg.Object([loc_xy, sg.Shape('square'), sg.Color('blue')],
                    when='now')
    objset.add(obj, epoch_now=0)

    movie = sg.render(objset, img_size=img_size)
    frame = movie.sum(axis=-1, keepdims=True)

    in_imgs = tf.placeholder('float', [None, img_size, img_size, 1])
    out = tf.contrib.layers.avg_pool2d(in_imgs, 16, 16)

    with tf.Session() as sess:
      out_ = sess.run(out, feed_dict={in_imgs: frame})

    out_ = np.reshape(out_, (n_epoch, -1))
    out_ = (out_.T / (1e-7 + out_.sum(axis=1))).T
    loc_decoded = np.dot(out_, prefs)[0]
    print('Input loc ' + str(loc_xy))
    print('Decoded loc' + str(loc_decoded))
    dist = ((loc_decoded[0] - loc_xy.value[0])**2 +
            (loc_decoded[1] - loc_xy.value[1])**2)
    self.assertLess(dist, 0.01)


if __name__ == '__main__':
  unittest.main()
