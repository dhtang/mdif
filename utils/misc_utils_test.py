#!/usr/bin/python
#
# Copyright 2020 Google LLC
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

"""Tests for google3.vr.perception.volume_compression.mdif.utils.misc_utils."""

import os

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from google3.pyglib import file_util
from google3.vr.perception.volume_compression.mdif.utils import misc_utils


class MiscUtilsTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(('first', False), ('last', True))
  def test_get_image_summary_2d(self, channels_use, normalize):
    summary_key = 'test'
    spatial_dims = (16, 8)
    summary_config = None
    data_mode = 'all'

    raw_data = tf.zeros((2, 16, 8, 5), dtype=tf.float32)
    extra_axis = 0
    image_summary = misc_utils.get_image_summary(summary_key, raw_data,
                                                 channels_use, spatial_dims,
                                                 normalize, summary_config,
                                                 extra_axis, data_mode)
    with self.subTest(name='extra_axis_0'):
      self.assertSequenceEqual(image_summary[summary_key].shape, (2, 16, 8, 3))

    raw_data = tf.zeros((2, 3, 16, 8, 5), dtype=tf.float32)
    extra_axis = 1
    image_summary = misc_utils.get_image_summary(summary_key, raw_data,
                                                 channels_use, spatial_dims,
                                                 normalize, summary_config,
                                                 extra_axis, data_mode)
    with self.subTest(name='extra_axis_1'):
      self.assertSequenceEqual(image_summary[summary_key].shape, (2, 48, 8, 3))

  @parameterized.parameters(('first', False), ('last', True))
  def test_get_image_summary_3d_volume(self, channels_use, normalize):
    summary_key = 'test'
    spatial_dims = (16, 8, 4)
    summary_config = {
        'slice_idx_z': [0.5],
        'slice_idx_y': [0.5],
        'slice_idx_x': [0.5],
    }
    data_mode = 'all'

    raw_data = tf.zeros((2, 16, 8, 4, 5), dtype=tf.float32)
    extra_axis = 0
    image_summary = misc_utils.get_image_summary(summary_key, raw_data,
                                                 channels_use, spatial_dims,
                                                 normalize, summary_config,
                                                 extra_axis, data_mode)
    with self.subTest(name='extra_axis_0_slice_z'):
      self.assertSequenceEqual(
          image_summary[summary_key + '/slice_z/0.5'].shape, (2, 8, 4, 3))
    with self.subTest(name='extra_axis_0_slice_y'):
      self.assertSequenceEqual(
          image_summary[summary_key + '/slice_y/0.5'].shape, (2, 16, 4, 3))
    with self.subTest(name='extra_axis_0_slice_x'):
      self.assertSequenceEqual(
          image_summary[summary_key + '/slice_x/0.5'].shape, (2, 16, 8, 3))

    raw_data = tf.zeros((2, 3, 16, 8, 4, 5), dtype=tf.float32)
    extra_axis = 1
    image_summary = misc_utils.get_image_summary(summary_key, raw_data,
                                                 channels_use, spatial_dims,
                                                 normalize, summary_config,
                                                 extra_axis, data_mode)
    with self.subTest(name='extra_axis_1_slice_z'):
      self.assertSequenceEqual(
          image_summary[summary_key + '/slice_z/0.5'].shape, (2, 24, 4, 3))
    with self.subTest(name='extra_axis_1_slice_y'):
      self.assertSequenceEqual(
          image_summary[summary_key + '/slice_y/0.5'].shape, (2, 48, 4, 3))
    with self.subTest(name='extra_axis_1_slice_x'):
      self.assertSequenceEqual(
          image_summary[summary_key + '/slice_x/0.5'].shape, (2, 48, 8, 3))

  @parameterized.parameters(('first', False), ('last', True))
  def test_get_image_summary_3d_slices(self, channels_use, normalize):
    summary_key = 'test'
    spatial_dims = (16, 8, 4)
    summary_config = {
        'slice_idx_z': [0.5],
        'slice_idx_y': [0.5],
        'slice_idx_x': [0.5],
    }
    data_mode = 'slices'
    raw_data = tf.zeros((2, 8 * 4 + 16 * 4 + 16 * 8, 5), dtype=tf.float32)
    extra_axis = 1

    image_summary = misc_utils.get_image_summary(summary_key, raw_data,
                                                 channels_use, spatial_dims,
                                                 normalize, summary_config,
                                                 extra_axis, data_mode)
    with self.subTest(name='slice_z'):
      self.assertSequenceEqual(
          image_summary[summary_key + '/slice_z/0.5'].shape, (2, 8, 4, 3))
    with self.subTest(name='slice_y'):
      self.assertSequenceEqual(
          image_summary[summary_key + '/slice_y/0.5'].shape, (2, 16, 4, 3))
    with self.subTest(name='slice_x'):
      self.assertSequenceEqual(
          image_summary[summary_key + '/slice_x/0.5'].shape, (2, 16, 8, 3))

  def test_write_read_grd(self):
    with file_util.TemporaryDirectory() as tempdir:
      path = [os.path.join(tempdir, 'temp.grd')]
      volume = np.zeros((1, 8, 8, 8), np.float32)
      world2grid = None
      misc_utils.write_grd_batch(path, volume, world2grid)
      with self.subTest(name='world_grid_none'):
        for i in range(len(path)):
          tx, grd = misc_utils.read_grd(path[i])
          self.assertAllEqual(tx, np.identity(4, dtype=np.float32))
          self.assertAllEqual(grd, volume[i, ...])

      world2grid = np.zeros((1, 4, 4), np.float32)
      misc_utils.write_grd_batch(path, volume, world2grid)
      with self.subTest(name='world_grid_not_none'):
        for i in range(len(path)):
          tx, grd = misc_utils.read_grd(path[i])
          self.assertAllEqual(tx, world2grid[i, ...])
          self.assertAllEqual(grd, volume[i, ...])


if __name__ == '__main__':
  tf.test.main()
