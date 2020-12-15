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

"""Tests for google3.vr.perception.volume_compression.mdif.utils.point_sampling."""

from absl.testing import parameterized
import tensorflow as tf

from google3.vr.perception.volume_compression.mdif.utils import point_sampling


class PointSamplingTest(parameterized.TestCase, tf.test.TestCase):

  def test_sample2d_all_pixels(self):
    height = 16
    width = 8
    normalize = True

    pixels, pixels_grid = point_sampling.sample2d_all_pixels(
        height, width, normalize)

    with self.subTest(name='out0_shape'):
      self.assertSequenceEqual(pixels.shape, (height * width, 2))
    with self.subTest(name='out1_shape'):
      self.assertSequenceEqual(pixels_grid.shape, (height, width, 2))

  @parameterized.parameters(('global'), ('untruncated'))
  def test_sample2d_region(self, region):
    pixels = tf.zeros((128, 2), dtype=tf.float32)
    pixels_sdf_gt = tf.zeros((pixels.shape[0], 1), dtype=tf.float32)
    num_point = 9
    mode = 'uniform'

    points, points_sdf_gt = point_sampling.sample2d_region(
        num_point, pixels, pixels_sdf_gt, region, mode, truncate=5, mask=None)

    with self.subTest(name='out0_shape'):
      self.assertSequenceEqual(points.shape, (num_point, 2))
    with self.subTest(name='out1_shape'):
      self.assertSequenceEqual(points_sdf_gt.shape, (num_point, 1))

  def test_sample2d_regular(self):
    pixels_grid = tf.zeros((16, 8, 2), dtype=tf.float32)
    sdf_map = tf.zeros((16, 8, 1), dtype=tf.float32)
    num_point = 9

    points, points_sdf_gt = point_sampling.sample2d_regular(
        num_point, pixels_grid, sdf_map)

    num_point_actual = 8
    with self.subTest(name='out0_shape'):
      self.assertSequenceEqual(points.shape, (num_point_actual, 2))
    with self.subTest(name='out1_shape'):
      self.assertSequenceEqual(points_sdf_gt.shape, (num_point_actual, 1))

  def test_extract_value_at_points2d(self):
    batch_size = 1
    num_point = 9
    channels = 1
    points = tf.zeros((batch_size, num_point, 2), dtype=tf.float32)
    sdf_map = tf.zeros((batch_size, 16, 8, channels), dtype=tf.float32)

    points_value = point_sampling.extract_value_at_points2d(
        points, value_map=sdf_map, denormalize=True)

    self.assertSequenceEqual(points_value.shape,
                             (batch_size, num_point, channels))

  def test_sample3d_all_pixels(self):
    depth = 12
    height = 16
    width = 8
    spatial_dims = (depth, height, width)
    normalize = True

    pixels, pixels_grid = point_sampling.sample3d_all_pixels(
        spatial_dims, normalize)

    with self.subTest(name='out0_shape'):
      self.assertSequenceEqual(pixels.shape, (depth * height * width, 3))
    with self.subTest(name='out1_shape'):
      self.assertSequenceEqual(pixels_grid.shape, (depth, height, width, 3))

  def test_sample3d_random(self):
    point_samples = tf.zeros((1000, 4), dtype=tf.float32)
    num_point = 9

    points, points_sdf_gt = point_sampling.sample3d_random(
        num_point, point_samples)

    with self.subTest(name='out0_shape'):
      self.assertSequenceEqual(points.shape, (num_point, 3))
    with self.subTest(name='out1_shape'):
      self.assertSequenceEqual(points_sdf_gt.shape, (num_point, 1))

  @parameterized.parameters(('remove_outside'), ('only_keep_truncated'), ({
      'mask_params': {
          'mode': 'right_half'
      }
  }))
  def test_filter_points3d(self, mask_params):
    point_samples = tf.concat((tf.zeros(
        (1000, 3), dtype=tf.float32), tf.ones(
            (1000, 1), dtype=tf.float32) * 5.),
                              axis=-1)

    point_samples_new = point_sampling.filter_points3d(point_samples,
                                                       mask_params)

    self.assertSequenceEqual(point_samples_new.shape, (1000, 4))


if __name__ == '__main__':
  tf.test.main()
