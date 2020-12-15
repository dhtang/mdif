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

"""Tests for google3.vr.perception.volume_compression.mdif.utils.grid_utils."""

import tensorflow as tf

from google3.vr.perception.volume_compression.mdif.utils import grid_utils


class GridUtilsTest(tf.test.TestCase):

  def test_process_grid_and_point(self):
    grid_shape = (2, 2, 2)
    grid_range_min = (-1, -1, -1)
    grid_range_max = (1, 1, 1)
    mode = 'regular'
    num_dim = len(grid_shape)

    grid_centers, side_length = grid_utils.subdivide_space(
        grid_shape, grid_range_min, grid_range_max, mode)

    self.assertSequenceEqual(grid_centers.shape, (*grid_shape, num_dim))
    self.assertSequenceEqual(side_length.shape, (num_dim,))

    num_point = 10
    points = tf.zeros((num_point, num_dim), dtype=tf.float32)
    grid_indices = grid_utils.get_grid_index(
        points, grid_shape, grid_range_min, side_length, grid_mode=mode)
    points_normalize = grid_utils.normalize_point_coord(points, grid_centers,
                                                        side_length,
                                                        grid_indices)

    self.assertSequenceEqual(grid_indices.shape, (num_point, num_dim))
    self.assertSequenceEqual(points_normalize.shape, (num_point, num_dim))

  def test_ravel_index(self):
    num_point = 10
    num_dim = 3
    dims = tf.constant((2, 2, 2), dtype=tf.int32)
    indices = tf.zeros((num_point, num_dim), dtype=tf.int32)

    indices_flat = grid_utils.ravel_index(indices, dims)

    self.assertSequenceEqual(indices_flat.shape, (num_point,))

  def test_extract_patches(self):
    data = tf.zeros((2, 16, 16, 3), dtype=tf.float32)
    grid_shape = (2, 2)
    batch_size, _, _, c = data.shape
    h_patch = 8
    w_patch = 8
    num_grid = grid_shape[0] * grid_shape[1]

    data_patches = grid_utils.extract_patches(data, grid_shape)

    self.assertSequenceEqual(data_patches.shape,
                             (batch_size, num_grid, h_patch, w_patch, c))

  def test_get_heatmap(self):
    batch_size = 2
    num_center = 10
    heatmap_h = 16
    heatmap_w = 16
    num_dim = 2
    dims = tf.constant((heatmap_h, heatmap_w), dtype=tf.int32)
    centers = tf.zeros((batch_size, num_center, num_dim), dtype=tf.float32)
    sigma = 0.1

    heatmaps = grid_utils.get_heatmap(dims, centers, sigma, normalize=True)

    self.assertSequenceEqual(heatmaps.shape,
                             (batch_size, heatmap_h, heatmap_w, num_center))


if __name__ == '__main__':
  tf.test.main()
