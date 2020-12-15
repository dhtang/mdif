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

"""Tests for google3.vr.perception.volume_compression.mdif.utils.mesh_utils."""

import tensorflow as tf

from google3.vr.perception.volume_compression.mdif.utils import mesh_utils


class MeshUtilsTest(tf.test.TestCase):

  def test_sdf_to_contour_dummy(self):
    sdf_map = tf.ones((5, 3, 1), dtype=tf.float32)
    contour_points = mesh_utils.sdf_to_contour(sdf_map)

    self.assertSequenceEqual(contour_points.shape, (0, 2))

  def test_sdf_to_contour(self):
    sdf_map = tf.concat((tf.ones(
        (2, 3, 1), dtype=tf.float32), tf.zeros(
            (1, 3, 1), dtype=tf.float32), tf.ones(
                (2, 3, 1), dtype=tf.float32) * -1.),
                        axis=0)
    contour_points = mesh_utils.sdf_to_contour(sdf_map)

    self.assertSequenceEqual(contour_points.shape, (3, 2))

  def test_contour_to_image(self):
    height = 16
    width = 8
    num_point = 32
    contour_points = tf.concat(
        (tf.random.uniform((num_point, 1), 0, height - 1, tf.float32),
         tf.random.uniform((num_point, 1), 0, width - 1, tf.float32)),
        axis=-1)
    contour_points_error = tf.random.uniform((num_point, 1), 0, 10, tf.float32)
    error_max = 5

    contour_image = mesh_utils.contour_to_image(contour_points, height, width,
                                                contour_points_error, error_max)

    self.assertSequenceEqual(contour_image.shape, (height, width, 3))


if __name__ == '__main__':
  tf.test.main()
