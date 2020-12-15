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

"""Tests for google3.vr.perception.volume_compression.mdif.utils.chamfer_distance."""

import tensorflow as tf

from google3.vr.perception.volume_compression.mdif.utils import chamfer_distance


class ChamferDistanceTest(tf.test.TestCase):

  def test_compute_chamfer_distance(self):
    batch_dims = (2, 5)
    num_point_a = 15
    num_point_b = 20
    num_dim = 3

    point_set_a = tf.zeros((*batch_dims, num_point_a, num_dim),
                           dtype=tf.float32)
    point_set_b = tf.ones((*batch_dims, num_point_b, num_dim), dtype=tf.float32)

    (chamfer_l2, chamfer_l2_unsquare, min_unsquare_dist_a_to_b,
     min_unsquare_dist_b_to_a) = chamfer_distance.evaluate(
         point_set_a, point_set_b)

    self.assertSequenceEqual(chamfer_l2.shape, batch_dims)
    self.assertSequenceEqual(chamfer_l2_unsquare.shape, batch_dims)
    self.assertSequenceEqual(min_unsquare_dist_a_to_b.shape,
                             (*batch_dims, num_point_a))
    self.assertSequenceEqual(min_unsquare_dist_b_to_a.shape,
                             (*batch_dims, num_point_b))


if __name__ == '__main__':
  tf.test.main()
