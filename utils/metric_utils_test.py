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

"""Tests for google3.vr.perception.volume_compression.mdif.utils.metric_utils."""

import tensorflow as tf

from google3.vr.perception.volume_compression.mdif.utils import metric_utils


class MetricUtilsTest(tf.test.TestCase):

  def test_point_iou_tf(self):
    tensor_shape = (2, 16)
    occ_pred = tf.ones(tensor_shape, dtype=tf.int32)
    occ_gt = tf.ones(tensor_shape, dtype=tf.int32)

    iou = metric_utils.point_iou_tf(occ_pred, occ_gt)

    self.assertSequenceEqual(iou.shape, (tensor_shape[0],))


if __name__ == '__main__':
  tf.test.main()
