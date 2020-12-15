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

"""Functions for computing metrics."""

import tensorflow as tf


def point_iou_tf(occ_pred: tf.Tensor,
                 occ_gt: tf.Tensor) -> tf.Tensor:
  """Computes IoU for batched tensor.

  Args:
    occ_pred: [batch_size, ...] tensor, predicted occupancy.
    occ_gt: [batch_size, ...] tensor, GT occupancy.

  Returns:
    iou: [batch_size] tensor, iou for each sample in the batch.
  """
  occ_pred = tf.reshape(occ_pred, (occ_pred.shape[0], -1))
  occ_gt = tf.reshape(occ_gt, (occ_gt.shape[0], -1))

  intersection = tf.cast((occ_pred == 1) & (occ_gt == 1), dtype=tf.float32)
  union = tf.cast((occ_pred == 1) | (occ_gt == 1), dtype=tf.float32)
  iou = tf.reduce_sum(intersection, axis=-1) / (
      tf.reduce_sum(union, axis=-1) + 1e-05)

  return iou
