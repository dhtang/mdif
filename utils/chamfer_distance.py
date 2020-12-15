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

"""This module implements the chamfer distance."""

from typing import Any, Tuple

import tensorflow as tf


def evaluate(point_set_a: Any,
             point_set_b: Any) -> Tuple[Any, Any, Any, Any]:
  """Computes the Chamfer distance for the given two point sets.

  Note:
    This is a symmetric version of the Chamfer distance, calculated as the sum
    of the average minimum distance from point_set_a to point_set_b and vice
    versa.
    The average minimum distance from one point set to another is calculated as
    the average of the distances between the points in the first set and their
    closest point in the second set, and is thus not symmetrical.

  Note:
    This function returns the exact Chamfer distance and not an approximation.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    point_set_a: A tensor of shape `[A1, ..., An, N, D]`, where N is the number
      of points and D is the dimension of each point.
    point_set_b: A tensor of shape `[A1, ..., An, M, D]`, where M is the number
      of points and D is the dimension of each point.

  Returns:
    chamfer_l2: A tensor of shape `[A1, ..., An]`, L2 chamfer distance.
    chamfer_l2_unsquare: A tensor of shape `[A1, ..., An]`, unsquared L2 chamfer
      distance.
    min_unsquare_dist_a_to_b: A tensor of shape `[A1, ..., An, N]`, minimum
      unsquared distance from point set a to point set b.
    min_unsquare_dist_b_to_a: A tensor of shape `[A1, ..., An, M]`, minimum
      unsquared distance from point set b to point set a.
  """
  # Compute difference of point coordinates between each point in set a and each
  #  point in set b.
  difference = (
      tf.expand_dims(point_set_a, axis=-2) -
      tf.expand_dims(point_set_b, axis=-3))

  # Calculate the square distances between each two points: |ai - bj|^2.
  square_distances = tf.einsum('...i,...i->...', difference, difference)

  min_square_dist_a_to_b = tf.reduce_min(
      input_tensor=square_distances, axis=-1)
  min_square_dist_b_to_a = tf.reduce_min(
      input_tensor=square_distances, axis=-2)

  min_unsquare_dist_a_to_b = tf.math.sqrt(
      min_square_dist_a_to_b)
  min_unsquare_dist_b_to_a = tf.math.sqrt(
      min_square_dist_b_to_a)
  if tf.shape(point_set_b)[0] == 0:
    min_unsquare_dist_a_to_b = tf.ones_like(min_unsquare_dist_a_to_b) * 1e5

  chamfer_l2 = tf.reduce_mean(
      input_tensor=min_square_dist_a_to_b, axis=-1) + tf.reduce_mean(
          input_tensor=min_square_dist_b_to_a, axis=-1)
  chamfer_l2_unsquare = tf.reduce_mean(
      min_unsquare_dist_a_to_b, axis=-1) + tf.reduce_mean(
          min_unsquare_dist_b_to_a, axis=-1)

  return (chamfer_l2, chamfer_l2_unsquare, min_unsquare_dist_a_to_b,
          min_unsquare_dist_b_to_a)
