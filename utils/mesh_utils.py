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

"""Utility functions for contour and mesh."""

import numpy as np
from skimage import measure
import tensorflow as tf


def sdf_to_contour_np(sdf_map: np.ndarray) -> np.ndarray:
  """Computes contour points from SDF map.

  Args:
    sdf_map: [height, width, 1] numpy array, SDF map.

  Returns:
    contour_points: [num_point, 2] numpy array, (y, x) contour points.
  """
  contours = measure.find_contours(sdf_map[..., 0], 0)

  if not contours:
    contour_points = np.zeros((0, 2), dtype=np.float32)
  else:
    contour_points = np.concatenate(contours, axis=0).astype(np.float32)

  return contour_points


def sdf_to_contour(sdf_map: tf.Tensor) -> tf.Tensor:
  """tf.numpy_function wrapper for sdf_to_contour_np.

  Args:
    sdf_map: [height, width, 1] tensor, SDF map.

  Returns:
    contour_points: [num_point, 2] tensor, (y, x) contour points.
  """
  contour_points = tf.numpy_function(sdf_to_contour_np, [sdf_map], tf.float32)
  return contour_points


def contour_to_image(contour_points: tf.Tensor,
                     height: int,
                     width: int,
                     contour_points_error: tf.Tensor = None,
                     error_max: float = 5) -> tf.Tensor:
  """Marks contour points on an image.

  Args:
    contour_points: [num_point, 2] tensor, (y, x) contour points.
    height: height of image.
    width: width of image.
    contour_points_error: [num_point, 1] tensor, error of contour points.
    error_max: maximum error for visualizing.

  Returns:
    contour_image: [height, width, 3] tensor, image with marked contour points.
  """
  if contour_points.shape[0] == 0:
    contour_image = tf.zeros((height, width, 3), tf.float32)
  else:
    # Get indices of valid points.
    valid_point_indices = tf.where((contour_points[:, 0] >= 0)
                                   & (contour_points[:, 0] <= height - 1)
                                   & (contour_points[:, 1] >= 0)
                                   & (contour_points[:, 1] <= width - 1))[:, 0]
    # Tensor with shape [num_valid_point].

    # Filter out invalid points.
    num_valid_point = tf.shape(valid_point_indices)[0]
    contour_points = tf.gather(contour_points, valid_point_indices, axis=0)
    if contour_points_error is not None:
      contour_points_error = tf.gather(
          contour_points_error, valid_point_indices, axis=0)

    contour_points = tf.cast(contour_points, dtype=tf.int32)
    # Tensor with shape [num_valid_point, 2].

    if contour_points_error is None:
      contour_image = tf.scatter_nd(contour_points,
                                    tf.ones((num_valid_point, 3), tf.float32),
                                    (height, width, 3))
    else:
      # Convert error to color.
      contour_points_error = tf.clip_by_value(
          contour_points_error, 0, error_max) / error_max  # Value range [0, 1].
      contour_points_color = tf.concat(
          (contour_points_error, 1 - contour_points_error,
           tf.zeros_like(contour_points_error)),
          axis=-1)  # Tensor with shape [num_valid_point, 3].

      contour_image = tf.scatter_nd(contour_points, contour_points_color,
                                    (height, width, 3))

      # Compensate for multiple updates at same element.
      update_count_image = tf.ones((height, width, 1), dtype=tf.float32)
      update_count_image = tf.tensor_scatter_nd_update(
          update_count_image, contour_points,
          tf.ones((num_valid_point, 1), dtype=tf.float32))
      contour_image = contour_image / update_count_image

  return contour_image
