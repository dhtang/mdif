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

"""Functions for point sampling in 2D/3D space."""

from typing import Any, Tuple, Dict, Sequence, Union

import tensorflow as tf


def sample2d_all_pixels(height: int,
                        width: int,
                        normalize: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
  """Samples all pixels as points.

  Args:
    height: height of 2D map.
    width: width of 2D map.
    normalize: whether normalize coordinates to [-1, 1].

  Returns:
    pixels: [height * width, 2] tensor, (x, y) coordinates of all pixels.
    pixels_grid: [height, width, 2] tensor, (x, y) coordinates of all pixels.
  """
  # Get coodinates of all pixels.
  pixels_x, pixels_y = tf.meshgrid(
      tf.range(width), tf.range(height), indexing='xy')
  pixels_grid = tf.stack((pixels_x, pixels_y), axis=-1)
  pixels_grid = tf.cast(pixels_grid, dtype=tf.float32)
  # Normalize coordinates to [-1, 1].
  if normalize:
    pixels_grid = pixels_grid * 2.0 / tf.convert_to_tensor(
        (width - 1, height - 1), dtype=tf.float32)[None, None, :] - 1
  # Reshape.
  pixels = tf.reshape(pixels_grid, (-1, 2))

  return pixels, pixels_grid


def sample2d_region(num_point: int,
                    pixels: tf.Tensor,
                    pixels_sdf_gt: tf.Tensor = None,
                    region: str = 'global',
                    mode: str = 'uniform',
                    truncate: float = 5,
                    mask: tf.Tensor = None) -> Tuple[tf.Tensor, tf.Tensor]:
  """Samples pixels with chosen region and mode.

  Args:
    num_point: number of point to sample.
    pixels: [height * width, 2] tensor, (x, y) coordinates of all pixels.
    pixels_sdf_gt: [height * width, 1] tensor, GT SDF of all pixels.
    region: sampling region. 'global' for all regions, 'untruncated' for regions
      with untruncated GT SDF.
    mode: sampling mode. 'uniform' for uniform random sampling.
    truncate: SDF truncation value.
    mask: [height * width] tensor, mask for point sampling.

  Returns:
    points: [num_point, 2] tensor, (x, y) coordinates of sampled points.
    points_sdf_gt: [num_point, 1] tensor, GT SDF of sampled points.
  """
  if num_point <= 0:
    return (tf.zeros((0, 2),
                     dtype=tf.float32), tf.zeros((0, 1), dtype=tf.float32))

  if mask is None:
    mask = tf.ones_like(pixels, dtype=tf.bool)[:, 0]
  else:
    mask = tf.cast(mask, dtype=tf.bool)

  # Get indices of candidate points.
  if region == 'global':
    points_mask = mask
  elif region == 'untruncated':
    if pixels_sdf_gt is None:
      raise ValueError(
          'pixels_sdf_gt must not be None when region is `untruncated`.')
    points_mask = (tf.math.less(pixels_sdf_gt, truncate)
                   & tf.math.greater(pixels_sdf_gt, -truncate))[:, 0]
    points_mask = points_mask & mask
  else:
    raise ValueError('Unknown region: %s' % region)
  candidate_indices = tf.where(points_mask)[:, 0]
  # Tensor with shape [num_candidate_point].

  # Get indices of sampled pixels.
  if mode == 'uniform':
    selected_indices = tf.random.categorical(
        tf.math.log(tf.ones_like(candidate_indices, dtype=tf.float32) *
                    0.5)[None, :],
        num_point)[0, :]  # Tensor with shape [num_point].
    points_indices = tf.gather(candidate_indices, selected_indices)
  else:
    raise ValueError('Unknown mode: %s' % mode)

  # Get coordinates and GT SDF of sampled points.
  points = tf.gather(pixels, points_indices, axis=0)
  if pixels_sdf_gt is not None:
    points_sdf_gt = tf.gather(pixels_sdf_gt, points_indices, axis=0)
  else:
    points_sdf_gt = tf.zeros((0, 1), dtype=tf.float32)

  return points, points_sdf_gt


def sample2d_regular(num_point: int,
                     pixels_grid: tf.Tensor,
                     sdf_map: tf.Tensor = None) -> Tuple[tf.Tensor, tf.Tensor]:
  """Samples pixels in a 2D strided grid.

  Args:
    num_point: number of points to sample (actual number will be smaller when
      the specified number cannot be achieved with equal integer stride along
      both axes).
    pixels_grid: [height, width, 2] tensor, (x, y) coordinates of all pixels.
    sdf_map: [height, width, 1] tensor, GT SDF of all pixels.

  Returns:
    points: [num_point, 2] tensor, (x, y) coordinates of sampled points.
    points_sdf_gt: [num_point, 1] tensor, (x, y) GT SDF of sampled points.
  """
  if num_point <= 0:
    return (tf.zeros((0, 2),
                     dtype=tf.float32), tf.zeros((0, 1), dtype=tf.float32))

  height, width = tf.shape(pixels_grid)[0], tf.shape(pixels_grid)[1]

  # Get strided indices in x, y axis.
  step_width = tf.cast(
      tf.math.ceil(tf.math.sqrt((height * width) / num_point)), dtype=tf.int32)
  indices_x, indices_y = tf.meshgrid(
      tf.range(0, width, step_width),
      tf.range(0, height, step_width),
      indexing='xy')
  indices_yx = tf.reshape(
      tf.stack((indices_y[..., None], indices_x[..., None]), axis=-1),
      (-1, 2))  # Tensor with shape [height * width, 2].

  # Get random starting offset.
  offset = tf.concat(
      (tf.math.floor(
          tf.random.uniform(
              (1,), 0, tf.cast(height - indices_y[-1, -1], dtype=tf.float32))),
       tf.math.floor(
           tf.random.uniform(
               (1,), 0, tf.cast(width - indices_x[-1, -1], dtype=tf.float32)))),
      axis=0)[None, :]
  indices_yx = indices_yx + tf.cast(offset, dtype=tf.int32)

  # Get coordinates and GT SDF of sampled points.
  points = tf.gather_nd(pixels_grid, indices_yx)
  if sdf_map is not None:
    points_sdf_gt = tf.gather_nd(sdf_map, indices_yx)
  else:
    points_sdf_gt = tf.zeros((0, 1), dtype=tf.float32)
  return points, points_sdf_gt


def extract_value_at_points2d(points: tf.Tensor,
                              value_map: tf.Tensor,
                              denormalize: bool = True) -> tf.Tensor:
  """Extracts value at given points from value_map (e.g., SDF map).

  Args:
    points: [batch_size, num_point, 2] tensor, (x, y) coordinates of points.
    value_map: [batch_size, height, width, channels] tensor, value map.
    denormalize: whether need to denormalize coordinates to the spatial
      dimensions of value map.

  Returns:
    points_value: [batch_size, num_point, channels] tensor, value at given
      points.
  """
  # Denormalize.
  if denormalize:
    _, height, width, _ = value_map.shape
    points = (points + 1) / 2.0 * tf.constant(
        (width - 1, height - 1), dtype=tf.float32)
    points = tf.cast(points, tf.int32)

  # Gather value.
  points_value = tf.gather_nd(
      tf.transpose(value_map, perm=(0, 2, 1, 3)), points, batch_dims=1)

  return points_value


def sample3d_all_pixels(spatial_dims: Sequence[int],
                        normalize: bool = True) -> Tuple[tf.Tensor, tf.Tensor]:
  """Samples all pixels as points.

  Here in 3D case, pixels are actually voxels, but still named as pixels to
  align with 2D case. Same for all other places where there is a 'pixels'
  substring in the name.

  Args:
    spatial_dims: the spatial dimensions of 3D grid in the order of depth,
      height, width.
    normalize: whether normalize coordinates to [-1, 1].

  Returns:
    pixels: [depth * height * width, 3] tensor, (z, y, x) coordinates of all
      pixels.
    pixels_grid: [depth, height, width, 3] tensor, (z, y, x) coordinates of all
      pixels.
  """
  depth, height, width = spatial_dims

  # Get coodinates of all pixels.
  pixels_z, pixels_y, pixels_x = tf.meshgrid(
      tf.range(depth), tf.range(height), tf.range(width), indexing='ij')
  pixels_grid = tf.stack((pixels_z, pixels_y, pixels_x), axis=-1)
  pixels_grid = tf.cast(pixels_grid, dtype=tf.float32) + 0.5

  # Normalize coordinates to [-1, 1].
  if normalize:
    pixels_grid = pixels_grid * 2.0 / tf.convert_to_tensor(
        (depth, height, width), dtype=tf.float32)[None, None, None, :] - 1
  # Reshape.
  pixels = tf.reshape(pixels_grid, (-1, 3))

  return pixels, pixels_grid


def sample3d_random(num_point: int,
                    point_samples: tf.Tensor) -> Tuple[tf.Tensor, tf.Tensor]:
  """Samples a random subset of points from given point samples.

  Args:
    num_point: number of points to sample.
    point_samples: [num_point_all, 3] tensor or [num_point_all, 4] tensor,
      containing (z, y, x, [sdf]) of pre-computed point samples.

  Returns:
    points: [num_point, 3] tensor, (z, y, x) coordinates of sampled points.
    points_sdf_gt: [num_point, 1] tensor, GT SDF of sampled points.
  """
  if num_point <= 0:
    return (tf.zeros((0, 3),
                     dtype=tf.float32), tf.zeros((0, 1), dtype=tf.float32))

  num_point_all = tf.shape(point_samples)[0]

  # Get indices of sampled pixels.
  points_indices = tf.random.uniform(
      shape=(num_point,), minval=0, maxval=num_point_all, dtype=tf.int32)
  # Tensor with shape [num_point].

  # Get coordinates and GT SDF of sampled points.
  points = tf.gather(point_samples, points_indices, axis=0)
  if points.shape[-1] == 3:
    points_sdf_gt = tf.zeros((0, 1), dtype=tf.float32)
  else:
    points_sdf_gt = points[:, -1:]
    points = points[:, :-1]

  return points, points_sdf_gt


def filter_points3d(point_samples: tf.Tensor,
                    mask_params: Union[str, Dict[str, Any]]) -> tf.Tensor:
  """Samples a subset of points from given point samples.

  Args:
    point_samples: [num_point, 3] tensor or [num_point, 4] tensor, containing
      (z, y, x, [sdf]) of pre-computed point samples. Assume the cooridinates of
      the point samples are already normalized to [-1, 1].
    mask_params: parameters for the filtering process.

  Returns:
    point_samples: [num_rest_point, 3] tensor or [num_rest_point, 4] tensor,
      containing (z, y, x, [sdf]) of the remaining point samples.
  """
  if mask_params == 'remove_outside':
    # Only keep points inside [-1, 1].
    points_mask = ((tf.reduce_min(point_samples[:, :3], axis=-1) > -1) &
                   (tf.reduce_max(point_samples[:, :3], axis=-1) < 1))
  elif mask_params == 'only_keep_truncated':
    # Only keep points with truncated sdf (truncation is set as 5).
    sdf_truncation_value = 5
    points_mask = tf.abs(point_samples[:, 3:4]) >= sdf_truncation_value
  elif isinstance(mask_params, dict) and 'mode' in mask_params:
    # Apply special mask.
    if mask_params['mode'] == 'right_half':
      points_mask = point_samples[:, 2] <= 0
    else:
      raise ValueError('Unknown mode: %s' % mask_params['mode'])
  else:
    raise ValueError('Unknown mask_params:', mask_params)

  points_indices = tf.where(points_mask)[:, 0]
  # Tensor with shape [num_rest_point].

  # Get remaining point samples.
  point_samples = tf.gather(point_samples, points_indices, axis=0)

  return point_samples
