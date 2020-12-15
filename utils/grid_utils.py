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

"""Utility functions for manipulating grids and points."""

from typing import Any, Sequence, Text, Tuple

import tensorflow as tf


def subdivide_space(grid_shape: Sequence[int],
                    grid_range_min: Sequence[float],
                    grid_range_max: Sequence[float],
                    mode: Text = 'regular') -> Tuple[Any, Any]:
  """Subdivides space into grids.

  Args:
    grid_shape: [num_dim] list, desired shape of subdivided grids. num_dim is
      the number of dimensions of the space. For 2D space, the list is in the
      order of H, W. For 3D space, the list is in the order of D, H, W.
    grid_range_min: [num_dim] list, minimum coordinate for each dimension.
    grid_range_max: [num_dim] list, maximum coordinate for each dimension.
    mode: str, mode of subdivision, only supports 'regular' for now.

  Returns:
    grid_centers: [*grid_shape, num_dim] tensor, coordinates of grid centers.
    side_length: [num_dim] tensor, side length of each subdivided grid.
  """
  grid_shape = tf.constant(grid_shape, tf.float32)
  grid_range_min = tf.constant(grid_range_min, tf.float32)
  grid_range_max = tf.constant(grid_range_max, tf.float32)
  num_dim = grid_shape.shape[0]

  side_length_total = tf.stack([
      grid_range_max[i_dim] - grid_range_min[i_dim] for i_dim in range(num_dim)
  ],
                               axis=0)

  if mode == 'regular':
    # Create grids within a bounding box with range [0, grid_shape[i_dim]] for
    #   each dimension.
    grid_centers = tf.meshgrid(
        *[tf.range(grid_shape[i_dim]) for i_dim in range(num_dim)],
        indexing='ij')
    grid_centers = tf.stack(grid_centers, axis=-1) + 0.5
    # Tensor with shape [*grid_shape, num_dim], coordinate of each grid center.

    # Scale and shift the grids to desired range
    scale_factor = side_length_total / grid_shape
    shift_factor = grid_range_min
    grid_centers = grid_centers * tf.broadcast_to(
        scale_factor, grid_centers.shape) + tf.broadcast_to(
            shift_factor, grid_centers.shape)
    # Tensor with shape [*grid_shape, num_dim], coordinate of each grid center.

    # Get side length of each subdivided grid.
    side_length = scale_factor
  else:
    raise ValueError('Unknown mode: %s' % mode)

  return grid_centers, side_length


def get_grid_index(points: Any, grid_shape: Sequence[int],
                   grid_range_min: Sequence[int], side_length: Any,
                   grid_mode: Text) -> Any:
  """Gets grid indices of input points.

  Args:
    points: [..., num_dim] tensor, coordinates of input points. num_dim is the
      dimension of each point.
    grid_shape: [num_dim] list, desired shape of subdivided grids.
    grid_range_min: [num_dim] list, minimum coordinate for each dimension.
    side_length: [num_dim] tensor, side length of each subdivided grid.
    grid_mode: str, mode of grid subdivision, only supports 'regular' for now.

  Returns:
    grid_indices: [..., num_dim] tf.int32 tensor, grid indices for input points.
  """
  grid_range_min = tf.constant(grid_range_min, tf.float32)
  num_dim = len(grid_shape)

  if grid_mode == 'regular':
    grid_indices = tf.cast(
        tf.math.floor((points - tf.broadcast_to(grid_range_min, points.shape)) /
                      tf.broadcast_to(side_length, points.shape)),
        dtype=tf.int32)
  else:
    raise ValueError('Unknown grid_mode: %s' % grid_mode)

  grid_indices = tf.stack([
      tf.clip_by_value(grid_indices[..., i_dim], 0, grid_shape[i_dim] - 1)
      for i_dim in range(num_dim)
  ],
                          axis=-1)

  return grid_indices


def normalize_point_coord(points: Any, grid_centers: Any, side_length: Any,
                          grid_indices: Any) -> Any:
  """Normalizes point coordinates within corresponding grid.

  Args:
    points: [..., num_dim] tensor, coordinates of input points. num_dim is the
      dimension of each point.
    grid_centers: [*grid_shape, num_dim] tensor, coordinates of grid centers.
    side_length: [num_dim] tensor, side length of each subdivided grid.
    grid_indices: [..., num_dim] tf.int32 tensor, grid indices for input points.

  Returns:
    points_normalize: [..., num_dim] tensor, normalized coordinates of points.
  """
  # Get corresponding grid centers for each point.
  point_grid_centers = tf.gather_nd(grid_centers, grid_indices)
  # Tensor with shape [..., num_dim].

  # Normalize to [-1, 1].
  points_normalize = (points - point_grid_centers) / side_length[None, :] * 2

  return points_normalize


def ravel_index(indices: Any, dims: Any) -> Any:
  """Flattens indices.

  Args:
    indices: [..., num_dim] tf.int32 tensor, unflattened indices. num_dim is the
      dimension of each unflattened index.
    dims: [num_dim] tf.int32 tensor, shape of each dimension.

  Returns:
    indices: [...] tf.int32 tensor, flattened indices.
  """
  strides = tf.math.cumprod(dims, exclusive=True, reverse=True)
  indices_flat = tf.reduce_sum(
      indices * tf.broadcast_to(strides, indices.shape), axis=-1)
  return indices_flat


def extract_patches(data: Any, grid_shape: Sequence[int]) -> Any:
  """Extracts patches from data based on grid shape.

  Args:
    data: [batch_size, H, W, C] tensor.
    grid_shape: A list with 2 elements, desired shape of subdivided grids.

  Returns:
    data_patches: [batch_size, num_grid, H_patch, W_patch, C] tensor.
  """
  patch_size = (1, int(data.shape[1] / grid_shape[1]),
                int(data.shape[2] / grid_shape[0]), 1)
  data_patches = tf.image.extract_patches(
      images=data,
      sizes=patch_size,
      strides=patch_size,
      rates=(1, 1, 1, 1),
      padding='VALID')

  data_patches = tf.reshape(data_patches,
                            (data.shape[0], grid_shape[1], grid_shape[0],
                             patch_size[1], patch_size[2], data.shape[3]))
  # Permute dimensions to align with grid index.
  data_patches = tf.transpose(data_patches, perm=(0, 2, 1, 3, 4, 5))

  data_patches = tf.reshape(
      data_patches,
      (data.shape[0], -1, patch_size[1], patch_size[2], data.shape[3]))

  return data_patches


def get_heatmap(dims: Any,
                centers: Any,
                sigma: float,
                normalize: bool = True) -> Any:
  """Gets Gaussian heatmaps for each center.

  Args:
    dims: [num_dim] tensor, shape of each dimension, only supports num_dim = 2
      for now.
    centers: [batch_size, num_center, num_dim] tensor, (x, y) center for each
      heatmap.
    sigma: float, sigma of Gaussian.
    normalize: bool, whether center coordinates are already normalized to [-1,
      1].

  Returns:
    heatmaps: [batch_size, *dims, num_center] tensor.
  """
  heatmap_h, heatmap_w = dims
  batch_size = centers.shape[0]
  num_center = centers.shape[1]

  # Get coodinates of all pixels.
  pixels_x, pixels_y = tf.meshgrid(
      tf.range(heatmap_w), tf.range(heatmap_h), indexing='xy')
  pixels_grid = tf.stack((pixels_x, pixels_y), axis=-1)
  pixels_grid = tf.cast(pixels_grid, dtype=tf.float32)
  # Tensor with shape [heatmap_h, heatmap_w, 2], (x, y) coordinate of each pixel
  #  center.

  # Normalize coordinates to [-1, 1].
  if normalize:
    pixels_grid = pixels_grid * 2.0 / tf.convert_to_tensor(
        (heatmap_w - 1, heatmap_h - 1), dtype=tf.float32)[None, None, :] - 1
  # Repeat.
  pixels_grid = tf.tile(pixels_grid[None, ..., None, :],
                        (batch_size, 1, 1, num_center, 1))
  # Tensor with shape [batch_size, heatmap_h, heatmap_w, num_center, 2].

  # Compute heatmaps.
  centers = centers[:, None, None, ...]
  heatmaps = -((pixels_grid[..., 0] - centers[..., 0])**2 +
               (pixels_grid[..., 1] - centers[..., 1])**2) / (2 * sigma**2)
  heatmaps = tf.math.exp(heatmaps)

  return heatmaps
