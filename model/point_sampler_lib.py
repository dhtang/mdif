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

"""Library defining point samplers."""

from typing import Any, Dict, Sequence

import tensorflow as tf

from google3.vr.perception.volume_compression.mdif.utils import point_sampling


class PointSampler(tf.keras.Model):
  """Samples points in 2D space."""

  def __init__(self, params: Dict[str, Any] = None, name: str = 'PointSampler'):
    """Initialization function.

    Args:
      params: parameters for point sampling.
      name: name of the model.
    """
    super(PointSampler, self).__init__(name=name)
    self.default_params = params

  def call(self,
           spatial_dims: Sequence[int],
           sdf_map: tf.Tensor = None,
           params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Forward method.

    Args:
      spatial_dims: [height, width] of 2D map.
      sdf_map: [height, width, 1] tensor, GT SDF map.
      params: parameters for point sampling. Keyword 'mask' contains
        configuration for the mask of available points. Keyword 'all_pixels'
        contains configuration for sampling all pixels. Keyword 'untruncated'
        contains configuration for point sampling in untruncated region. Keyword
        'regular' contains configuration for strided point sampling. Keyword
        'global' contains configuration for global point sampling.

    Returns:
    """
    height, width = spatial_dims
    output = {}

    # Choose between parameters.
    if params is None:
      if self.default_params is None:
        return output
      else:
        params = self.default_params

    # Get mask for point sampling.
    if 'mask' not in params or not params['mask']['apply']:
      mask_for_point = tf.ones([height, width])
    else:
      mask_mode = params['mask']['mode']
      if mask_mode == 'none':
        mask_for_point = tf.ones([height, width])
      elif mask_mode == 'right_half':
        offset = params['mask']['offset']
        x_right = int(width / 2 + offset[0])
        mask_for_point = tf.concat((tf.ones(
            (height, x_right)), tf.zeros((height, width - x_right))),
                                   axis=-1)
      else:
        raise ValueError('Unknown mask_mode:', mask_mode)
    # Mask for all pixels, tensor with shape [height * width].
    mask_for_point = tf.reshape(mask_for_point, [-1])

    output['mask_for_point'] = mask_for_point

    # Get coodinates and GT SDF of all pixels.
    pixels, pixels_grid = point_sampling.sample2d_all_pixels(
        height, width, normalize=params['normalize_coordinates'])
    if sdf_map is not None:
      pixels_sdf_gt = tf.reshape(sdf_map, (-1, 1))
      # Tensor with shape [height * width, 1].
    else:
      pixels_sdf_gt = None

    # Sample all pixels.
    if params['all_pixels']:
      key = 'all_pixels'
      output['points/' + key] = pixels
      output['points_sdf_gt/' + key] = pixels_sdf_gt

    # Sample points within untruncated regions.
    if params['untruncated']:
      key = 'untruncated/' + params['untruncated/mode']
      output['points/' +
             key], output['points_sdf_gt/' +
                          key] = point_sampling.sample2d_region(
                              params['untruncated/num_point'],
                              pixels,
                              pixels_sdf_gt,
                              region='untruncated',
                              mode=params['untruncated/mode'],
                              truncate=params['untruncated/truncate'],
                              mask=mask_for_point)

    # Sample points regularly in 2D space.
    if params['regular']:
      key = 'regular'
      output['points/' + key], output['points_sdf_gt/' +
                                      key] = point_sampling.sample2d_regular(
                                          params['regular/num_point'],
                                          pixels_grid, sdf_map)

    # Sample points within entire 2D space.
    if params['global']:
      key = 'global/' + params['global/mode']
      output['points/' + key], output['points_sdf_gt/' +
                                      key] = point_sampling.sample2d_region(
                                          params['global/num_point'],
                                          pixels,
                                          pixels_sdf_gt,
                                          region='global',
                                          mode=params['global/mode'],
                                          mask=mask_for_point)

    return output


class PointSampler3D(tf.keras.Model):
  """Samples points in 3D space."""

  def __init__(self,
               params: Dict[str, Any] = None,
               name: str = 'PointSampler3D'):
    """Initialization function.

    Args:
      params: parameters for point sampling.
      name: name of the model.
    """
    super(PointSampler3D, self).__init__(name=name)
    self.default_params = params

  def call(self,
           spatial_dims: Sequence[int],
           point_samples: Dict[str, tf.Tensor],
           params: Dict[str, Any] = None) -> Dict[str, Any]:
    """Forward method.

    Args:
      spatial_dims: [dim_d, height, width] of 3D grid.
      point_samples: contains pre-computed point samples to choose from.
      params: parameters for point sampling. Keyword 'mask' contains
        configuration for the mask of available points. Keyword 'all_pixels'
        contains configuration for sampling all voxels. Keyword 'regular'
        contains configuration for sampling random voxels. Keyword 'global'
        contains configuration for global point sampling. Keyword 'near_surface'
        contains configuration for near surface point sampling. Keyword
        'symmetry' contains configuration for point sampling of symmetry loss.
        Keyword 'consistency' contains configuration for point sampling of
        consistency loss.

    Returns:
    """
    dim_d, height, width = spatial_dims
    output = {}

    # Choose between parameters.
    params_use = self.default_params
    if params is not None:
      params_use.update(params)
    params = params_use
    if params is None:
      return output

    # Get mask for visible grid samples that are used for evaluation.
    if 'mask' not in params or not params['mask']['apply']:
      mask_for_point = tf.ones(spatial_dims)
    else:
      mask_mode = params['mask']['mode']
      if mask_mode == 'right_half':
        offset = params['mask']['offset']
        x_right = int(width / 2 + offset[0])
        mask_for_point = tf.concat((tf.ones(
            (dim_d, height, x_right)), tf.zeros(
                (dim_d, height, width - x_right))),
                                   axis=-1)
      else:
        raise ValueError('Unknown mask_mode:', mask_mode)
    output['mask_for_point'] = tf.reshape(mask_for_point, (-1,))

    # Get coodinates and GT SDF of all voxels.
    pixels = point_samples['grid_samples']
    # Tensor with shape [dim_d, height, width, dim_c].

    # Concat with mask.
    pixels = tf.concat((pixels, mask_for_point[..., None]), axis=-1)

    # Sample voxels.
    if params['all_pixels']:
      sample_voxels(spatial_dims, pixels, point_samples, params, output)
    mask_for_point = output['mask_for_point']

    # Sample regular points.
    if params['regular']:
      sample_regular_points(pixels, params, output)

    # Sample global points.
    if params['global']:
      sample_global_points(point_samples, params, output)

    # Sample near surface points.
    if params['near_surface']:
      sample_near_surface_points(point_samples, params, output)

    # Sample points for symmetry loss.
    if ('symmetry' in params and params['symmetry'] and
        'symmetry_loss' in params and params['symmetry_loss']):
      sample_symmetry_points(point_samples, params, output)

    # Sample points for consistency loss.
    if ('consistency' in params and params['consistency'] and
        'consistency_loss' in params and params['consistency_loss']):
      sample_consistency_points(point_samples, params, output)

    return output


def sample_voxels(spatial_dims: Sequence[int], pixels: tf.Tensor,
                  point_samples: Dict[str, tf.Tensor], params: Dict[str, Any],
                  output: Dict[str, Any]):
  """Samples voxels."""
  dim_d, height, width = spatial_dims

  # Sample voxels in slices.
  if 'all_pixels/mode' in params and params['all_pixels/mode'] == 'slices':
    points = tf.zeros([0, pixels.shape[-1]], dtype=tf.float32)
    for slice_idx_z in params['all_pixels/slices']['slice_idx_z']:
      slice_idx = tf.cast(
          tf.math.round(slice_idx_z * tf.cast(dim_d, tf.float32)), tf.int32) - 1
      points_slice = tf.reshape(pixels[slice_idx, ...], (-1, pixels.shape[-1]))
      points = tf.concat((points, points_slice), axis=0)
    for slice_idx_y in params['all_pixels/slices']['slice_idx_y']:
      slice_idx = tf.cast(
          tf.math.round(slice_idx_y * tf.cast(height, tf.float32)),
          tf.int32) - 1
      points_slice = tf.reshape(pixels[:, slice_idx, ...],
                                (-1, pixels.shape[-1]))
      points = tf.concat((points, points_slice), axis=0)
    for slice_idx_x in params['all_pixels/slices']['slice_idx_x']:
      slice_idx = tf.cast(
          tf.math.round(slice_idx_x * tf.cast(width, tf.float32)), tf.int32) - 1
      points_slice = tf.reshape(pixels[:, :, slice_idx, ...],
                                (-1, pixels.shape[-1]))
      points = tf.concat((points, points_slice), axis=0)

  # Sample all voxels.
  else:
    points = tf.reshape(pixels, (-1, pixels.shape[-1]))

  points_sdf_gt = None
  if point_samples['grid_samples'].shape[-1] > 3:
    points_sdf_gt = points[..., 3:4]  # Tensor with shape [num_point, 1].
  mask_for_point = points[..., -1]  # Tensor with shape [num_point].
  points = points[..., :3]  # Tensor with shape [num_point, 3].

  key = 'all_pixels'
  output['mask_for_point'] = mask_for_point
  output['points/' + key] = points
  output['points_sdf_gt/' + key] = points_sdf_gt


def sample_regular_points(pixels: tf.Tensor, params: Dict[str, Any],
                          output: Dict[str, Any]):
  """Samples regular points."""
  samples = tf.reshape(pixels[..., :4], (-1, 4))

  # Remove points outside [-1, 1].
  samples = point_sampling.filter_points3d(samples, 'remove_outside')

  # Filter out certain points.
  if 'mask' in params and params['mask']['apply']:
    samples = point_sampling.filter_points3d(samples, params['mask'])

  # Randomly sample from the rest.
  key = 'regular'
  output['points/' + key], output['points_sdf_gt/' + key] = (
      point_sampling.sample3d_random(params['regular/num_point'], samples))


def sample_global_points(point_samples: Dict[str, tf.Tensor],
                         params: Dict[str, Any], output: Dict[str, Any]):
  """Samples global points."""
  if 'depth_views' not in params or params['depth_views'] is None:
    samples = point_samples['uniform_samples']
    # Tensor with shape [num_point, 4].
  else:
    # Use points from depth views.
    samples = tf.zeros((0, 4))
    for view_id in params['depth_views']:
      samples = tf.concat(
          [samples, point_samples['uniform_samples_per_camera'][view_id, ...]],
          axis=0)

  # Remove points outside [-1, 1].
  samples = point_sampling.filter_points3d(samples, 'remove_outside')

  # Filter out certain points.
  if 'mask' in params and params['mask']['apply']:
    samples = point_sampling.filter_points3d(samples, params['mask'])

  # Randomly sample from the rest.
  key = 'global/uniform'
  output['points/' + key], output['points_sdf_gt/' +
                                  key] = point_sampling.sample3d_random(
                                      params['global/num_point'], samples)


def sample_near_surface_points(point_samples: Dict[str, tf.Tensor],
                               params: Dict[str, Any], output: Dict[str, Any]):
  """Samples near surface points."""
  if 'depth_views' not in params or params['depth_views'] is None:
    samples = point_samples['near_surface_samples']
    # Tensor with shape [num_point, 4].
  else:
    # Use points from depth views.
    samples = tf.zeros((0, 4))
    for view_id in params['depth_views']:
      samples = tf.concat([
          samples, point_samples['near_surface_samples_per_camera'][view_id,
                                                                    ...]
      ],
                          axis=0)

  # Remove points outside [-1, 1].
  samples = point_sampling.filter_points3d(samples, 'remove_outside')

  # Filter out certain points.
  if 'mask' in params and params['mask']['apply']:
    samples = point_sampling.filter_points3d(samples, params['mask'])

  # Randomly sample from the rest.
  key = 'near_surface/uniform'
  output['points/' + key], output['points_sdf_gt/' +
                                  key] = point_sampling.sample3d_random(
                                      params['near_surface/num_point'], samples)


def sample_symmetry_points(point_samples: Dict[str, tf.Tensor],
                           params: Dict[str, Any], output: Dict[str, Any]):
  """Samples points for symmetry loss."""
  # Get visible points.
  samples_visible = tf.zeros([0, 3])
  if 'global' in params['symmetry/visible/point_source']:
    if 'depth_views' not in params or params['depth_views'] is None:
      samples_visible_i = point_samples['uniform_samples'][..., :3]
      samples_visible = tf.concat([samples_visible, samples_visible_i], axis=0)
    else:
      for view_id in params['depth_views']:
        samples_visible_i = point_samples['uniform_samples_per_camera'][view_id,
                                                                        ..., :3]
        samples_visible = tf.concat([samples_visible, samples_visible_i],
                                    axis=0)
  if 'near_surface' in params['symmetry/visible/point_source']:
    if 'depth_views' not in params or params['depth_views'] is None:
      samples_visible_i = point_samples['near_surface_samples'][..., :3]
      samples_visible = tf.concat([samples_visible, samples_visible_i], axis=0)
    else:
      for view_id in params['depth_views']:
        samples_visible_i = point_samples['near_surface_samples_per_camera'][
            view_id, ..., :3]
        samples_visible = tf.concat([samples_visible, samples_visible_i],
                                    axis=0)
  if 'depth_xyz' in params['symmetry/visible/point_source']:
    if 'depth_views' in params and params['depth_views'] is not None:
      for view_id in params['depth_views']:
        samples_visible_i = point_samples['depth_xyzn_per_camera'][view_id,
                                                                   ..., :3]
        samples_visible = tf.concat([samples_visible, samples_visible_i],
                                    axis=0)

  output_keys = list(output.keys())
  for key in output_keys:
    # Get symmetry points.
    if key.startswith('points/global') or key.startswith('points/near_surface'):
      samples = output[key]
      samples_symmetry = tf.zeros([0, 3])
      if 'reflect_x' in params['symmetry/mode']:
        samples_i = samples * tf.constant([1., 1., -1.])[None, :]
        samples_symmetry = tf.concat([samples_symmetry, samples_i], axis=0)
      if 'reflect_y' in params['symmetry/mode']:
        samples_i = samples * tf.constant([1., -1., 1.])[None, :]
        samples_symmetry = tf.concat([samples_symmetry, samples_i], axis=0)
      if 'reflect_z' in params['symmetry/mode']:
        samples_i = samples * tf.constant([-1., 1., 1.])[None, :]
        samples_symmetry = tf.concat([samples_symmetry, samples_i], axis=0)

      # Compute min k distance with visible points for each sampled point.
      dist = tf.norm(
          samples_symmetry[:, None, :] - samples_visible[None, :, :], axis=-1)
      # Tensor with shape [num_symmetry_point, num_visible_point].

      dist_min_k, _ = tf.math.top_k(
          -dist, k=params['symmetry/point_dist/min_k'], sorted=True)
      # Tensor with shape [num_symmetry_point, k].

      dist_min_k = -dist_min_k

      output['points_symmetry/' + key[7:]] = samples_symmetry
      output['points_symmetry_dist/' + key[7:]] = dist_min_k


def sample_consistency_points(point_samples: Dict[str, tf.Tensor],
                              params: Dict[str, Any], output: Dict[str, Any]):
  """Samples points for consistency loss."""
  # Sample invisible points.
  if params['consistency/invisible/point_source'] == 'global':
    samples = point_samples['uniform_samples']
    # Tensor with shape [num_point, 4]
  else:
    raise NotImplementedError(
        'Can only sample consistency points from uniform samples')
  # Remove points outside [-1, 1].
  samples = point_sampling.filter_points3d(samples, 'remove_outside')
  # Randomly sample from the rest.
  samples_invisible, _ = point_sampling.sample3d_random(
      params['consistency/invisible/num_point'], samples)

  # Get visible points.
  samples_visible = tf.zeros((0, 3))
  if 'global' in params['consistency/visible/point_source']:
    if 'depth_views' not in params or params['depth_views'] is None:
      samples_visible_i = point_samples['uniform_samples'][..., :3]
      samples_visible = tf.concat([samples_visible, samples_visible_i], axis=0)
    else:
      for view_id in params['depth_views']:
        samples_visible_i = point_samples['uniform_samples_per_camera'][view_id,
                                                                        ..., :3]
        samples_visible = tf.concat([samples_visible, samples_visible_i],
                                    axis=0)
  if 'near_surface' in params['consistency/visible/point_source']:
    if 'depth_views' not in params or params['depth_views'] is None:
      samples_visible_i = point_samples['near_surface_samples'][..., :3]
      samples_visible = tf.concat([samples_visible, samples_visible_i], axis=0)
    else:
      for view_id in params['depth_views']:
        samples_visible_i = point_samples['near_surface_samples_per_camera'][
            view_id, ..., :3]
        samples_visible = tf.concat([samples_visible, samples_visible_i],
                                    axis=0)
  if 'depth_xyz' in params['consistency/visible/point_source']:
    if 'depth_views' in params and params['depth_views'] is not None:
      for view_id in params['depth_views']:
        samples_visible_i = point_samples['depth_xyzn_per_camera'][view_id,
                                                                   ..., :3]
        samples_visible = tf.concat([samples_visible, samples_visible_i],
                                    axis=0)

  # Compute min k distance with visible points for each invisible point.
  dist = tf.norm(
      samples_invisible[:, None, :] - samples_visible[None, :, :], axis=-1)
  # Tensor with shape [num_invisible_point, num_visible_point].

  dist_min_k, _ = tf.math.top_k(
      -dist, k=params['consistency/point_dist/min_k'], sorted=True)
  # Tensor with shape [num_invisible_point, k].

  dist_min_k = -dist_min_k

  output['points_consistency'] = samples_invisible
  output['points_consistency_dist'] = dist_min_k
