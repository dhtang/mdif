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

"""Network library defining complete pipelines."""

import math
from typing import Any, Dict, Sequence, Tuple, Union

import gin.tf
import tensorflow as tf
import tensorflow_addons.image as tfa_image
import tensorflow_graphics.math.interpolation.trilinear as tfg_trilinear

from google3.vr.perception.volume_compression.mdif.model import network_autoencoder
from google3.vr.perception.volume_compression.mdif.model import network_multilevel
from google3.vr.perception.volume_compression.mdif.model import network_utils
from google3.vr.perception.volume_compression.mdif.model import point_sampler_lib
from google3.vr.perception.volume_compression.mdif.utils import chamfer_distance
from google3.vr.perception.volume_compression.mdif.utils import grid_utils
from google3.vr.perception.volume_compression.mdif.utils import mesh_utils
from google3.vr.perception.volume_compression.mdif.utils import metric_utils
from google3.vr.perception.volume_compression.mdif.utils import misc_utils
from google3.vr.perception.volume_compression.mdif.utils import point_sampling


@gin.configurable
class MultiresDeepImplicitFunction(tf.keras.Model):
  """Complete pipeline for multi-level deep implicit function."""

  def __init__(self,
               general_params: Dict[str, Any],
               loss_params: Dict[str, Any],
               input_encoder_params: Sequence[Dict[str, Any]],
               feature_to_code_net_params: Dict[str, Any],
               decoder_params: Dict[str, Any],
               train_sampling_params: Dict[str, Any],
               eval_sampling_params: Dict[str, Any],
               latent_optim_sampling_params: Dict[str, Any],
               name: str = 'MultiresDeepImplicitFunction'):
    """Initialization function.

    Args:
      general_params: general parameters for the model.
      loss_params: loss parameters for the model.
      input_encoder_params: parameters for input encoder.
      feature_to_code_net_params: parameters for feature to code net.
      decoder_params: parameters for implicit decoder.
      train_sampling_params: parameters for point sampling during training.
      eval_sampling_params: parameters for point sampling during evaluation.
      latent_optim_sampling_params: parameters for point sampling during latent
        optimization.
      name: name of the model.
    """
    super(MultiresDeepImplicitFunction, self).__init__(name=name)
    self.loss_params = loss_params

    self._debug_mode = general_params['debug_mode']
    self._mode = general_params['mode']
    self._max_point_per_chunk = general_params['max_point_per_chunk']
    self._num_point_dim = general_params['num_point_dim']
    self.num_level = general_params['num_level']

    if 'sdf_scale' in general_params:
      self._sdf_scale = float(general_params['sdf_scale'])
    else:
      self._sdf_scale = 1.0

    self._pipeline_mode = general_params['pipeline_mode']

    self._encoder_mode = general_params['encoder_mode']
    if self._encoder_mode not in ['input_enc+f2c']:
      raise ValueError('Unknown encoder_mode: %s' % self._encoder_mode)

    self._code_for_point_mode = general_params['code_for_point_mode']
    if self._code_for_point_mode not in ['bind', 'interpolate']:
      raise ValueError('Unknown code_for_point_mode: %s' %
                       self._code_for_point_mode)

    self._input_config_unified = general_params['input_config_unified']
    if 'decoder_input_config' in general_params:
      self._decoder_input_config = general_params['decoder_input_config']
    else:
      default_config = {
          'data': general_params['decoder_input_mode'],
          'empty_vars': []
      }
      self._decoder_input_config = [
          dict(default_config) for _ in range(self.num_level)
      ]
    self._label_config_unified = general_params['label_config_unified']
    self._label_config = general_params['label_config']
    self.summary_config = general_params['summary_config']
    if 'eval_data_mode' in general_params:
      self._eval_data_mode = general_params['eval_data_mode']
    else:
      self._eval_data_mode = 'all'

    # Alter parameters based on mode.
    if self._mode in ['multi_level', 'fully_multi_level']:
      self._grid_range_min = general_params['grid_range_min']
      self._grid_range_max = general_params['grid_range_max']
      self._grid_mode = general_params['grid_mode']
      self._grid_shape = general_params['grid_shape']
      assert len(self._grid_shape) >= self.num_level

      # Compute grid subdivision for each level.
      self._num_grid = []
      self._grid_centers = []
      self._grid_side_length = []
      for i in range(self.num_level):
        self._num_grid.append(
            int(tf.math.reduce_prod(self._grid_shape[i]).numpy()))
        grid_centers, grid_side_length = grid_utils.subdivide_space(
            self._grid_shape[i], self._grid_range_min, self._grid_range_max,
            self._grid_mode)
        self._grid_centers.append(grid_centers)
        self._grid_side_length.append(grid_side_length)
    else:
      raise ValueError('Unknown mode: %s' % self._mode)

    # Update parameters for decoder.
    decoder_params['num_level'] = self.num_level

    # Update params for point sampling.
    train_sampling_params['num_point_dim'] = self._num_point_dim
    eval_sampling_params['num_point_dim'] = self._num_point_dim
    latent_optim_sampling_params['num_point_dim'] = self._num_point_dim
    if self._eval_data_mode == 'slices':
      eval_sampling_params['all_pixels/mode'] = 'slices'
      eval_sampling_params['all_pixels/slices'] = {
          'slice_idx_z': self.summary_config['slice_idx_z'],
          'slice_idx_y': self.summary_config['slice_idx_y'],
          'slice_idx_x': self.summary_config['slice_idx_x'],
      }

    # Set up encoders.
    if self._encoder_mode == 'input_enc+f2c':
      assert self._mode == 'fully_multi_level'
      self._input_encoder = [
          network_autoencoder.EncoderTemplate(
              **(input_encoder_params[0]), name='InputEncoder_0')
      ]
      self._feature_to_code_net = network_multilevel.FeatureToCodeNet(
          **feature_to_code_net_params, name='f2c_net_0')

    # Parameters for auto-decoding or latent optimization.
    self._num_train_data = general_params['num_train_data']
    self._num_test_data = general_params['num_test_data']
    self._codes_init_std = general_params['codes_init_std']
    if 'latent_optim_target' in general_params:
      self._latent_optim_target = general_params['latent_optim_target']
    else:
      self._latent_optim_target = 'root_feature'
    if 'codes_init_from_encoder' in general_params:
      self.codes_init_from_encoder = general_params['codes_init_from_encoder']
    else:
      self.codes_init_from_encoder = False
    if 'codes_init_from_ref' in general_params:
      self.codes_init_from_ref = general_params['codes_init_from_ref']
    else:
      self.codes_init_from_ref = False
    if 'code_grid_enc_shape' in general_params:
      self.code_grid_enc_shape = general_params['code_grid_enc_shape']
    else:
      self.code_grid_enc_shape = [[1, 1, 512], [4, 4, 32], [16, 16, 8]]

    # Initialize latent codes.
    self._init_latent_codes()

    # Set up decoder.
    self._decoder = network_multilevel.MultiLevelImplicitDecoder(
        **decoder_params)

    # Update feature_to_code_net.
    if 'dec_only_apply_mask' in general_params:
      self._feature_to_code_net.dec_only_apply_mask = general_params[
          'dec_only_apply_mask']

    # Update config for masking layers.
    if 'masking_layer_update' in general_params:
      masking_layer_update = general_params['masking_layer_update']
    else:
      masking_layer_update = None
    if masking_layer_update is not None:
      for block_i in self._feature_to_code_net._blocks:
        for subnet_i in block_i:
          if isinstance(subnet_i, network_utils.MaskingLayer):
            subnet_i.update_config(masking_layer_update)

    # Set up point samplers.
    self._eval_sampling_params_init = eval_sampling_params
    if self._num_point_dim == 2:
      self._train_point_sampler = point_sampler_lib.PointSampler(
          train_sampling_params)
      self._eval_point_sampler = point_sampler_lib.PointSampler(
          eval_sampling_params)
      self._latent_optim_point_sampler = point_sampler_lib.PointSampler(
          latent_optim_sampling_params)
    elif self._num_point_dim == 3:
      self._train_point_sampler = point_sampler_lib.PointSampler3D(
          train_sampling_params)
      self._eval_point_sampler = point_sampler_lib.PointSampler3D(
          eval_sampling_params)
      self._latent_optim_point_sampler = point_sampler_lib.PointSampler3D(
          latent_optim_sampling_params)

    # Whether need to evaluate all pixels every iteration.
    self._do_eval_every_iter = False

  def _init_latent_codes(self):
    """Initialize latent codes."""
    self.codes_train_data = []
    self.codes_test_data = []
    if (self._mode == 'fully_multi_level' and
        self._encoder_mode == 'input_enc+f2c'):
      if self._latent_optim_target == 'code_grid_enc':
        for i in range(self.num_level):
          output_dim = tf.reduce_prod(
              tf.constant(self.code_grid_enc_shape[i], dtype=tf.int32))
          self.codes_train_data.append(
              tf.keras.layers.Embedding(
                  self._num_train_data,
                  output_dim,
                  embeddings_initializer='uniform',
                  name='latent_codes/train/level' + str(i)))
          self.codes_test_data.append(
              tf.keras.layers.Embedding(
                  self._num_test_data,
                  output_dim,
                  embeddings_initializer='uniform',
                  name='latent_codes/test/level' + str(i)))
      else:
        raise ValueError('Unknown latent_optim_target: %s' %
                         self._latent_optim_target)

  def reset_latent_codes(self,
                         latent_code_type: str = 'test',
                         reset_data: Dict[str, tf.Tensor] = None):
    """Resets latent codes."""
    if latent_code_type == 'train':
      codes_use = self.codes_train_data
    elif latent_code_type == 'test':
      codes_use = self.codes_test_data

    init_data = {}
    for i in range(len(codes_use)):
      layer = codes_use[i]

      # Forward the embedding layer to ensure it is initialized.
      layer(0)

      for key, initializer in layer.__dict__.items():
        if 'initializer' not in key:
          continue
        var_key = key.replace('_initializer', '')
        if var_key not in layer.__dict__:
          return
        var = getattr(layer, var_key)
        print('Re-initialize', var.name, 'shape', var.shape)
        if self._latent_optim_target == 'code_grid_enc':
          if reset_data is None:
            codes_init = initializer(var.shape, var.dtype) * \
                self._codes_init_std[i]
          else:
            codes_init = reset_data['code_grid_enc/level' + str(i)]
            codes_init = tf.reshape(codes_init, [codes_init.shape[0], -1])
          var.assign(codes_init)
          codes_init = tf.reshape(codes_init, [1, *self.code_grid_enc_shape[i]])
          init_data['code_grid_enc_init/level' + str(i)] = codes_init
        else:
          raise ValueError('Unknown latent_optim_target: %s' %
                           self._latent_optim_target)

    return init_data

  def _gather_latent_codes(self,
                           latent_code_type: str = 'train'
                          ) -> Sequence[tf.Tensor]:
    """Gather latent codes for auto-decoding or latent optimization.

    Args:
      latent_code_type: type of latent code, one of ['train', 'test'].

    Returns:
      codes_each_level: with shape [batch_size, code_dim_c], latent codes at
        each level.
    """
    if latent_code_type == 'train':
      codes_use = self.codes_train_data
    elif latent_code_type == 'test':
      codes_use = self.codes_test_data

    if (self._mode == 'fully_multi_level' and
        self._encoder_mode == 'input_enc+f2c'):
      if self._latent_optim_target == 'code_grid_enc':
        codes_each_level = []
        for i in range(len(codes_use)):
          codes_each_level.append(
              tf.reshape(codes_use[i](0), [1, *self.code_grid_enc_shape[i]]))

    return codes_each_level

  def _interpolate_codes_at_points(
      self, code_grid: tf.Tensor, points: tf.Tensor,
      level: int) -> Tuple[tf.Tensor, tf.Tensor, Dict[str, tf.Tensor]]:
    """Interpolates codes at given points from code_grid.

    Args:
      code_grid: [batch_size, [code_grid_d], code_grid_h, code_grid_w,
        code_grid_c] tensor, latent code grid.
      points: [batch_size, num_point, num_point_dim] tensor, point coordinates.
      level: selected level to be evaluated.

    Returns:
      codes_for_points: [batch_size, num_point, code_grid_c] tensor, latent
        codes for each point.
      points_normalize: [batch_size, num_point, num_point_dim] tensor,
        normalized point coordinates within corresponding grid.
      debug_data: contains data for debugging.
    """
    if self._num_point_dim == 2:
      _, code_grid_h, code_grid_w, _ = code_grid.shape
      _, num_point, _ = points.shape

      # Get grid indices of each point.
      grid_indices = grid_utils.get_grid_index(points, self._grid_shape[level],
                                               self._grid_range_min,
                                               self._grid_side_length[level],
                                               self._grid_mode)
      # Tensor with shape [batch_size, num_point, num_point_dim].

      # Normalize point coordinates within each grid.
      points_normalize = grid_utils.normalize_point_coord(
          points, self._grid_centers[level], self._grid_side_length[level],
          grid_indices)

      # Renormalize point coordinates to the spatial resolution of code grid.
      points_renormalize = grid_utils.normalize_point_coord(
          points, self._grid_centers[0], self._grid_side_length[0],
          tf.zeros_like(grid_indices))
      points_renormalize = points_renormalize * 0.5 + 0.5  # Map to [0, 1].
      points_renormalize = points_renormalize * tf.constant(
          [code_grid_h - 1, code_grid_w - 1], dtype=tf.float32)[None, None, :]
      # Tensor with shape [batch_size, num_point, num_point_dim].

      # Interpolate codes from code_grid.
      if code_grid_h == 1 and code_grid_w == 1:
        codes_for_points = tf.tile(code_grid[:, 0, ...], [1, num_point, 1])
      else:
        codes_for_points = tfa_image.interpolate_bilinear(
            code_grid, points_renormalize, indexing='xy')
      # Tensor with shape [batch_size, num_point, code_grid_c].

    elif self._num_point_dim == 3:
      _, code_grid_d, code_grid_h, code_grid_w, _ = code_grid.shape
      num_point = tf.shape(points)[1]

      if self._grid_shape[level] != [1, 1, 1]:
        raise NotImplementedError('Grid shape must be [1, 1, 1].')
      else:
        points_normalize = points

      # Renormalize point coordinates to the spatial resolution of code grid,
      #  after which points should be in the range of [-1, 1].
      points_renormalize = points * 0.5 + 0.5  # Map to [0, 1].
      points_renormalize = points_renormalize * tf.constant(
          [code_grid_d - 1, code_grid_h - 1, code_grid_w - 1],
          dtype=tf.float32)[None, None, :]

      # Interpolate codes from code_grid. Output tensor is with shape
      #  [batch_size, num_point, code_grid_c].
      if code_grid_d == 1 and code_grid_h == 1 and code_grid_w == 1:
        codes_for_points = tf.tile(code_grid[:, 0, 0, ...], [1, num_point, 1])
      else:
        # The dimensions of code_grid should be [batch_size, code_grid_d,
        #  code_grid_h, code_grid_w, code_grid_c]. The dimensions of
        #  points_renormalize should be [batch_size, num_point, num_point_dim],
        #  and the order of coordinates should be (z, y, x).
        codes_for_points = tfg_trilinear.interpolate(code_grid,
                                                     points_renormalize)

    debug_data = {}
    debug_data['points_normalize'] = points_normalize
    debug_data['points_renormalize'] = points_renormalize
    debug_data['latent_codes'] = tf.concat(
        [codes_for_points[..., :3], codes_for_points[..., -3:]], axis=-1)

    return codes_for_points, points_normalize, debug_data

  def _sample_points(self,
                     spatial_dims: Sequence[int],
                     gt_data_for_label: Dict[str, tf.Tensor],
                     sampler: Union[point_sampler_lib.PointSampler,
                                    point_sampler_lib.PointSampler3D],
                     params: Dict[str, Any] = None) -> Dict[str, tf.Tensor]:
    """A wrapper function for point sampling.

    Args:
      spatial_dims: spatial dimensions of SDF grid.
      gt_data_for_label: contains GT data for SDF.
      sampler: point sampler to be used.
      params: override for point sampling parameters.

    Returns:
      points_data: contains sampled points and sampling mask.
    """
    # Sample points for each data sample in a batch.
    output = []
    if self._num_point_dim == 2:
      sdf_map = gt_data_for_label['sdf_map']
      batch_size = sdf_map.shape[0]
      for i_sample in range(batch_size):
        output.append(sampler(spatial_dims, sdf_map[i_sample, ...]))
    elif self._num_point_dim == 3:
      batch_size = gt_data_for_label['grid_samples'].shape[0]
      for i_sample in range(batch_size):
        gt_data_for_label_i = {}
        for key in gt_data_for_label:
          gt_data_for_label_i[key] = gt_data_for_label[key][i_sample, ...]
        output.append(sampler(spatial_dims, gt_data_for_label_i, params))

    # Batch outputs and merge into one dict
    points_data = {}
    for key in output[0].keys():
      if key.startswith('points/'):
        points_data[key] = tf.stack([output[i][key] for i in range(batch_size)],
                                    axis=0)
        # Tensor with shape [batch_size, num_point, num_point_dim].

        points_data['points_sdf_gt/' + key[7:]] = tf.stack(
            [output[i]['points_sdf_gt/' + key[7:]] for i in range(batch_size)],
            axis=0)  # Tensor with shape [batch_size, num_point, 1].

      elif key.startswith('points_symmetry/'):
        points_data[key] = tf.stack([output[i][key] for i in range(batch_size)],
                                    axis=0)
        # Tensor with shape [batch_size, num_point, num_point_dim].

        key_dist = key.replace('points_symmetry/', 'points_symmetry_dist/')
        points_data[key_dist] = tf.stack(
            [output[i][key_dist] for i in range(batch_size)],
            axis=0)  # Tensor with shape [batch_size, num_point, 1].

      elif key == 'points_consistency':
        points_data[key] = tf.stack([output[i][key] for i in range(batch_size)],
                                    axis=0)
        # Tensor with shape [batch_size, num_point, num_point_dim].

        points_data[key + '_dist'] = tf.stack(
            [output[i][key + '_dist'] for i in range(batch_size)],
            axis=0)  # Tensor with shape [batch_size, num_point, 1].

      elif key == 'mask_for_point':
        points_data[key] = tf.stack([output[i][key] for i in range(batch_size)],
                                    axis=0)
        # Tensor with shape [batch_size, dim_h * dim_w] or [batch_size,
        #  dim_d * dim_h * dim_w]

    return points_data

  def _preprocess_data(
      self, data_batch: Dict[str, Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Prepares network input and GT.

    Args:
      data_batch: a dictionary containing all relevant data.

    Returns:
      input_data: network input data.
      gt_data: network GT data.
    """
    # Get input data, with shape [batch_size, dim_h, dim_w, dim_c].
    input_data = data_batch['sdf_map']

    gt_data = input_data

    return input_data, gt_data

  def _preprocess_data_3d(
      self, data_batch: Dict[str, Any]) -> Tuple[tf.Tensor, tf.Tensor]:
    """Prepares network input and GT.

    Args:
      data_batch: a dictionary containing all relevant data.

    Returns:
      input_data: network input data.
      gt_data: network GT data.
    """
    world2grid = data_batch['world2grid']
    uniform_samples = data_batch['uniform_samples']
    near_surface_samples = data_batch['near_surface_samples']
    if 'uniform_samples_per_camera' in data_batch:
      uniform_samples_per_camera = data_batch['uniform_samples_per_camera']
    if 'near_surface_samples_per_camera' in data_batch:
      near_surface_samples_per_camera = data_batch[
          'near_surface_samples_per_camera']
    if 'depth_xyzn_per_camera' in data_batch:
      depth_xyzn_per_camera = data_batch['depth_xyzn_per_camera']

    batch_size = data_batch['near_surface_samples'].shape[0]
    spatial_dims = data_batch['grid_samples'].shape[1:4]

    # Assume grid size is the same in all dimensions.
    grid_size = spatial_dims[0]

    # Generate normalized [-1, 1] coordinates for grid samples.
    _, pixels_grid = point_sampling.sample3d_all_pixels(
        spatial_dims, normalize=True)
    pixels_grid = tf.tile(pixels_grid[None, ...], [batch_size, 1, 1, 1, 1])
    data_batch['grid_samples'] = tf.concat(
        [pixels_grid, data_batch['grid_samples']], axis=-1)
    # Tensor with shape [batch_size, dim_d, dim_h, dim_w, 4], (z, y, x).

    # Transform to grid space [0, 127] and map grid space [-0.5, 127.5] to
    #  [-1, 1].
    uniform_samples = tf.linalg.matmul(
        world2grid[:, :3, :],
        tf.transpose(
            tf.concat([
                uniform_samples[..., :3],
                tf.ones(
                    [batch_size, tf.shape(uniform_samples)[1], 1],
                    dtype=tf.float32)
            ],
                      axis=-1), [0, 2, 1]))
    # Tensor with shape [batch_size, 3, num_point].

    uniform_samples = (uniform_samples + 0.5) * 2.0 / grid_size - 1
    uniform_samples = uniform_samples[:, ::-1, :]  # Convert to (z, y, x).
    uniform_samples = tf.concat([
        tf.transpose(uniform_samples, [0, 2, 1]),
        data_batch['uniform_samples'][..., 3:]
    ],
                                axis=-1)
    data_batch['uniform_samples'] = uniform_samples

    near_surface_samples = tf.linalg.matmul(
        world2grid[:, :3, :],
        tf.transpose(
            tf.concat([
                near_surface_samples[..., :3],
                tf.ones([batch_size,
                         tf.shape(near_surface_samples)[1], 1],
                        dtype=tf.float32)
            ],
                      axis=-1), [0, 2, 1]))
    near_surface_samples = (near_surface_samples + 0.5) * 2.0 / grid_size - 1
    near_surface_samples = near_surface_samples[:, ::-1, :]
    near_surface_samples = tf.concat([
        tf.transpose(near_surface_samples, [0, 2, 1]),
        data_batch['near_surface_samples'][..., 3:]
    ],
                                     axis=-1)
    data_batch['near_surface_samples'] = near_surface_samples

    if 'uniform_samples_per_camera' in data_batch:
      num_view = tf.shape(uniform_samples_per_camera)[1]
      num_point_per_view = tf.shape(uniform_samples_per_camera)[2]
      num_channel = tf.shape(uniform_samples_per_camera)[3]
      uniform_samples_per_camera = tf.reshape(uniform_samples_per_camera,
                                              [batch_size, -1, num_channel])
      uniform_samples_per_camera = tf.linalg.matmul(
          world2grid[:, :3, :],
          tf.transpose(
              tf.concat([
                  uniform_samples_per_camera[..., :3],
                  tf.ones(
                      [batch_size,
                       tf.shape(uniform_samples_per_camera)[1], 1],
                      dtype=tf.float32)
              ],
                        axis=-1), [0, 2, 1]))
      uniform_samples_per_camera = (uniform_samples_per_camera +
                                    0.5) * 2.0 / grid_size - 1
      uniform_samples_per_camera = uniform_samples_per_camera[:, ::-1, :]
      uniform_samples_per_camera = tf.reshape(
          tf.transpose(uniform_samples_per_camera, [0, 2, 1]),
          [batch_size, num_view, num_point_per_view, -1])
      # Tensor with shape [batch_size, num_view, num_point_per_view, 3].

      uniform_samples_per_camera = tf.concat([
          uniform_samples_per_camera,
          data_batch['uniform_samples_per_camera'][..., 3:]
      ],
                                             axis=-1)
      data_batch['uniform_samples_per_camera'] = uniform_samples_per_camera

    if 'near_surface_samples_per_camera' in data_batch:
      num_view = tf.shape(near_surface_samples_per_camera)[1]
      num_point_per_view = tf.shape(near_surface_samples_per_camera)[2]
      num_channel = tf.shape(near_surface_samples_per_camera)[3]
      near_surface_samples_per_camera = tf.reshape(
          near_surface_samples_per_camera, [batch_size, -1, num_channel])
      near_surface_samples_per_camera = tf.linalg.matmul(
          world2grid[:, :3, :],
          tf.transpose(
              tf.concat([
                  near_surface_samples_per_camera[..., :3],
                  tf.ones([
                      batch_size,
                      tf.shape(near_surface_samples_per_camera)[1], 1
                  ],
                          dtype=tf.float32)
              ],
                        axis=-1), [0, 2, 1]))
      near_surface_samples_per_camera = (near_surface_samples_per_camera +
                                         0.5) * 2.0 / grid_size - 1
      near_surface_samples_per_camera = near_surface_samples_per_camera[:, ::
                                                                        -1, :]
      near_surface_samples_per_camera = tf.reshape(
          tf.transpose(near_surface_samples_per_camera, [0, 2, 1]),
          [batch_size, num_view, num_point_per_view, -1])
      # Tensor with shape [batch_size, num_view, num_point_per_view, 3].

      near_surface_samples_per_camera = tf.concat([
          near_surface_samples_per_camera,
          data_batch['near_surface_samples_per_camera'][..., 3:]
      ],
                                                  axis=-1)
      data_batch[
          'near_surface_samples_per_camera'] = near_surface_samples_per_camera

    if 'depth_xyzn_per_camera' in data_batch:
      num_view = tf.shape(depth_xyzn_per_camera)[1]
      num_point_per_view = tf.shape(depth_xyzn_per_camera)[2]
      num_channel = tf.shape(depth_xyzn_per_camera)[3]
      depth_xyzn_per_camera = tf.reshape(depth_xyzn_per_camera,
                                         [batch_size, -1, num_channel])
      depth_xyzn_per_camera = tf.linalg.matmul(
          world2grid[:, :3, :],
          tf.transpose(
              tf.concat([
                  depth_xyzn_per_camera[..., :3],
                  tf.ones([batch_size,
                           tf.shape(depth_xyzn_per_camera)[1], 1],
                          dtype=tf.float32)
              ],
                        axis=-1), [0, 2, 1]))
      depth_xyzn_per_camera = (depth_xyzn_per_camera +
                               0.5) * 2.0 / grid_size - 1
      depth_xyzn_per_camera = depth_xyzn_per_camera[:, ::-1, :]
      depth_xyzn_per_camera = tf.reshape(
          tf.transpose(depth_xyzn_per_camera, [0, 2, 1]),
          [batch_size, num_view, num_point_per_view, -1])
      # Tensor with shape [batch_size, num_view, num_point_per_view, 3].

      depth_xyzn_per_camera = tf.concat(
          [depth_xyzn_per_camera, data_batch['depth_xyzn_per_camera'][..., 3:]],
          axis=-1)
      data_batch['depth_xyzn_per_camera'] = depth_xyzn_per_camera

    # Scale SDF
    data_batch['grid_samples'] = data_batch['grid_samples'] * \
        tf.constant([1, 1, 1, self._sdf_scale], dtype=tf.float32)[None, None, None, None, :]
    data_batch['uniform_samples'] = data_batch['uniform_samples'] * \
        tf.constant([1, 1, 1, self._sdf_scale], dtype=tf.float32)[None, None, :]
    data_batch['near_surface_samples'] = data_batch['near_surface_samples'] * \
        tf.constant([1, 1, 1, self._sdf_scale], dtype=tf.float32)[None, None, :]
    if 'uniform_samples_per_camera' in data_batch:
      data_batch['uniform_samples_per_camera'] = data_batch['uniform_samples_per_camera'] * \
          tf.constant([1, 1, 1, self._sdf_scale], dtype=tf.float32)[None, None, None, :]
    if 'near_surface_samples_per_camera' in data_batch:
      data_batch['near_surface_samples_per_camera'] = data_batch['near_surface_samples_per_camera'] * \
          tf.constant([1, 1, 1, self._sdf_scale], dtype=tf.float32)[None, None, None, :]

    input_data = data_batch['grid_samples'][..., 3:4]
    # Tensor with shape [batch_size, dim_d, dim_h, dim_w, 1].

    gt_data = input_data

    return input_data, gt_data

  def _forward_decoder_on_points_data(
      self, codes: tf.Tensor, points_data: Dict[str, tf.Tensor], level: int,
      training: bool,
      save_key_prefix: str) -> Tuple[Dict[str, Any], Dict[str, tf.Tensor]]:
    """Forwards implicit decoder on points data for one level.

    Args:
      codes: [batch_size, code_grid_d， code_grid_h, code_grid_w, code_grid_c]
        tensor, latent code grid at selected level.
      points_data: contains sampled points.
      level: selected level to be evaluated.
      training: flag indicating training phase.
      save_key_prefix: key prefix when storing outputs to dict.

    Returns:
      model_outputs_and_targets: contains outputs.
      debug_data: contains data for debugging.
    """
    if not training:
      codes = tf.stop_gradient(codes)

    model_outputs_and_targets = {}
    for key in points_data.keys():
      # Get sampled points.
      if key.startswith('points/') or key.startswith('points_symmetry/') or \
          key == 'points_consistency':
        points = points_data[key]
        # Tensor with shape [batch_size, num_point, num_point_dim].
      else:
        continue

      # Divide into chunks, pad dummy points if needed.
      batch_size, num_point, _ = points.shape
      num_chunk = math.ceil(num_point / self._max_point_per_chunk)
      num_dummy_point = num_chunk * self._max_point_per_chunk - num_point
      points_all_chunks = tf.concat(
          [points,
           tf.zeros([batch_size, num_dummy_point, points.shape[-1]])],
          axis=1)

      out = tf.zeros([batch_size, 0, 1])
      points_normalize_chunks = tf.zeros([batch_size, 0, self._num_point_dim])
      points_renormalize_chunks = tf.zeros([batch_size, 0, self._num_point_dim])
      latent_codes_chunks = tf.zeros([batch_size, 0, 6])
      decoder_inputs = tf.zeros([batch_size, 0, 0])

      # Go through each chunk.
      for ith_chunk in tf.range(num_chunk):
        tf.autograph.experimental.set_loop_options(shape_invariants=[
            (decoder_inputs, tf.TensorShape([batch_size, None, None])),
            (out, tf.TensorShape([batch_size, None, None])),
            (points_normalize_chunks, tf.TensorShape([batch_size, None, None])),
            (points_renormalize_chunks,
             tf.TensorShape([batch_size, None, None])),
            (latent_codes_chunks, tf.TensorShape([batch_size, None, None])),
        ])

        start_idx = self._max_point_per_chunk * ith_chunk
        end_idx = self._max_point_per_chunk * (ith_chunk + 1)
        points_chunk_i = points_all_chunks[:, start_idx:end_idx, :]

        # Get latent codes for each point.
        if self._code_for_point_mode == 'interpolate':
          (codes_for_points, points_normalize,
           debug_data_chunk_i) = self._interpolate_codes_at_points(
               codes, points_chunk_i, level)
        else:
          raise ValueError('Unknown code_for_point_mode',
                           self._code_for_point_mode)

        if not training:
          for debug_data_key in debug_data_chunk_i:
            debug_data_chunk_i[debug_data_key] = tf.stop_gradient(
                debug_data_chunk_i[debug_data_key])
        points_normalize_chunks = tf.concat(
            [points_normalize_chunks, debug_data_chunk_i['points_normalize']],
            axis=1)
        points_renormalize_chunks = tf.concat([
            points_renormalize_chunks, debug_data_chunk_i['points_renormalize']
        ],
                                              axis=1)
        latent_codes_chunks = tf.concat(
            [latent_codes_chunks, debug_data_chunk_i['latent_codes']], axis=1)

        # Get decoder inputs.
        decoder_input_config_i = self._decoder_input_config[level]
        if 'lat_code' in decoder_input_config_i['empty_vars']:
          codes_for_points = codes_for_points * 0
        if 'coord' in decoder_input_config_i['empty_vars']:
          points_normalize = points_normalize * 0
        if decoder_input_config_i['data'] == 'lat_code+coord':
          points_feats_to_concat = points_normalize
          # Stack points with corresponding latent codes.
          decoder_inputs = tf.concat([codes_for_points, points_feats_to_concat],
                                     axis=-1)
        elif decoder_input_config_i['data'] == 'lat_code':
          decoder_inputs = codes_for_points

        # Forward decoder.
        out_chunk_i = self._decoder([decoder_inputs],
                                    levels=[level],
                                    training=training)
        out_chunk_i = out_chunk_i[0]

        if not training:
          out_chunk_i = tf.stop_gradient(out_chunk_i)

        out = tf.concat([out, out_chunk_i], axis=1)

      # Merge all chunks.
      points_normalize_chunks = points_normalize_chunks[:, :num_point, :]
      points_renormalize_chunks = points_renormalize_chunks[:, :num_point, :]
      latent_codes_chunks = latent_codes_chunks[:, :num_point, :]
      debug_data = {
          'points_normalize': points_normalize_chunks,
          'points_renormalize': points_renormalize_chunks,
          'latent_codes': latent_codes_chunks
      }
      out = out[:, :num_point, :]

      # For points with GT supervision.
      if key.startswith('points/'):
        points_sdf_gt = points_data['points_sdf_gt/' + key[7:]]
        points_sdf_gt_residual = points_data['points_sdf_gt_residual/' + \
            key[7:]]
        points_sdf_gt_cumulative = points_data['points_sdf_gt_cumulative/' + \
            key[7:]]

        # Get cumulative prediction.
        out_cumulative_old = points_sdf_gt - points_sdf_gt_residual
        out_cumulative = out_cumulative_old + out

        # Set label.
        if self._label_config[level]['data'] == 'gt_full':
          label_data = points_sdf_gt
        elif self._label_config[level]['data'] in ['gt_residual', 'none']:
          label_data = points_sdf_gt_residual
        else:
          raise ValueError('Unknown label_config data at level', level, ':',
                           self._label_config[level]['data'])

        # Stop gradient for label.
        if self._label_config[level]['stop_grad']:
          label_data = tf.stop_gradient(label_data)

        # Save outputs.
        if self._label_config[level]['data'] == 'gt_full':
          model_outputs_and_targets[save_key_prefix + 'points_sdf/' + \
              key[7:] + '/level' + str(level)] = (
                  out, label_data, level,
                  'loss/sdf' if not save_key_prefix else 'metric/sdf')
        elif self._label_config[level]['data'] in ['gt_residual']:
          model_outputs_and_targets[save_key_prefix + 'points_residual_sdf/' + \
              key[7:] + '/level' + str(level)] = (
                  out, label_data, level,
                  'loss/sdf' if not save_key_prefix else 'metric/sdf')
          model_outputs_and_targets[save_key_prefix + 'points_sdf/' + \
              key[7:] + '/level' + str(level)] = (
                  out_cumulative, points_sdf_gt, level, 'metric/sdf')
        elif self._label_config[level]['data'] == 'none':
          model_outputs_and_targets[save_key_prefix + 'points_residual_sdf/' + \
              key[7:] + '/level' + str(level)] = (
                  out, label_data, level, 'metric/sdf')
          model_outputs_and_targets[save_key_prefix + 'points_sdf/' + \
              key[7:] + '/level' + str(level)] = (
                  out_cumulative, points_sdf_gt, level, 'metric/sdf')

        # Update GT residual.
        points_data['points_sdf_gt_residual/' + \
            key[7:]] = points_sdf_gt - out_cumulative
        points_data['points_sdf_gt_cumulative/' + \
            key[7:]] = points_sdf_gt_cumulative

      # For points used in symmetry loss.
      elif key.startswith('points_symmetry/'):
        key_dist = key.replace('points_symmetry/', 'points_symmetry_dist/')
        model_outputs_and_targets[save_key_prefix + key + '/level' + \
            str(level)] = (
                out, points_data[key_dist], level,
                self._label_config[level]['data'], self._label_config[0]['data'],
                'loss/symmetry' if not save_key_prefix else 'metric/symmetry')

      # For points used in consistency loss.
      elif key == 'points_consistency':
        model_outputs_and_targets[save_key_prefix + 'points_consistency/level' + \
            str(level)] = (
                out, points_data['points_consistency_dist'], level,
                self._label_config[level]['data'],
                'loss/consistency' if not save_key_prefix else 'metric/consistency')

    return model_outputs_and_targets, debug_data

  def _pipeline_general(self,
                        data_batch: Dict[str, Any],
                        levels: Sequence[int],
                        training: bool = False,
                        do_eval: bool = False,
                        optim_mode: str = 'full',
                        latent_code_type: str = 'train',
                        eval_data_mode: str = None,
                        flags: Dict[str, Any] = None):
    """General pipeline for encoding SDF grid and then decoding into SDF at sampled points.

    Args:
      data_batch: contains input data to be used.
      levels: selected levels to be evaluated.
      training: flag indicating training phase.
      do_eval: whether do evaluation.
      optim_mode: mode of optimization, one of ['full', 'latent_optim'].
      latent_code_type: type of latent code.
      eval_data_mode: mode for evaluation data.
      flags: additional flags.

    Returns:
      model_outputs_and_targets: contains model outputs and targets.
      image_summaries: contains image tensors to visualize in tensorboard.
    """
    if self._num_point_dim == 2:
      _, dim_h, dim_w, _ = data_batch['input'].shape
      spatial_dims = [dim_h, dim_w]
    elif self._num_point_dim == 3:
      _, dim_d, dim_h, dim_w, _ = data_batch['input'].shape
      spatial_dims = [dim_d, dim_h, dim_w]

    gt_map = data_batch['gt']
    gt_map_for_input = gt_map
    if self._num_point_dim == 2:
      gt_data_for_label = {'sdf_map': gt_map}
    elif self._num_point_dim == 3:
      gt_data_for_label = {
          'grid_samples': data_batch['grid_samples'],
          'uniform_samples': data_batch['uniform_samples'],
          'near_surface_samples': data_batch['near_surface_samples']
      }
      if 'uniform_samples_per_camera' in data_batch:
        gt_data_for_label['uniform_samples_per_camera'] = data_batch[
            'uniform_samples_per_camera']
      if 'near_surface_samples_per_camera' in data_batch:
        gt_data_for_label['near_surface_samples_per_camera'] = data_batch[
            'near_surface_samples_per_camera']
      if 'depth_xyzn_per_camera' in data_batch:
        gt_data_for_label['depth_xyzn_per_camera'] = data_batch[
            'depth_xyzn_per_camera']

    # Unified processing on input.
    if self._input_config_unified['clip'][0]:
      clip_min_max = self._input_config_unified['clip'][1]
      gt_map_for_input = tf.clip_by_value(gt_map_for_input, clip_min_max[0],
                                          clip_min_max[1])

    # Unified processing on label.
    if self._label_config_unified['clip'][0]:
      clip_min_max = self._label_config_unified['clip'][1]
      if self._num_point_dim == 2:
        gt_data_for_label['sdf_map'] = tf.clip_by_value(
            gt_data_for_label['sdf_map'], clip_min_max[0], clip_min_max[1])
      else:
        for key in gt_data_for_label:
          if gt_data_for_label[key].shape[-1] > 3 and key not in [
              'depth_xyzn_per_camera'
          ]:
            gt_data_for_label[key] = tf.concat([
                gt_data_for_label[key][..., :3],
                tf.clip_by_value(gt_data_for_label[key][..., 3:4],
                                 clip_min_max[0], clip_min_max[1])
            ],
                                               axis=-1)

    # Initialize dictionaries to store outputs.
    model_outputs_and_targets = {
        'training': training,
        'image_size': spatial_dims
    }
    image_summaries = {}

    # Sample points for training/latent optimization.
    if optim_mode == 'full':
      points_data = self._sample_points(
          spatial_dims,
          gt_data_for_label,
          self._train_point_sampler,
          params=flags)
      optim_sampler = self._train_point_sampler
    elif optim_mode == 'latent_optim':
      points_data = self._sample_points(
          spatial_dims,
          gt_data_for_label,
          self._latent_optim_point_sampler,
          params=flags)
      optim_sampler = self._latent_optim_point_sampler

    # Sample points for evaluation.
    params_for_eval = self._eval_sampling_params_init
    if 'mask' in optim_sampler.default_params:
      params_for_eval['mask'] = optim_sampler.default_params['mask']
    if eval_data_mode == 'all':
      params_for_eval['all_pixels/mode'] = 'all'
    points_eval_data = self._sample_points(spatial_dims, gt_data_for_label,
                                           self._eval_point_sampler,
                                           params_for_eval)
    model_outputs_and_targets['mask_for_point'] = points_eval_data[
        'mask_for_point']

    # Initialize GT residual and GT cumulative.
    keys = list(points_data.keys())
    for key in keys:
      if key.startswith('points/'):
        points_data['points_sdf_gt_residual/' +
                    key[7:]] = points_data['points_sdf_gt/' + key[7:]]
        points_data['points_sdf_gt_cumulative/' + key[7:]] = tf.zeros_like(
            points_data['points_sdf_gt/' + key[7:]])
    keys = list(points_eval_data.keys())
    for key in keys:
      if key.startswith('points/'):
        points_eval_data['points_sdf_gt_residual/' + \
            key[7:]] = points_eval_data['points_sdf_gt/' + key[7:]]
        points_eval_data['points_sdf_gt_cumulative/' + key[7:]] = tf.zeros_like(
            points_eval_data['points_sdf_gt/' + key[7:]])

    # Iterate over all levels.
    for ith_level in range(len(levels)):
      i = levels[ith_level]
      debug_data = {}

      # Set input.
      input_data = gt_map_for_input

      if self._mode == 'fully_multi_level':
        if self._encoder_mode == 'input_enc+f2c':
          # Get root feature from input encoder.
          if ith_level == 0:
            if optim_mode == 'full':
              root_feature = self._input_encoder[0](
                  input_data, training=training)
              debug_data['root_feature'] = root_feature
              model_outputs_and_targets['root_feature'] = root_feature
            elif optim_mode == 'latent_optim':
              pass
            else:
              raise ValueError('Unknown optim_mode: %s' % optim_mode)

          # Get latent code grid from feature to code net.
          if 'code_grid/level' + str(i) in model_outputs_and_targets:
            codes_this_level = model_outputs_and_targets['code_grid/level' +
                                                         str(i)][0]
          else:
            if self._feature_to_code_net.mode == 'separate_branch':
              if optim_mode != 'latent_optim':
                codes_this_level = self._feature_to_code_net(
                    root_feature, levels=[i], training=training)[i]
                model_outputs_and_targets['code_grid/level' +
                                          str(i)] = (codes_this_level, i)
              else:
                codes_all_level = self._gather_latent_codes(
                    latent_code_type=latent_code_type)
                codes_this_level = codes_all_level[i]
                codes_this_level = self._feature_to_code_net(
                    codes_this_level,
                    levels=[i],
                    training=training,
                    decoder_only=True)[i]
            elif self._feature_to_code_net.mode in [
                'single_branch', 'single_dec_branch'
            ]:
              if optim_mode != 'latent_optim':
                codes_all_level = self._feature_to_code_net(
                    root_feature, training=training)
              else:
                codes_all_level = self._gather_latent_codes(
                    latent_code_type=latent_code_type)
                codes_all_level = self._feature_to_code_net(
                    codes_all_level, training=training, decoder_only=True)
              codes_this_level = codes_all_level[i]
              for level_idx, codes_all_level_i in enumerate(codes_all_level):
                model_outputs_and_targets['code_grid/level' +
                                          str(level_idx)] = (codes_all_level_i,
                                                             level_idx)
        else:
          raise ValueError('mode: %s does not support encoder_mode: %s' % \
                           (self._mode, self._encoder_mode))

        # Forward decoder on training points.
        if training:
          model_outputs_and_targets_train, _ = \
              self._forward_decoder_on_points_data(
                  codes_this_level, points_data, i, training=True,
                  save_key_prefix='')
          model_outputs_and_targets.update(model_outputs_and_targets_train)

        # Forward decoder on evaluation points.
        if self._do_eval_every_iter or do_eval:
          model_outputs_and_targets_eval, debug_data_temp = \
              self._forward_decoder_on_points_data(
                  codes_this_level, points_eval_data, i, training=False,
                  save_key_prefix='eval_')
          model_outputs_and_targets.update(model_outputs_and_targets_eval)
          debug_data.update(debug_data_temp)

        # Add summaries for this level.
        if do_eval:
          # Input data.
          summary_key = 'input_data/level' + str(i)
          summary_data = tf.stack(
              [input_data[..., i_channel] for i_channel in
               range(input_data.shape[-1])], axis=1)[..., None] / \
              self.summary_config['sdf_range'] * 0.5 + 0.5
          image_summaries_update = misc_utils.get_image_summary(
              summary_key,
              summary_data,
              channels_use='first',
              spatial_dims=None,
              normalize=False,
              summary_config=self.summary_config,
              extra_axis=1)
          image_summaries.update(image_summaries_update)

          # Root feature.
          if i == 0:
            data_key = 'root_feature'
            summary_key = 'misc/root_feature/level' + str(i)
            if data_key in debug_data:
              image_summaries_update = misc_utils.get_image_summary(
                  summary_key,
                  debug_data[data_key],
                  channels_use='first',
                  spatial_dims=None,
                  normalize=True,
                  summary_config=self.summary_config)
              image_summaries.update(image_summaries_update)

          # Latent code grid.
          summary_key = 'misc/code_grid/level' + str(i)
          image_summaries_update = misc_utils.get_image_summary(
              summary_key,
              codes_this_level,
              channels_use='first',
              spatial_dims=None,
              normalize=True,
              summary_config=self.summary_config)
          image_summaries.update(image_summaries_update)

          summary_key = 'misc/code_grid_last/level' + str(i)
          image_summaries_update = misc_utils.get_image_summary(
              summary_key,
              codes_this_level,
              channels_use='last',
              spatial_dims=None,
              normalize=True,
              summary_config=self.summary_config)
          image_summaries.update(image_summaries_update)

          # Latent codes for points.
          data_key = 'latent_codes'
          if data_key in debug_data:
            summary_key = 'misc/latent_codes/level' + str(i)
            image_summaries_update = misc_utils.get_image_summary(
                summary_key,
                debug_data[data_key],
                channels_use='first',
                spatial_dims=spatial_dims,
                normalize=True,
                summary_config=self.summary_config,
                data_mode=eval_data_mode)
            image_summaries.update(image_summaries_update)

            summary_key = 'misc/latent_codes_last/level' + str(i)
            image_summaries_update = misc_utils.get_image_summary(
                summary_key,
                debug_data[data_key],
                channels_use='last',
                spatial_dims=spatial_dims,
                normalize=True,
                summary_config=self.summary_config,
                data_mode=eval_data_mode)
            image_summaries.update(image_summaries_update)
      else:
        raise ValueError('Unknown mode: %s' % self._mode)

    return model_outputs_and_targets, image_summaries

  def call(self,
           data_batch: Dict[str, Any],
           training: bool = False,
           do_eval: bool = False,
           optim_mode: str = 'full',
           latent_code_type: str = 'train',
           eval_data_mode: str = None,
           flags: Dict[str, Any] = None) -> Dict[str, Any]:
    """Forward method.

    Args:
      data_batch: contains input data to be used.
      training: flag indicating training phase.
      do_eval: whether do evaluation.
      optim_mode: mode of optimization, one of ['full', 'latent_optim'].
      latent_code_type: type of latent code.
      eval_data_mode: mode for evaluation data.
      flags: additional flags.

    Returns:
      A dictionary containing:
        model_outputs_and_targets: contains model outputs and targets.
        image_summaries: contains image tensors to visualize in tensorboard.
    """
    # Parse options.
    if 'levels' in data_batch.keys():
      levels = data_batch['levels']
    else:
      levels = None

    if levels is None:
      levels = list(range(self.num_level))
    if eval_data_mode is None:
      eval_data_mode = self._eval_data_mode

    # Preprocess data.
    if self._num_point_dim == 2:
      data_batch['input'], data_batch['gt'] = self._preprocess_data(data_batch)
      batch_size, dim_h, dim_w, _ = data_batch['input'].shape
      spatial_dims = [dim_h, dim_w]
    elif self._num_point_dim == 3:
      data_batch['input'], data_batch['gt'] = self._preprocess_data_3d(
          data_batch)
      batch_size, dim_d, dim_h, dim_w, _ = data_batch['input'].shape
      spatial_dims = [dim_d, dim_h, dim_w]

    # Forward pipeline.
    if self._pipeline_mode == 'general':
      model_outputs_and_targets, image_summaries = self._pipeline_general(
          data_batch,
          levels,
          training,
          do_eval,
          optim_mode,
          latent_code_type,
          eval_data_mode,
          flags=flags)
    else:
      raise ValueError('Unknown pipeline_mode: %s' % self._pipeline_mode)

    # Add summaries.
    if do_eval:
      for _, level in enumerate(levels):
        if self._num_point_dim == 2:
          # Contours error.
          data_key = 'eval_points_sdf/all_pixels/level' + str(level)
          summary_key = 'contour/level' + str(level)
          if data_key in model_outputs_and_targets.keys():
            sdf_map_pred, sdf_map_gt, _, _ = model_outputs_and_targets[data_key]

            # Get GT contours.
            sdf_map_gt = tf.reshape(sdf_map_gt, [batch_size, *spatial_dims, 1])
            contour_points_gt = [
                mesh_utils.sdf_to_contour(sdf_map_gt[j, ...])
                for j in range(batch_size)
            ]

            # Get predicted contours at this level.
            sdf_map_pred = tf.reshape(sdf_map_pred,
                                      [batch_size, *spatial_dims, 1])
            contour_points_pred = [
                mesh_utils.sdf_to_contour(sdf_map_pred[j, ...])
                for j in range(batch_size)
            ]

            # Compute Chamfer distance.
            chamfer_eval_data = [
                chamfer_distance.evaluate(contour_points_gt[j],
                                          contour_points_pred[j])
                for j in range(batch_size)
            ]

            # Get contours image.
            contour_image_gt = []
            contour_image_pred = []
            for j in range(batch_size):
              contour_image_gt.append(
                  mesh_utils.contour_to_image(
                      contour_points_gt[j], *spatial_dims,
                      chamfer_eval_data[j][2][:, None],
                      self.summary_config['contours_err_max']))
              contour_image_pred.append(
                  mesh_utils.contour_to_image(
                      contour_points_pred[j], *spatial_dims,
                      chamfer_eval_data[j][3][:, None],
                      self.summary_config['contours_err_max']))
            contour_image_gt = tf.stack(contour_image_gt, axis=0)
            contour_image_pred = tf.stack(contour_image_pred, axis=0)

            # Save image summaries.
            image_summaries[summary_key] = tf.concat(
                [contour_image_gt, contour_image_pred,
                 tf.clip_by_value(contour_image_gt + contour_image_pred, 0, 1),
                 tf.tile(tf.abs(
                     tf.cast(tf.reduce_sum(contour_image_gt, axis=-1) > 0,
                             tf.float32) - \
                     tf.cast(tf.reduce_sum(contour_image_pred, axis=-1) > 0,
                             tf.float32))[..., None], [1, 1, 1, 3])
                 ],
                axis=1)

            data_key = 'contour/level' + str(level)
            model_outputs_and_targets[data_key] = (chamfer_eval_data, level,
                                                   'metric/contour')

        # Absolute per-pixel error on full SDF.
        data_key = 'eval_points_sdf/all_pixels/level' + str(level)
        summary_key = 'gt_vs_pred/sdf_map/level' + str(level)
        if data_key in model_outputs_and_targets.keys():
          sdf_map_pred, sdf_map_gt, _, _ = model_outputs_and_targets[data_key]
          sdf_map_pred_err = tf.abs(sdf_map_gt - sdf_map_pred)
          scale_factor = 0.5 / self.summary_config['sdf_range']
          summary_data = tf.stack(
              [sdf_map_gt * scale_factor + 0.5,
               sdf_map_pred * scale_factor + 0.5,
               sdf_map_pred_err * self.summary_config['sdf_err_factor'] * \
               scale_factor], axis=1)
          image_summaries_update = misc_utils.get_image_summary(
              summary_key,
              summary_data,
              channels_use='first',
              spatial_dims=spatial_dims,
              normalize=False,
              summary_config=self.summary_config,
              extra_axis=1,
              data_mode=eval_data_mode)
          image_summaries.update(image_summaries_update)

        # Absolute per-pixel error on residual SDF.
        data_key = 'eval_points_residual_sdf/all_pixels/level' + str(level)
        summary_key = 'residual_gt_vs_pred/sdf_map/level' + str(level)
        if data_key in model_outputs_and_targets.keys():
          sdf_map_pred, sdf_map_gt, _, _ = model_outputs_and_targets[data_key]
          sdf_map_pred_err = tf.abs(sdf_map_gt - sdf_map_pred)
          scale_factor = 0.5 / self.summary_config['sdf_range']
          summary_data = tf.stack(
              [sdf_map_gt * scale_factor + 0.5,
               sdf_map_pred * scale_factor + 0.5,
               sdf_map_pred_err * self.summary_config['sdf_err_factor'] * \
               scale_factor], axis=1)
          image_summaries_update = misc_utils.get_image_summary(
              summary_key,
              summary_data,
              channels_use='first',
              spatial_dims=spatial_dims,
              normalize=False,
              summary_config=self.summary_config,
              extra_axis=1,
              data_mode=eval_data_mode)
          image_summaries.update(image_summaries_update)

        # Occupancy error.
        data_key = 'eval_points_sdf/all_pixels/level' + str(level)
        summary_key = 'occupancy/level' + str(level)
        if data_key in model_outputs_and_targets.keys():
          sdf_map_pred, sdf_map_gt, _, _ = model_outputs_and_targets[data_key]
          occ_gt = tf.cast(sdf_map_gt < 0, dtype=tf.float32)
          occ_pred = tf.cast(sdf_map_pred < 0, dtype=tf.float32)
          occ_pred_err = tf.abs(occ_gt - occ_pred)
          summary_data = tf.stack([occ_gt, occ_pred, occ_pred_err], axis=1)
          image_summaries_update = misc_utils.get_image_summary(
              summary_key,
              summary_data,
              channels_use='first',
              spatial_dims=spatial_dims,
              normalize=False,
              summary_config=self.summary_config,
              extra_axis=1,
              data_mode=eval_data_mode)
          image_summaries.update(image_summaries_update)

          # Compute IoU.
          data_key = 'iou/level' + str(level)
          iou = metric_utils.point_iou_tf(occ_pred, occ_gt)
          # Tensor with shape [batch_size].

          model_outputs_and_targets[data_key] = (iou, level, 'metric/iou')

    return {
        'model_outputs_and_targets': model_outputs_and_targets,
        'image_summaries': image_summaries
    }
