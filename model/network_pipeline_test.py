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

"""Tests for google3.vr.perception.volume_compression.mdif.model.network_pipeline."""

from absl.testing import parameterized
import tensorflow as tf

from google3.vr.perception.volume_compression.mdif.model import network_pipeline


def _get_general_params():
  general_params = {
      'debug_mode': True,
      'num_train_data': 1,
      'num_test_data': 1,
      'latent_optim_target': 'code_grid_enc',
      'code_grid_enc_shape': [[1, 1, 1, 4], [2, 2, 2, 2]],
      'codes_init_std': [0., 0.],
      'mode': 'fully_multi_level',
      'encoder_mode': 'input_enc+f2c',
      'code_for_point_mode': 'interpolate',
      'pipeline_mode': 'general',
      'sdf_scale': 100,
      'max_point_per_chunk': 32,
      'num_point_dim': 3,
      'num_level': 2,
      'grid_shape': [[1, 1, 1], [1, 1, 1]],
      'grid_range_min': [-1, -1, -1],
      'grid_range_max': [1, 1, 1],
      'grid_mode': 'regular',
      'input_config_unified': {
          'clip': [True, [-5, 5]],
      },
      'label_config_unified': {
          'clip': [True, [-5, 5]]
      },
      'decoder_input_config': [
          {
              'data': 'lat_code+coord',
              'empty_vars': [],
          },
          {
              'data': 'lat_code',
              'empty_vars': []
          },
      ],
      'label_config': [
          {
              'data': 'gt_residual',
              'stop_grad': False
          },
          {
              'data': 'gt_residual',
              'stop_grad': False
          },
      ],
      'summary_config': {
          'sdf_range': 5,
          'sdf_err_factor': 2,
          'contours_err_max': 5,
          'slice_idx_z': [0.5],
          'slice_idx_y': [0.5],
          'slice_idx_x': [0.5],
      },
      'eval_data_mode': 'slices',
  }

  return general_params


def _get_loss_params():
  loss_params = {
      'sdf_l1': {
          'term_weight': [1.0, 1.0]
      },
      'sdf_reg_l1': {
          'term_weight': [0., 0.]
      },
      'sdf_consistency_l1': {
          'mode': ['every', 1],
          'stop_grad_ref': True,
          'point_weight_config/dist_to_visible': ['gaussian', 0.1],
          'term_weight': [0e0, 0e0]
      },
      'code_reg_l2': {
          'term_weight': [0e0, 0e0]
      },
      'root_feat_reg_l2': {
          'term_weight': [0e0]
      },
      'point_weight_config': [
          {
              'gt_gaussian': {
                  'apply': False,
                  'sigma': 32.0
              },
              'pred_gaussian': {
                  'apply': False,
                  'sigma': 2.0
              }
          },
          {
              'gt_gaussian': {
                  'apply': False,
                  'sigma': 8.0
              },
              'pred_gaussian': {
                  'apply': False,
                  'sigma': 2.0
              }
          },
      ],
      'summary_config': {
          'slice_idx_z': [0.5],
          'slice_idx_y': [0.5],
          'slice_idx_x': [0.5],
      },
  }
  return loss_params


def _get_input_encoder_params():
  input_encoder_params = [
      {
          'data_type': '3d',
          'net_type': 'fully_conv',
          'num_filters': [2],
          'num_out_channel': 2,
          'strides': [2],
          'num_conv_per_level': 2,
          'num_levels': 1,
          'final_pooling': None,
          'activation_params': {
              'type': 'leaky_relu',
              'alpha': 0.2
          },
      },
  ]

  return input_encoder_params


def _get_feature_to_code_net_params():
  feature_to_code_net_params = {
      'data_type':
          '3d',
      'mode':
          'single_dec_branch',
      'out_pre_upsample_id': [0, 1],
      'dec_only_apply_mask':
          False,
      'unified_mask_config':
          None,
      'fusion_params': {
          'mode': 'concat'
      },
      'block_params': [
          [
              [
                  'EncoderTemplate', {
                      'net_type': 'fully_conv',
                      'num_levels': 1,
                      'num_filters': [4],
                      'num_out_channel': 4,
                      'strides': [2],
                      'num_conv_per_level': 2,
                      'kernel_size': [3, 1],
                      'final_pooling': None,
                      'normalization_params': None,
                      'activation_params': {
                          'type': 'leaky_relu',
                          'alpha': 0.2
                      },
                  }
              ],
              [
                  'DecoderConv', {
                      'num_levels': 2,
                      'num_filters': [4, 1],
                      'num_out_channel': None,
                      'initial_upsample': [False, [8, 8], 'bilinear'],
                      'kernel_size': [1, 3, 1],
                      'kernel_size_deconv': 4,
                      'num_conv_per_level': 2,
                      'upsample_type': 'deconv',
                      'normalization_params': None,
                      'activation_params': {
                          'type': 'leaky_relu',
                          'alpha': 0.2
                      },
                  }
              ],
          ],
          [
              [
                  'EncoderTemplate', {
                      'net_type': 'fully_conv',
                      'num_levels': 0,
                      'num_filters': [],
                      'num_out_channel': 2,
                      'strides': [],
                      'kernel_size': [3],
                      'num_conv_per_level': 2,
                      'final_pooling': None,
                      'normalization_params': None,
                      'activation_params': {
                          'type': 'leaky_relu',
                          'alpha': 0.2
                      },
                  }
              ],
              [
                  'MaskingLayer', {
                      'mode': 'random',
                      'offset': (0, 0),
                      'masked_value': 0,
                      'dropout_rate': 0.5,
                      'dropout_rescale': False,
                      'resize_mode': 'downsample',
                      'resize_factor': 2,
                      'noise_config': None,
                  }
              ],
          ],
      ]
  }

  return feature_to_code_net_params


def _get_decoder_params():
  decoder_params = {
      'num_filter': 16,
      'num_out_channel': 1,
      'implicit_net_type': 'imnet',
      'share_net_level_groups': None,
      'activation_params': {
          'type': 'leaky_relu',
          'alpha': 0.2
      },
  }

  return decoder_params


def _get_sampling_params():
  train_sampling_params = {
      'normalize_coordinates': True,
      'all_pixels': False,
      'untruncated': False,
      'untruncated/num_point': 0,
      'untruncated/mode': 'uniform',
      'untruncated/truncate': 5,
      'regular': False,
      'regular/num_point': 0,
      'global': True,
      'global/num_point': 32,
      'near_surface': True,
      'near_surface/num_point': 32,
  }

  eval_sampling_params = {
      'normalize_coordinates': True,
      'all_pixels': True,
      'untruncated': False,
      'untruncated/num_point': 0,
      'untruncated/mode': 'uniform',
      'untruncated/truncate': 5,
      'regular': False,
      'regular/num_point': 0,
      'global': False,
      'global/num_point': 0,
      'near_surface': False,
      'near_surface/num_point': 0,
  }

  latent_optim_sampling_params = {
      'normalize_coordinates': True,
      'all_pixels': False,
      'untruncated': False,
      'untruncated/num_point': 0,
      'untruncated/mode': 'uniform',
      'untruncated/truncate': 5,
      'regular': False,
      'regular/num_point': 0,
      'global': True,
      'global/num_point': 32,
      'near_surface': True,
      'near_surface/num_point': 32,
  }

  return (train_sampling_params, eval_sampling_params,
          latent_optim_sampling_params)


class NetworkPipelineTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(('full'), ('latent_optim'))
  def test_multires_deep_implicit_function(self, optim_mode):
    general_params = _get_general_params()
    loss_params = _get_loss_params()
    input_encoder_params = _get_input_encoder_params()
    feature_to_code_net_params = _get_feature_to_code_net_params()
    decoder_params = _get_decoder_params()
    (train_sampling_params, eval_sampling_params,
     latent_optim_sampling_params) = _get_sampling_params()

    net = network_pipeline.MultiresDeepImplicitFunction(
        general_params, loss_params, input_encoder_params,
        feature_to_code_net_params, decoder_params, train_sampling_params,
        eval_sampling_params, latent_optim_sampling_params)

    batch_size = 1
    spatial_dims = (4, 4, 4)
    num_view = 1
    data_batch = {}
    data_batch['world2grid'] = tf.eye(
        4, batch_shape=[batch_size], dtype=tf.float32)
    data_batch['grid_samples'] = tf.zeros((batch_size, *spatial_dims, 1),
                                          dtype=tf.float32)
    data_batch['uniform_samples'] = tf.zeros((batch_size, 50, 4),
                                             dtype=tf.float32)
    data_batch['near_surface_samples'] = tf.zeros((batch_size, 50, 4),
                                                  dtype=tf.float32)
    data_batch['uniform_samples_per_camera'] = tf.zeros(
        (batch_size, num_view, 50, 4), dtype=tf.float32)
    data_batch['near_surface_samples_per_camera'] = tf.zeros(
        (batch_size, num_view, 50, 4), dtype=tf.float32)
    data_batch['depth_xyzn_per_camera'] = tf.zeros(
        (batch_size, num_view, 50, 6), dtype=tf.float32)

    data_batch_new = dict(data_batch)
    input_data, gt_data = net._preprocess_data_3d(data_batch_new)

    out = net(data_batch, training=True, do_eval=True, optim_mode=optim_mode)

    with self.subTest(name='init_latent_codes'):
      self.assertLen(net.codes_train_data, net.num_level)
      self.assertLen(net.codes_test_data, net.num_level)

    with self.subTest(name='gather_latent_codes'):
      for latent_code_type in ['train', 'test']:
        codes_each_level = net._gather_latent_codes(latent_code_type)
        for codes_level_i, code_grid_enc_shape_i in zip(
            codes_each_level, net.code_grid_enc_shape):
          self.assertSequenceEqual(codes_level_i.shape,
                                   (1, *code_grid_enc_shape_i))

    with self.subTest(name='interpolate_codes_at_points'):
      code_grid = tf.zeros((2, 4, 4, 4, 16), dtype=tf.float32)
      points = tf.zeros((2, 10, 3), dtype=tf.float32)
      (codes_for_points, points_normalize,
       debug_data) = net._interpolate_codes_at_points(
           code_grid, points, level=0)
      self.assertSequenceEqual(codes_for_points.shape, (2, 10, 16))
      self.assertSequenceEqual(points_normalize.shape, (2, 10, 3))
      self.assertSequenceEqual(debug_data['points_renormalize'].shape,
                               (2, 10, 3))
      self.assertSequenceEqual(debug_data['latent_codes'].shape, (2, 10, 6))

    with self.subTest(name='preprocess_data_3d'):
      self.assertSequenceEqual(input_data.shape, (batch_size, *spatial_dims, 1))
      self.assertSequenceEqual(gt_data.shape, (batch_size, *spatial_dims, 1))

    with self.subTest(name='sample_points'):
      gt_data_for_label = {
          'grid_samples':
              data_batch_new['grid_samples'],
          'uniform_samples':
              data_batch_new['uniform_samples'],
          'near_surface_samples':
              data_batch_new['near_surface_samples'],
          'uniform_samples_per_camera':
              data_batch_new['uniform_samples_per_camera'],
          'near_surface_samples_per_camera':
              data_batch_new['near_surface_samples_per_camera'],
          'depth_xyzn_per_camera':
              data_batch_new['depth_xyzn_per_camera'],
      }
      points_data = net._sample_points(spatial_dims, gt_data_for_label,
                                       net._train_point_sampler)
      self.assertSequenceEqual(points_data['mask_for_point'].shape,
                               (batch_size, 64))
      self.assertSequenceEqual(points_data['points/global/uniform'].shape,
                               (batch_size, 32, 3))
      self.assertSequenceEqual(
          points_data['points_sdf_gt/global/uniform'].shape,
          (batch_size, 32, 1))
      self.assertSequenceEqual(points_data['points/near_surface/uniform'].shape,
                               (batch_size, 32, 3))
      self.assertSequenceEqual(
          points_data['points_sdf_gt/near_surface/uniform'].shape,
          (batch_size, 32, 1))

    with self.subTest(name='final_outputs'):
      self.assertSequenceEqual(
          out['model_outputs_and_targets']['mask_for_point'].shape,
          (batch_size, 48))
      self.assertSequenceEqual(
          out['model_outputs_and_targets']['code_grid/level0'][0].shape,
          (batch_size, 1, 1, 1, 4))
      self.assertSequenceEqual(
          out['model_outputs_and_targets']['code_grid/level1'][0].shape,
          (batch_size, 2, 2, 2, 3))

      for level in range(2):
        for sdf_type in ['points_sdf', 'points_residual_sdf']:
          for sample_type in ['global/uniform', 'near_surface/uniform']:
            key = sdf_type + '/' + sample_type + '/level' + str(level)
            self.assertSequenceEqual(
                out['model_outputs_and_targets'][key][0].shape,
                (batch_size, 32, 1))
        for sdf_type in ['eval_points_sdf', 'eval_points_residual_sdf']:
          key = sdf_type + '/all_pixels/level' + str(level)
          self.assertSequenceEqual(
              out['model_outputs_and_targets'][key][0].shape,
              (batch_size, 48, 1))
        self.assertSequenceEqual(
            out['model_outputs_and_targets']['iou/level' + str(level)][0].shape,
            (batch_size,))


if __name__ == '__main__':
  tf.test.main()
