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

"""Tests for google3.vr.perception.volume_compression.mdif.train_eval_lib."""

import functools
import os
import tempfile
from unittest import mock

from absl.testing import parameterized
import tensorflow as tf

mock.patch('tensorflow.function', lambda func: func).start()

from google3.vr.perception.volume_compression.mdif import train_eval_lib  # pylint: disable=g-import-not-at-top
from google3.vr.perception.volume_compression.mdif.model import loss_lib  # pylint: disable=g-import-not-at-top
from google3.vr.perception.volume_compression.mdif.model import network_pipeline  # pylint: disable=g-import-not-at-top


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


def _generate_dataset():
  batch_size = 1
  spatial_dims = (4, 4, 4)
  num_view = 1

  data_batch = {}
  data_batch['data_key'] = tf.constant(['0000000000'])
  data_batch['data_id'] = tf.constant([0])
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
  data_batch['depth_xyzn_per_camera'] = tf.zeros((batch_size, num_view, 50, 6),
                                                 dtype=tf.float32)

  dataset = [data_batch]
  return dataset


class TrainEvalLibTest(parameterized.TestCase, tf.test.TestCase):

  @mock.patch.object(train_eval_lib, 'get_training_elements', autospec=True)
  def test_onedevice_strategy(self, mock_get_training_elements):
    general_params = _get_general_params()
    loss_params = _get_loss_params()
    input_encoder_params = _get_input_encoder_params()
    feature_to_code_net_params = _get_feature_to_code_net_params()
    decoder_params = _get_decoder_params()
    (train_sampling_params, eval_sampling_params,
     latent_optim_sampling_params) = _get_sampling_params()

    mock_get_training_elements.return_value = (
        [
            functools.partial(
                network_pipeline.MultiresDeepImplicitFunction,
                general_params=general_params,
                loss_params=loss_params,
                input_encoder_params=input_encoder_params,
                feature_to_code_net_params=feature_to_code_net_params,
                decoder_params=decoder_params,
                train_sampling_params=train_sampling_params,
                eval_sampling_params=eval_sampling_params,
                latent_optim_sampling_params=latent_optim_sampling_params)
        ],
        functools.partial(
            loss_lib.MultiresDeepImplicitLoss,
            model_params=general_params,
            loss_params=loss_params),
        [functools.partial(tf.keras.optimizers.Adam, learning_rate=1e-4)
        ], train_eval_lib._distributed_train_step,
        train_eval_lib._distributed_eval_step)

    dataset = _generate_dataset()

    train_base_folder = tempfile.mkdtemp()
    # Smoke test to make sure that the model builds and trains for an epoch.
    train_eval_lib.train_pipeline(
        training_mode='cpu',
        base_folder=train_base_folder,
        data_sources=dataset,
        train_data_filter='all',
        batch_size=1,
        n_iterations=1,
        n_iterations_per_batch=1,
        save_summaries_frequency=1,
        save_checkpoint_frequency=1,
        time_every_n_steps=1,
        data_sources_type='directly_use')

    # Smoke test to make sure that the model can be evaluated.
    eval_base_folder = tempfile.mkdtemp()
    with self.subTest(name='eval_feed_forward'):
      train_eval_lib.eval_pipeline(
          eval_mode='cpu',
          data_sources={'test': dataset},
          eval_data_filter='all',
          train_base_folder=train_base_folder,
          eval_base_folder=eval_base_folder,
          batch_size=1,
          eval_name='test',
          optim_mode='feed_forward',
          n_iterations_per_batch=1,
          data_sources_type='directly_use',
          save_mode_sdf_grid=['always'],
          only_eval_one_ckpt=True)
    with self.subTest(name='eval_latent_optim'):
      train_eval_lib.eval_pipeline(
          eval_mode='cpu',
          data_sources={'test': dataset},
          eval_data_filter='all',
          train_base_folder=train_base_folder,
          eval_base_folder=eval_base_folder,
          batch_size=1,
          eval_name='test',
          optim_mode='latent_optim',
          n_iterations_per_batch=1,
          data_sources_type='directly_use',
          save_mode_sdf_grid=['always'],
          only_eval_one_ckpt=True)

    # Smoke test to make sure that the model can be used for inference.
    infer_base_folder = tempfile.mkdtemp()
    with self.subTest(name='infer_feed_forward'):
      train_eval_lib.inference_pipeline(
          eval_mode='cpu',
          data_sources=dataset,
          data_sources_ref=dataset,
          data_filter='all',
          model_path=os.path.join(train_base_folder, 'ckpt-1'),
          output_path=infer_base_folder,
          optim_mode='feed_forward',
          n_iterations_per_batch=1,
          save_summaries_frequency_latent_optim=1,
          timing_frequency=1,
          data_sources_type='directly_use',
          save_mode_sdf_grid=['every', 1],
          override_save=True)
    with self.subTest(name='infer_latent_optim'):
      train_eval_lib.inference_pipeline(
          eval_mode='cpu',
          data_sources=dataset,
          data_sources_ref=dataset,
          data_filter='all',
          model_path=os.path.join(train_base_folder, 'ckpt-1'),
          output_path=infer_base_folder,
          optim_mode='latent_optim',
          n_iterations_per_batch=1,
          save_summaries_frequency_latent_optim=1,
          timing_frequency=1,
          data_sources_type='directly_use',
          save_mode_sdf_grid=['every', 1],
          override_save=True)


if __name__ == '__main__':
  tf.test.main()
