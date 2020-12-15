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

"""Tests for google3.vr.perception.volume_compression.mdif.model.network_multilevel."""

from absl.testing import parameterized
import tensorflow as tf

from google3.vr.perception.volume_compression.mdif.model import network_multilevel


def _get_block_params():
  block_params = [
      [
          [
              'EncoderTemplate', {
                  'net_type': 'fully_conv',
                  'num_levels': 4,
                  'num_filters': [16, 16, 16, 16],
                  'num_out_channel': 16,
                  'strides': [2, 2, 2, 2],
                  'num_conv_per_level': 2,
                  'kernel_size': [3, 3, 3, 3, 1],
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
                  'num_levels': 5,
                  'num_filters': [16, 16, 16, 16, 16],
                  'num_out_channel': None,
                  'initial_upsample': [False, [8, 8], 'bilinear'],
                  'kernel_size': [1, 3, 3, 3, 3, 1],
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
                  'num_levels': 3,
                  'num_filters': [16, 16, 16],
                  'num_out_channel': 32,
                  'strides': [2, 2, 2],
                  'kernel_size': [3, 3, 3, 3],
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
                  'mode': 'input_mask',
                  'offset': (0, 0),
                  'masked_value': 0,
                  'dropout_rate': 0.5,
                  'dropout_rescale': False,
                  'resize_mode': 'downsample',
                  'resize_factor': 2,
                  'noise_config': None,
              }
          ],
          [
              'DecoderConv', {
                  'num_levels': 4,
                  'num_filters': [32, 32, 32, 32],
                  'num_out_channel': None,
                  'initial_upsample': [False, [8, 8], 'bilinear'],
                  'kernel_size': [3, 3, 3, 3, 1],
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
  ]

  return block_params


class NetworkMultilevelTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((128, None, None),
                            ((128, 128, 64), ((0, 1),), None),
                            ((128, 64, 64), None, (0, 1)))
  def test_multilevel_implicit_decoder(self, num_filter, share_net_level_groups,
                                       levels):
    net = network_multilevel.MultiLevelImplicitDecoder(
        num_filter=num_filter,
        share_net_level_groups=share_net_level_groups,
        activation_params={'type': 'relu'})

    if levels is None:
      inputs = [
          tf.ones((2, 10, 8), dtype=tf.float32) for _ in range(net.num_level)
      ]
    else:
      inputs = [
          tf.ones((2, 10, 8), dtype=tf.float32) for _ in range(len(levels))
      ]
    output = net(inputs, levels)

    with self.subTest(name='output_len'):
      if levels is None:
        self.assertLen(output, net.num_level)
      else:
        self.assertLen(output, len(levels))
    with self.subTest(name='output_each_level_shape'):
      for output_i in output:
        self.assertSequenceEqual(output_i.shape, (2, 10, 1))

  @parameterized.parameters((None), ((100, 32),))
  def test_geometry_net(self, num_filter):
    net = network_multilevel.GeometryNet(num_filter=num_filter)
    inputs = tf.ones((2, 512), dtype=tf.float32)
    output = net(inputs)
    self.assertSequenceEqual(output.shape, (2, net.num_code_channel))

  @parameterized.parameters((None), ((100, 2048),))
  def test_partition_net(self, num_filter):
    net = network_multilevel.PartitionNet(num_filter=num_filter)
    inputs = tf.ones((2, net.num_in_channel), dtype=tf.float32)
    output = net(inputs)
    self.assertSequenceEqual(output.shape,
                             (2, net.num_out_channel * net.num_children))

  @parameterized.parameters(
      ('separate_branch', ((2, 16, 16, 16, 16), (2, 16, 16, 16, 32))),
      ('single_branch', ((2, 1, 1, 1, 16), (2, 2, 2, 2, 16))),
      ('single_dec_branch', ((2, 1, 1, 1, 16), (2, 2, 2, 2, 48))),
  )
  def test_feature_to_code_net(self, mode, out_shape):
    block_params = _get_block_params()
    unified_mask_config = {
        'spatial_dims': [4, 4, 4],
        'dropout_rate': 0.5,
        'dropout_rescale': True,
    }
    net = network_multilevel.FeatureToCodeNet(
        block_params=block_params,
        data_type='3d',
        mode=mode,
        out_pre_upsample_id=(0, 1),
        unified_mask_config=unified_mask_config)

    with self.subTest(name='encoder_decoder'):
      inputs = tf.zeros((2, 16, 16, 16, 1), dtype=tf.float32)
      out = net(inputs)
      for out_i, out_shape_i in zip(out, out_shape):
        self.assertSequenceEqual(out_i.shape, out_shape_i)

    with self.subTest(name='decoder_only'):
      inputs = (tf.zeros((2, 1, 1, 1, 16), dtype=tf.float32),
                tf.zeros((2, 2, 2, 2, 32), dtype=tf.float32))
      out = net(inputs, decoder_only=True)
      for out_i, out_shape_i in zip(out, out_shape):
        self.assertSequenceEqual(out_i.shape, out_shape_i)


if __name__ == '__main__':
  tf.test.main()
