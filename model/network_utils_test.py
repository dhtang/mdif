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

"""Tests for google3.vr.perception.volume_compression.mdif.model.network_utils."""

from absl.testing import parameterized
import tensorflow as tf
import tensorflow_addons.layers as tfa_layers

from google3.vr.perception.volume_compression.mdif.model import network_utils


class NetworkUtilsTest(parameterized.TestCase, tf.test.TestCase):

  def test_get_activation(self):
    layer = network_utils.get_activation(None)
    with self.subTest(name='dummy'):
      self.assertIsInstance(layer, tf.keras.layers.LeakyReLU)

    layer = network_utils.get_activation('leaky_relu')
    with self.subTest(name='leaky_relu'):
      self.assertIsInstance(layer, tf.keras.layers.LeakyReLU)

    layer = network_utils.get_activation('relu')
    with self.subTest(name='relu'):
      self.assertIsInstance(layer, tf.keras.layers.ReLU)

    layer = network_utils.get_activation('sin')
    with self.subTest(name='sin'):
      self.assertIsInstance(layer, tf.keras.layers.Activation)

  def test_get_normalization(self):
    layer = network_utils.get_normalization(None)
    with self.subTest(name='dummy'):
      self.assertIs(layer, None)

    layer = network_utils.get_normalization('batch_norm')
    with self.subTest(name='batch_norm'):
      self.assertIsInstance(layer, tf.keras.layers.BatchNormalization)

    layer = network_utils.get_normalization('instance_norm')
    with self.subTest(name='instance_norm'):
      self.assertIsInstance(layer, tfa_layers.InstanceNormalization)

  @parameterized.parameters((9), ((8, 9),))
  def test_fully_connected_net(self, num_filters):
    activation_params = None
    out_spatial_dims = (3, 3)
    layer = network_utils.FullyConnectedNet(num_filters, activation_params,
                                            out_spatial_dims)

    x = tf.zeros((2, 5), dtype=tf.float32)
    output = layer(x, False)
    self.assertSequenceEqual(output.shape, (2, *out_spatial_dims, 1))

  @parameterized.parameters(('random'), ('input_nonzeros'), ('none'),
                            ('right_half'))
  def test_masking_layer(self, mode):
    layer = network_utils.MaskingLayer(mode=mode)

    x = tf.zeros((2, 16, 8, 5), dtype=tf.float32)
    output, mask = layer(x, mask=None, training=False)
    with self.subTest(name='2D'):
      self.assertSequenceEqual(output.shape, x.shape)
      self.assertSequenceEqual(mask.shape[:-1], x.shape[:-1])

    x = tf.zeros((2, 16, 8, 4, 5), dtype=tf.float32)
    output, mask = layer(x, mask=None, training=False)
    with self.subTest(name='3D'):
      self.assertSequenceEqual(output.shape, x.shape)
      self.assertSequenceEqual(mask.shape[:-1], x.shape[:-1])

  def test_masking_layer_input_mask(self):
    layer = network_utils.MaskingLayer(
        mode='input_mask', resize_mode='upsample', resize_factor=1)
    x = tf.ones((2, 16, 8, 4, 5), dtype=tf.float32)
    input_mask = tf.ones((2, 16, 8, 4, 1), dtype=tf.float32)
    output, mask = layer(x, mask=input_mask, training=False)
    with self.subTest(name='upsample_1'):
      self.assertSequenceEqual(output.shape, x.shape)
      self.assertSequenceEqual(mask.shape[:-1], x.shape[:-1])

    layer = network_utils.MaskingLayer(
        mode='input_mask', resize_mode='upsample', resize_factor=2)
    x = tf.ones((2, 16, 8, 4, 5), dtype=tf.float32)
    input_mask = tf.ones((2, 8, 4, 2, 1), dtype=tf.float32)
    output, mask = layer(x, mask=input_mask, training=False)
    with self.subTest(name='upsample_2'):
      self.assertSequenceEqual(output.shape, x.shape)
      self.assertAllEqual(output, x)
      self.assertSequenceEqual(mask.shape[:-1], x.shape[:-1])

    layer = network_utils.MaskingLayer(
        mode='input_mask', resize_mode='downsample', resize_factor=2)
    x = tf.ones((2, 16, 8, 4, 5), dtype=tf.float32)
    input_mask = tf.ones((2, 32, 16, 8, 1), dtype=tf.float32)
    output, mask = layer(x, mask=input_mask, training=False)
    with self.subTest(name='upsample_2'):
      self.assertSequenceEqual(output.shape, x.shape)
      self.assertAllEqual(output, x)
      self.assertSequenceEqual(mask.shape[:-1], x.shape[:-1])

  @parameterized.parameters(
      ({
          'noise_config': {
              'masked/apply': True,
              'masked/mode': 'add',
              'masked/std': 0.1
          }
      }),
      ({
          'noise_config': {
              'masked/apply': True,
              'masked/mode': 'multiply',
              'masked/std': 0.1
          }
      }),
      ({
          'noise_config': {
              'unmasked/apply': True,
              'unmasked/mode': 'add',
              'unmasked/std': 0.1
          }
      }),
      ({
          'noise_config': {
              'unmasked/apply': True,
              'unmasked/mode': 'multiply',
              'unmasked/std': 0.1
          }
      }),
  )
  def test_masking_layer_add_noises(self, noise_config):
    layer = network_utils.MaskingLayer(noise_config=noise_config)

    x = tf.ones((2, 16, 8, 5), dtype=tf.float32)
    output, mask = layer(x, mask=None, training=False)
    self.assertSequenceEqual(output.shape, x.shape)
    self.assertSequenceEqual(mask.shape[:-1], x.shape[:-1])

  def test_masking_layer_update_config(self):
    layer = network_utils.MaskingLayer()
    update = {
        'mode': 'random',
        'offset': (1, 1),
        'masked_value': 0.5,
        'dropout_rate': 0.2,
        'dropout_rescale': True,
        'noise_config': {
            'masked/apply': True,
            'masked/mode': 'add',
            'masked/std': 0.1
        }
    }
    layer.update_config(update)
    self.assertEqual(layer.mode, update['mode'])
    self.assertEqual(layer.offset, update['offset'])
    self.assertEqual(layer.masked_value, update['masked_value'])
    self.assertEqual(layer.dropout_rate, update['dropout_rate'])
    self.assertEqual(layer.dropout_rescale, update['dropout_rescale'])
    self.assertEqual(layer.noise_config, update['noise_config'])

  @parameterized.parameters(('concat', (4, 8)), ('mode1', (4, 4)),
                            ('mode2', (4, 8)))
  def test_fusion_net(self, mode, expected_dim_c):
    net = network_utils.FusionNet(
        mode=mode, num_filter=[[], [4]], kernel_size=[[], [3]])
    out_dec = (tf.zeros((2, 1, 1, 1, 4), dtype=tf.float32),
               tf.zeros((2, 2, 2, 2, 4), dtype=tf.float32))
    out_enc = (tf.zeros((2, 1, 1, 1, 4), dtype=tf.float32),
               tf.zeros((2, 2, 2, 2, 4), dtype=tf.float32))
    dropout_mask = (tf.ones((2, 1, 1, 1, 4), dtype=tf.float32),
                    tf.ones((2, 2, 2, 2, 4), dtype=tf.float32))
    out_fuse = net(out_dec, out_enc, dropout_mask, training=True)

    for out_fuse_i, out_dec_i, expected_dim_c_i in zip(out_fuse, out_dec,
                                                       expected_dim_c):
      self.assertSequenceEqual(out_fuse_i.shape,
                               (*out_dec_i.shape[:-1], expected_dim_c_i))


if __name__ == '__main__':
  tf.test.main()
