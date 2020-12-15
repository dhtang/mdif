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

"""Tests for google3.vr.perception.volume_compression.mdif.model.network_autoencoder."""

from absl.testing import parameterized
import tensorflow as tf

from google3.vr.perception.volume_compression.mdif.model import network_autoencoder


class NetworkAutoencoderTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters((None), ('avg'), ('flatten'), ('flatten_keepdims'))
  def test_encoder_template_fully_conv_2d(self, final_pooling):
    layer = network_autoencoder.EncoderTemplate(
        data_type='2d',
        net_type='fully_conv',
        num_out_channel=8,
        final_pooling=final_pooling)
    inputs = tf.ones((2, 16, 16, 1), dtype=tf.float32)
    out = layer(inputs)

    if final_pooling is None:
      self.assertSequenceEqual(out.shape, (2, 2, 2, 8))
    elif final_pooling == 'avg':
      self.assertSequenceEqual(out.shape, (2, 8))
    elif final_pooling == 'flatten':
      self.assertSequenceEqual(out.shape, (2, 32))
    elif final_pooling == 'flatten_keepdims':
      self.assertSequenceEqual(out.shape, (2, 1, 1, 32))

  @parameterized.parameters((None, (2, 2, 2, 2, 8)), ('avg', (2, 8)),
                            ('flatten', (2, 64)),
                            ('flatten_keepdims', (2, 1, 1, 1, 64)))
  def test_encoder_template_fully_conv_3d(self, final_pooling,
                                          expected_out_shape):
    layer = network_autoencoder.EncoderTemplate(
        data_type='3d',
        net_type='fully_conv',
        num_out_channel=8,
        final_pooling=final_pooling)
    inputs = tf.ones((2, 16, 16, 16, 1), dtype=tf.float32)
    out = layer(inputs)

    self.assertSequenceEqual(out.shape, expected_out_shape)

  def test_encoder_template_fully_fc(self):
    layer = network_autoencoder.EncoderTemplate(
        net_type='fully_fc', num_out_channel=8)
    inputs = tf.ones((2, 16, 16, 1), dtype=tf.float32)
    out = layer(inputs)

    self.assertSequenceEqual(out.shape, (2, 1, 1, 8))

  @parameterized.parameters(
      ('2d', 1, (False, 2, 'bilinear'), 'deconv', (2, 4, 4, 1),
       ((2, 1, 1, 4), (2, 2, 2, 4), (2, 4, 4, 1))),
      ('2d', None, (False, 2, 'bilinear'), 'deconv', (2, 4, 4, 4),
       ((2, 1, 1, 4), (2, 2, 2, 4), (2, 4, 4, 4))),
      ('2d', 1, (True, 2, 'bilinear'), 'deconv', (2, 8, 8, 1),
       ((2, 2, 2, 4), (2, 4, 4, 4), (2, 8, 8, 1))),
      ('2d', 1, (False, 2, 'bilinear'), 'bilinear', (2, 4, 4, 1),
       ((2, 1, 1, 4), (2, 2, 2, 4), (2, 4, 4, 1))),
  )
  def test_decoder_conv_2d(self, data_type, num_out_channel, initial_upsample,
                           upsample_type, out_shape, out_pre_upsample_shape):
    net = network_autoencoder.DecoderConv(
        data_type=data_type,
        num_out_channel=num_out_channel,
        initial_upsample=initial_upsample,
        upsample_type=upsample_type)
    inputs = tf.ones((2, 1, 1, 16), dtype=tf.float32)
    out, out_pre_upsample = net(inputs)

    with self.subTest('out_shape'):
      self.assertSequenceEqual(out.shape, out_shape)

    with self.subTest('out_pre_upsample_shape'):
      for out_pre_upsample_i, out_pre_upsample_shape_i in zip(
          out_pre_upsample, out_pre_upsample_shape):
        self.assertSequenceEqual(out_pre_upsample_i.shape,
                                 out_pre_upsample_shape_i)

  @parameterized.parameters(
      ('3d', 1, (False, 2, 'bilinear'), 'deconv', (2, 4, 4, 4, 1),
       ((2, 1, 1, 1, 4), (2, 2, 2, 2, 4), (2, 4, 4, 4, 1))),
      ('3d', None, (False, 2, 'bilinear'), 'deconv', (2, 4, 4, 4, 4),
       ((2, 1, 1, 1, 4), (2, 2, 2, 2, 4), (2, 4, 4, 4, 4))),
      ('3d', 1, (True, 2, 'bilinear'), 'deconv', (2, 8, 8, 8, 1),
       ((2, 2, 2, 2, 4), (2, 4, 4, 4, 4), (2, 8, 8, 8, 1))),
      ('3d', 1, (False, 2, 'bilinear'), 'bilinear', (2, 4, 4, 4, 1),
       ((2, 1, 1, 1, 4), (2, 2, 2, 2, 4), (2, 4, 4, 4, 1))),
  )
  def test_decoder_conv_3d(self, data_type, num_out_channel, initial_upsample,
                           upsample_type, out_shape, out_pre_upsample_shape):
    net = network_autoencoder.DecoderConv(
        data_type=data_type,
        num_out_channel=num_out_channel,
        initial_upsample=initial_upsample,
        upsample_type=upsample_type)
    inputs = tf.ones((2, 1, 1, 1, 16), dtype=tf.float32)
    out, out_pre_upsample = net(inputs)

    with self.subTest('out_shape'):
      self.assertSequenceEqual(out.shape, out_shape)

    with self.subTest('out_pre_upsample_shape'):
      for out_pre_upsample_i, out_pre_upsample_shape_i in zip(
          out_pre_upsample, out_pre_upsample_shape):
        self.assertSequenceEqual(out_pre_upsample_i.shape,
                                 out_pre_upsample_shape_i)

  @parameterized.parameters(
      (tf.ones((2, 16), dtype=tf.float32),),
      (tf.ones((2, 1, 1, 16), dtype=tf.float32),),
      (tf.ones((2, 1, 1, 1, 16), dtype=tf.float32),),
  )
  def test_decoder_fc(self, inputs):
    net = network_autoencoder.DecoderFC(
        num_filters=((64, 64), (128, 128)), out_spatial_dims=((2, 2), (4, 4)))
    out, out_pre_upsample = net(inputs)

    with self.subTest('out_shape'):
      self.assertSequenceEqual(out.shape, (2, 4, 4, 8))

    with self.subTest('out_pre_upsample_shape'):
      self.assertSequenceEqual(out_pre_upsample[0].shape, (2, 2, 2, 16))
      self.assertSequenceEqual(out_pre_upsample[1].shape, (2, 4, 4, 8))


if __name__ == '__main__':
  tf.test.main()
