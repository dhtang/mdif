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

"""Network modules for traditional autoencoder."""

from typing import Any, Dict, Union, Tuple, Sequence

import tensorflow as tf
from google3.vr.perception.volume_compression.mdif.model import network_utils


class EncoderTemplate(tf.keras.Model):
  """An encoder template with no residual connection."""

  def __init__(self,
               data_type: str = '2d',
               net_type: str = 'fully_conv',
               num_filters: Union[int, Sequence[int]] = 4,
               strides: Union[int, Sequence[int]] = 2,
               num_conv_per_level: Union[int, Sequence[int]] = 2,
               num_out_channel: int = 512,
               kernel_size: Union[int, Sequence[int]] = 3,
               num_levels: int = 3,
               final_pooling: str = 'avg',
               normalization_params: Dict[str, Any] = None,
               activation_params: Dict[str, Any] = None,
               name: str = 'EncoderTemplate'):
    """Initialization.

    Args:
      data_type: type of data, one of ['2d', '3d'].
      net_type: type of network, one of ['fully_conv', 'fully_fc'].
      num_filters: number of filters at each level.
      strides: strides at each level.
      num_conv_per_level: number of convolution at each level.
      num_out_channel: number of output channels.
      kernel_size: kernel size at each level.
      num_levels: number of levels.
      final_pooling: mode for final pooling, one of [None, 'avg', 'flatten',
        'flatten_keepdims']
      normalization_params: parameters for normalization layers.
      activation_params: parameters for activation layers.
      name: name of the model.
    """
    super(EncoderTemplate, self).__init__(name=name)

    self.data_type = data_type
    self.net_type = net_type

    if isinstance(num_filters, int):
      num_filters = [num_filters for i in range(num_levels)]
    elif isinstance(num_filters, list):
      if self.net_type == 'fully_conv':
        assert len(num_filters) >= num_levels
    else:
      raise TypeError('num_filters must be either int or list of int.')

    if isinstance(strides, int):
      strides = [strides for i in range(num_levels)]
    elif isinstance(strides, list):
      assert len(strides) >= num_levels
    else:
      raise TypeError('strides must be either int or list of int.')

    if isinstance(num_conv_per_level, int):
      num_conv_per_level = [num_conv_per_level for i in range(num_levels)]
    elif isinstance(num_conv_per_level, list):
      assert len(num_conv_per_level) >= num_levels
    else:
      raise TypeError('num_conv_per_level must be either int or list of int.')

    if isinstance(kernel_size, int):
      kernel_size = [kernel_size for i in range(num_levels + 1)]
    elif isinstance(kernel_size, list):
      assert len(kernel_size) == num_levels + 1
    else:
      raise TypeError('kernel_size must be either int or list of int.')

    if isinstance(activation_params, dict) and 'type' in activation_params:
      activation_params = dict(activation_params)
      activation_type = activation_params.pop('type')
    else:
      activation_type = None

    if isinstance(normalization_params,
                  dict) and 'type' in normalization_params:
      normalization_type = normalization_params.pop('type')
    else:
      normalization_type = None

    self.final_pooling = final_pooling
    self._layers = []

    if data_type == '2d':
      conv_layer_class = tf.keras.layers.Conv2D
    else:
      conv_layer_class = tf.keras.layers.Conv3D

    if self.net_type == 'fully_conv':
      for level in range(num_levels):
        stride = strides[level]

        if stride is None:
          num_conv = num_conv_per_level[level]
        else:
          num_conv = num_conv_per_level[level] - 1

        self._add_one_level(conv_layer_class, num_conv, num_filters[level],
                            kernel_size[level], stride, normalization_type,
                            normalization_params, activation_type,
                            activation_params)

      # Apply one more convolution to obtain output.
      conv = conv_layer_class(
          filters=num_out_channel,
          kernel_size=kernel_size[num_levels],
          strides=1,
          padding='same',
      )
      self._layers.append(conv)
      normalization = network_utils.get_normalization(normalization_type,
                                                      normalization_params)
      if normalization is not None:
        self._layers.append(normalization)
      activation = network_utils.get_activation(activation_type,
                                                activation_params)
      self._layers.append(activation)

    elif self.net_type == 'fully_fc':
      for num_filter in num_filters:
        dense = tf.keras.layers.Dense(num_filter)
        self._layers.append(dense)
        activation = network_utils.get_activation(activation_type,
                                                  activation_params)
        self._layers.append(activation)

      dense = tf.keras.layers.Dense(num_out_channel)
      self._layers.append(dense)
      activation = network_utils.get_activation(activation_type,
                                                activation_params)
      self._layers.append(activation)

  def _add_one_level(self, conv_layer_class, num_conv, num_filter, kernel_size,
                     stride, normalization_type, normalization_params,
                     activation_type, activation_params):
    """Add layers for one level."""
    # Run num_conv convolutions with stride 1.
    for _ in range(num_conv):
      conv = conv_layer_class(
          filters=num_filter,
          kernel_size=kernel_size,
          strides=1,
          padding='same',
      )
      self._layers.append(conv)
      normalization = network_utils.get_normalization(normalization_type,
                                                      normalization_params)
      if normalization is not None:
        self._layers.append(normalization)
      activation = network_utils.get_activation(activation_type,
                                                activation_params)
      self._layers.append(activation)

    # Downsample with a strided convolution.
    if stride is not None:
      conv = conv_layer_class(
          filters=num_filter,
          kernel_size=kernel_size,
          strides=stride,
          padding='same',
      )
      self._layers.append(conv)
      normalization = network_utils.get_normalization(normalization_type,
                                                      normalization_params)
      if normalization is not None:
        self._layers.append(normalization)
      activation = network_utils.get_activation(activation_type,
                                                activation_params)
      self._layers.append(activation)

  def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    """Forward method.

    Args:
      inputs: input tensor.
      training: flag indicating training phase.

    Returns:
      outputs: output tensor.
    """
    if self.net_type == 'fully_conv':
      outputs = inputs
      for layer in self._layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
          outputs = layer(outputs, training)
        else:
          outputs = layer(outputs)

      if self.final_pooling is not None:
        if self.final_pooling == 'avg':
          # Use average pooling to convert to a feature vector.
          if self.data_type == '2d':
            outputs = tf.math.reduce_mean(outputs, axis=(1, 2))
          else:
            outputs = tf.math.reduce_mean(outputs, axis=(1, 2, 3))
        elif self.final_pooling == 'flatten':
          # Flatten to a feature vector.
          outputs = tf.reshape(outputs, [outputs.shape[0], -1])
        elif self.final_pooling == 'flatten_keepdims':
          if self.data_type == '2d':
            outputs = tf.reshape(outputs, [outputs.shape[0], 1, 1, -1])
          else:
            outputs = tf.reshape(outputs, [outputs.shape[0], 1, 1, 1, -1])
        else:
          raise ValueError('Unknown final_pooling: %s' % self.final_pooling)

    elif self.net_type == 'fully_fc':
      # Flatten inputs to a feature vector.
      outputs = tf.reshape(inputs, [inputs.shape[0], -1])
      # Apply layers.
      for layer in self._layers:
        outputs = layer(outputs)
      # Expand to [batch_size, 1, 1, dim_c].
      outputs = outputs[:, None, None, :]

    return outputs


class DecoderConv(tf.keras.Model):
  """A convolutional decoder with no residual connection."""

  def __init__(self,
               data_type: str = '2d',
               num_filters: Union[int, Sequence[int]] = 4,
               num_conv_per_level: Union[int, Sequence[int]] = 2,
               num_out_channel: int = 1,
               kernel_size: Union[int, Sequence[int]] = 3,
               kernel_size_deconv: Union[int, Sequence[int]] = 4,
               num_levels: int = 3,
               initial_upsample: Tuple[bool, Sequence[int],
                                       str] = (False, 2, 'bilinear'),
               upsample_type: str = 'deconv',
               normalization_params: Dict[str, Any] = None,
               activation_params: Dict[str, Any] = None,
               name: str = 'DecoderConv'):
    """Initialization.

    Args:
      data_type: type of data, one of ['2d', '3d'].
      num_filters: number of filters at each level.
      num_conv_per_level: number of convolution at each level.
      num_out_channel: number of output channels.
      kernel_size: kernel size of convolution at each level.
      kernel_size_deconv: kernel size of deconvolution at each level.
      num_levels: number of levels.
      initial_upsample: configuration for initial upsampling.
      upsample_type: type of upsample layer, one of ['deconv', 'bilinear'].
      normalization_params: parameters for normalization layers.
      activation_params: parameters for activation layers.
      name: name of the model.
    """
    super(DecoderConv, self).__init__(name=name)

    if isinstance(num_filters, int):
      num_filters = [num_filters for i in range(num_levels)]
    elif isinstance(num_filters, list):
      assert len(num_filters) >= num_levels
    else:
      raise TypeError('num_filters must be either int or list of int.')

    if isinstance(num_conv_per_level, int):
      num_conv_per_level = [num_conv_per_level for i in range(num_levels)]
    elif isinstance(num_conv_per_level, list):
      assert len(num_conv_per_level) >= num_levels
    else:
      raise TypeError('num_conv_per_level must be either int or list of int.')

    if isinstance(kernel_size, int):
      kernel_size = [kernel_size for i in range(num_levels)] + [1]
    elif isinstance(kernel_size, list):
      assert len(kernel_size) == num_levels + 1
    else:
      raise TypeError('kernel_size must be either int or list of int.')

    if isinstance(kernel_size_deconv, int):
      kernel_size_deconv = [kernel_size_deconv for i in range(num_levels - 1)]
    elif isinstance(kernel_size_deconv, list):
      assert len(kernel_size_deconv) >= num_levels - 1
    else:
      raise TypeError('kernel_size_deconv must be either int or list of int.')

    if isinstance(activation_params, dict) and 'type' in activation_params:
      activation_params = dict(activation_params)
      activation_type = activation_params.pop('type')
    else:
      activation_type = None

    if isinstance(normalization_params,
                  dict) and 'type' in normalization_params:
      normalization_type = normalization_params.pop('type')
    else:
      normalization_type = None

    self._data_type = data_type
    self._upsample_type = upsample_type
    self._pre_upsample_layer_id = []
    self._layers = []

    if self._data_type == '2d':
      self._conv_layer_class = tf.keras.layers.Conv2D
      self._upsample_layer_class = tf.keras.layers.UpSampling2D
      self._deconv_layer_class = tf.keras.layers.Conv2DTranspose
    elif self._data_type == '3d':
      self._conv_layer_class = tf.keras.layers.Conv3D
      self._upsample_layer_class = tf.keras.layers.UpSampling3D
      self._deconv_layer_class = tf.keras.layers.Conv3DTranspose
    else:
      raise ValueError('Unknown data_type: %s' % self._data_type)

    # Initial upsampling layer.
    if initial_upsample[0]:
      if self._data_type == '2d':
        self._layers.append(
            self._upsample_layer_class(
                size=initial_upsample[1], interpolation=initial_upsample[2]))
      elif self._data_type == '3d':
        self._layers.append(
            self._upsample_layer_class(size=initial_upsample[1]))

    # Layers for each level except the last level.
    for level in range(num_levels - 1):
      self._add_one_level(num_conv_per_level[level], num_filters[level],
                          kernel_size[level], kernel_size_deconv[level],
                          normalization_type, normalization_params,
                          activation_type, activation_params)

    # Refine output in the final resolution.
    if num_out_channel is None:  # No final conv to map output channels.
      num_conv = num_conv_per_level[num_levels - 1]
    else:
      num_conv = num_conv_per_level[num_levels - 1] - 1
    for conv_index in range(num_conv):
      conv = self._conv_layer_class(
          filters=num_filters[num_levels - 1],
          kernel_size=kernel_size[num_levels - 1],
          strides=1,
          padding='same',
      )
      self._layers.append(conv)
      normalization = network_utils.get_normalization(normalization_type,
                                                      normalization_params)
      if normalization is not None:
        self._layers.append(normalization)
      activation = network_utils.get_activation(activation_type,
                                                activation_params)
      self._layers.append(activation)

    # Map to the output number of channels.
    if num_conv_per_level[num_levels - 1] >= 1 and num_out_channel is not None:
      conv = self._conv_layer_class(
          filters=num_out_channel,
          kernel_size=kernel_size[num_levels],
          strides=1,
          padding='same',
      )
      self._layers.append(conv)

    self._pre_upsample_layer_id.append(len(self._layers) - 1)

  def _add_one_level(self, num_conv: int, num_filter: int, kernel_size: int,
                     kernel_size_deconv: int, normalization_type: str,
                     normalization_params: Dict[str, Any], activation_type: str,
                     activation_params: Dict[str, Any]):
    if self._upsample_type == 'deconv':
      num_conv = num_conv - 1

    # Apply several convolutions before upsampling.
    for _ in range(num_conv):
      conv = self._conv_layer_class(
          filters=num_filter,
          kernel_size=kernel_size,
          strides=1,
          padding='same',
      )
      self._layers.append(conv)
      normalization = network_utils.get_normalization(normalization_type,
                                                      normalization_params)
      if normalization is not None:
        self._layers.append(normalization)
      activation = network_utils.get_activation(activation_type,
                                                activation_params)
      self._layers.append(activation)

    self._pre_upsample_layer_id.append(len(self._layers) - 1)

    # Upsample to one level higher resolution.
    if self._upsample_type == 'deconv':
      up = self._deconv_layer_class(
          filters=num_filter,
          kernel_size=kernel_size_deconv,
          strides=2,
          padding='same',
      )
      self._layers.append(up)
      normalization = network_utils.get_normalization(normalization_type,
                                                      normalization_params)
      if normalization is not None:
        self._layers.append(normalization)
      activation = network_utils.get_activation(activation_type,
                                                activation_params)
      self._layers.append(activation)
    elif self._upsample_type == 'bilinear':
      if self._data_type == '2d':
        up = self._upsample_layer_class(size=2, interpolation='bilinear')
      elif self._data_type == '3d':
        up = self._upsample_layer_class(size=2)
      self._layers.append(up)
    else:
      raise ValueError('Unknown upsample_type: %s' % self._upsample_type)

  def call(self, inputs: tf.Tensor,
           training: bool) -> Tuple[tf.Tensor, Sequence[tf.Tensor]]:
    """Forward method.

    Args:
      inputs: input tensor.
      training: flag indicating training phase.

    Returns:
      outputs: final output tensor.
      outputs_pre_upsample: output tensors at each level before upsampling.
    """
    # Expand dims if necessary.
    if len(tf.shape(inputs)) != 4 and len(tf.shape(inputs)) != 5:
      if len(tf.shape(inputs)) == 2:
        if self._data_type == '2d':
          inputs = inputs[:, None, None, :]
        elif self._data_type == '3d':
          inputs = inputs[:, None, None, None, :]
      else:
        raise ValueError('Unknown number of dimensions for inputs',
                         len(tf.shape(inputs)))

    # Go through each layer.
    outputs = inputs
    outputs_pre_upsample = []

    if -1 in self._pre_upsample_layer_id:
      outputs_pre_upsample.append(outputs)

    for layer_id, layer in enumerate(self._layers):
      if isinstance(layer, tf.keras.layers.BatchNormalization):
        outputs = layer(outputs, training=training)
      else:
        outputs = layer(outputs)

      if layer_id in self._pre_upsample_layer_id:
        outputs_pre_upsample.append(outputs)

    return outputs, outputs_pre_upsample


class DecoderFC(tf.keras.Model):
  """A fully connected decoder with no residual connection."""

  def __init__(self,
               num_filters: Sequence[Sequence[int]],
               out_spatial_dims: Sequence[Sequence[int]],
               activation_params: Dict[str, Any] = None,
               name: str = 'DecoderFC'):
    """Initialization.

    Args:
      num_filters: number of filters at each level and each layer.
      out_spatial_dims: spatial dimensions of outputs at each level.
      activation_params: parameters for activation layers.
      name: name of the model.
    """
    super(DecoderFC, self).__init__(name=name)

    num_levels = len(num_filters)
    assert len(out_spatial_dims) == num_levels
    self._out_spatial_dims = out_spatial_dims

    if isinstance(activation_params, dict) and 'type' in activation_params:
      activation_params = dict(activation_params)
      activation_type = activation_params.pop('type')
    else:
      activation_type = None

    self._layers = []

    # Dense layers at each level.
    for level in range(num_levels):
      self._layers.append([])
      for num_filter in num_filters[level]:
        dense = tf.keras.layers.Dense(num_filter)
        self._layers[-1].append(dense)
        activation = network_utils.get_activation(activation_type,
                                                  activation_params)
        self._layers[-1].append(activation)

  def call(self, inputs: tf.Tensor,
           training: bool) -> Tuple[tf.Tensor, Sequence[tf.Tensor]]:
    """Forward method.

    Args:
      inputs: input tensor.
      training: flag indicating training phase.

    Returns:
      outputs: final output tensor.
      outputs_pre_upsample: output tensors at the end of each level.
    """
    # Expand dims if necessary.
    if len(tf.shape(inputs)) != 2:
      if len(tf.shape(inputs)) == 4:
        if inputs.shape[1] != 1 or inputs.shape[2] != 1:
          raise ValueError('Dimension of inputs should be [_, 1, 1, _]')
        inputs = inputs[:, 0, 0, :]
      elif len(tf.shape(inputs)) == 5:
        if inputs.shape[1] != 1 or inputs.shape[2] != 1 or inputs.shape[3] != 1:
          raise ValueError('Dimension of inputs should be [_, 1, 1, 1, _]')
        inputs = inputs[:, 0, 0, 0, :]
      else:
        raise ValueError('Unknown number of dimensions for inputs',
                         len(tf.shape(inputs)))

    # Go through each layer.
    outputs = inputs
    outputs_pre_upsample = []

    for level in range(len(self._layers)):
      for layer in self._layers[level]:
        outputs = layer(outputs)
      outputs_reshape = tf.reshape(
          outputs, [outputs.shape[0], *self._out_spatial_dims[level], -1])
      outputs_pre_upsample.append(outputs_reshape)

    return outputs_reshape, outputs_pre_upsample
