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

"""Utility functions and modules for network."""

from typing import Any, Dict, Sequence, Tuple, Union

import tensorflow as tf
import tensorflow_addons.layers as tfa_layers


def get_activation(activation_type: str,
                   params: Dict[str, Any] = None) -> tf.keras.layers.Layer:
  """A wrapper for creating various types of activation layer.

  Args:
    activation_type: type of activation, one of ['leaky_relu', 'relu', 'sin'].
    params: parameters for the activation layer.

  Returns:
    activation layer.
  """
  if params is None:
    params = {}
  if activation_type is None:
    return tf.keras.layers.LeakyReLU(alpha=0.2)
  elif activation_type == 'leaky_relu':
    return tf.keras.layers.LeakyReLU(**params)
  elif activation_type == 'relu':
    return tf.keras.layers.ReLU(**params)
  elif activation_type == 'sin':
    return tf.keras.layers.Activation(tf.math.sin)
  else:
    raise ValueError('Unknown activation_type: %s' % activation_type)


def get_normalization(normalization_type: str,
                      params: Dict[str, Any] = None) -> Any:
  """A wrapper for creating various types of normalization layer.

  Args:
    normalization_type: type of normalization, one of ['batch_norm',
      'instance_norm'].
    params: parameters for the normalization layer.

  Returns:
    normalization layer.
  """
  if params is None:
    params = {}
  if normalization_type is None:
    return None
  elif normalization_type == 'batch_norm':
    return tf.keras.layers.BatchNormalization(**params)
  elif normalization_type == 'instance_norm':
    return tfa_layers.InstanceNormalization(**params)
  else:
    raise ValueError('Unknown normalization_type: %s' % normalization_type)


class FullyConnectedNet(tf.keras.Model):
  """Fully connected network."""

  def __init__(self,
               num_filters: Union[int, Sequence[int]] = 4,
               activation_params: Dict[str, Any] = None,
               out_spatial_dims: Sequence[int] = (1, 1),
               name: str = 'FullyConnectedNet') -> None:
    """Initialization function.

    Args:
      num_filters: number of filters for each layer.
      activation_params: parameters for activation layers.
      out_spatial_dims: spatial dimensions for reshaping output into.
      name: name of the model.
    """
    super(FullyConnectedNet, self).__init__(name=name)

    if isinstance(num_filters, int):
      num_filters = (num_filters,)
    elif isinstance(num_filters, Sequence):
      pass
    else:
      raise TypeError('num_filters must be either int or list of int.')

    if isinstance(activation_params, dict) and 'type' in activation_params:
      activation_params = dict(activation_params)
      activation_type = activation_params.pop('type')
    else:
      activation_type = None

    self.out_spatial_dims = out_spatial_dims
    self._layers = []

    for layer_index in range(len(num_filters)):
      dense = tf.keras.layers.Dense(num_filters[layer_index])
      self._layers.append(dense)
      activation = get_activation(activation_type, activation_params)
      self._layers.append(activation)

  def call(self, inputs: tf.Tensor, training: bool) -> tf.Tensor:
    """Forward method.

    Args:
      inputs: [batch_size, ...] tensor, input.
      training: flag indicating training phase.

    Returns:
      output: [batch_size, *out_spatial_dims, dim_c] tensor, output.
        out_spatial_dims is the spatial dimensions of output, dim_c is the
        channel dimension.
    """
    # Flatten input to a feature vector.
    output = tf.reshape(inputs, (inputs.shape[0], -1))
    # Apply layers.
    for layer in self._layers:
      output = layer(output)
    # Reshape output.
    output = tf.reshape(output, (output.shape[0], *self.out_spatial_dims, -1))

    return output


class MaskingLayer(tf.keras.Model):
  """Masking layer."""

  def __init__(self,
               mode: str = 'right_half',
               offset: Sequence[int] = (0, 0),
               masked_value: float = 0,
               dropout_rate: float = 0.5,
               dropout_rescale: bool = False,
               resize_mode: str = 'upsample',
               resize_factor: int = 1,
               noise_config: Dict[str, Any] = None,
               name: str = 'MaskingLayer') -> None:
    """Initialization function.

    Args:
      mode: mode for masking.
      offset: offset for masking, used when mode is 'right_half'.
      masked_value: value for masked region, used when mode is 'right_half'.
      dropout_rate: rate for dropout, used when mode is 'random'.
      dropout_rescale: whether revert the scaling of dropout, used when mode is
        'random'.
      resize_mode: mode for resizing input mask, used when mode is 'input_mask'.
      resize_factor: factor for resizing input mask, used when mode is
        'input_mask'.
      noise_config: configuration on adding noises.
      name: name of the model.
    """
    super(MaskingLayer, self).__init__(name=name)

    self.mode = mode
    self.offset = offset
    self.masked_value = masked_value
    self.dropout_rate = dropout_rate
    self.dropout_rescale = dropout_rescale
    self.resize_mode = resize_mode
    self.resize_factor = resize_factor
    if resize_mode == 'upsample':
      self._resize_layer = tf.keras.layers.UpSampling3D(self.resize_factor)
    else:
      self._resize_layer = tf.keras.layers.AveragePooling3D(self.resize_factor)
    self.noise_config = noise_config

  def update_config(self, update):
    """Function for updating the model configuration."""
    if 'mode' in update:
      self.mode = update['mode']
    if 'offset' in update:
      self.offset = update['offset']
    if 'masked_value' in update:
      self.masked_value = update['masked_value']
    if 'dropout_rate' in update:
      self.dropout_rate = update['dropout_rate']
    if 'dropout_rescale' in update:
      self.dropout_rescale = update['dropout_rescale']
    if 'noise_config' in update:
      self.noise_config = update['noise_config']

  def call(self, inputs: tf.Tensor, mask: tf.Tensor,
           training: bool) -> Tuple[tf.Tensor, tf.Tensor]:
    """Forward method.

    Args:
      inputs: [batch_size, [dim_d], dim_h, dim_w, dim_c] tensor, input.
      mask: [batch_size, [dim_d], dim_h, dim_w, 1 or dim_c] tensor, input mask.
      training: flag indicating training phase.

    Returns:
      output: [batch_size, [dim_d], dim_h, dim_w, dim_c] tensor, output.
      mask: [batch_size, [dim_d], dim_h, dim_w, 1 or dim_c] tensor, actual mask
        that is used for masking.
    """
    if len(inputs.shape) == 4:
      batch_size, dim_h, dim_w, dim_c = inputs.shape
      spatial_dims = (dim_h, dim_w)
    elif len(inputs.shape) == 5:
      batch_size, dim_d, dim_h, dim_w, dim_c = inputs.shape
      spatial_dims = (dim_d, dim_h, dim_w)
    else:
      raise NotImplementedError(
          'MaskingLayer: only supports inputs with 2D/3D spatial dimensions')

    # Mask regions.
    if self.mode == 'random':
      mask = tf.ones((batch_size, *spatial_dims, 1))
      mask = tf.nn.dropout(
          mask, self.dropout_rate, noise_shape=(batch_size, *spatial_dims, 1))
      if self.dropout_rescale:
        mask = tf.clip_by_value(mask, 0., 1.)
      output = inputs * mask
    elif self.mode == 'input_mask':
      if self.resize_factor != 1:
        mask = self._resize_layer(mask)
      output = inputs * mask
    elif self.mode == 'input_nonzeros':
      mask = tf.cast(inputs != 0, dtype=tf.float32)
      output = inputs
    elif self.mode == 'none':
      mask = tf.ones((batch_size, *spatial_dims, 1))
      output = inputs
    else:
      # Get mask (masked region is 1).
      if self.mode == 'right_half':
        x_right = int(dim_w / 2 + self.offset[0])
        mask = tf.concat(
            (tf.zeros((batch_size, *spatial_dims[0:-1], x_right)),
             tf.ones((batch_size, *spatial_dims[0:-1], dim_w - x_right))),
            axis=-1)
      else:
        raise NotImplementedError

      # Apply mask.
      mask_indices = tf.where(mask == 1)
      updates_shape = (tf.shape(mask_indices)[0], dim_c)
      updates = tf.ones(updates_shape) * self.masked_value
      output = tf.tensor_scatter_nd_update(inputs, mask_indices, updates)

      # Invert mask so that masked region is 0.
      mask = (tf.ones_like(mask) - mask)[..., None]

    # Add noises.
    mask_binary = tf.clip_by_value(mask, 0., 1.)
    noise_config = self.noise_config
    if noise_config is not None:
      output_masked = output
      output_unmasked = output
      if 'masked/apply' in noise_config and noise_config['masked/apply']:
        if noise_config['masked/mode'] == 'add':
          noise = tf.random.normal(
              tf.shape(output_masked),
              mean=0.0,
              stddev=noise_config['masked/std'])
          output_masked = output_masked + noise
        elif noise_config['masked/mode'] == 'multiply':
          noise = tf.random.normal(
              tf.shape(output_masked),
              mean=1.0,
              stddev=noise_config['masked/std'])
          output_masked = output_masked * noise
        else:
          raise ValueError('Unknown noise mode: %s' %
                           noise_config['masked/mode'])
      if 'unmasked/apply' in noise_config and noise_config['unmasked/apply']:
        if noise_config['unmasked/mode'] == 'add':
          noise = tf.random.normal(
              tf.shape(output_unmasked),
              mean=0.0,
              stddev=noise_config['unmasked/std'])
          output_unmasked = output_unmasked + noise
        elif noise_config['unmasked/mode'] == 'multiply':
          noise = tf.random.normal(
              tf.shape(output_unmasked),
              mean=1.0,
              stddev=noise_config['unmasked/std'])
          output_unmasked = output_unmasked * noise
        else:
          raise ValueError('Unknown noise mode: %s' %
                           noise_config['unmasked/mode'])

      output_masked = output_masked * (tf.ones_like(mask_binary) - mask_binary)
      output_unmasked = output_unmasked * mask_binary
      output = output_masked + output_unmasked

    return output, mask


class FusionNet(tf.keras.Model):
  """Fusion network for fusing multilevel outputs from encoder and decoder parts."""

  def __init__(self,
               mode: str = 'concat',
               num_filter: Union[int, Sequence[Sequence[int]]] = 4,
               kernel_size: Union[int, Sequence[Sequence[int]]] = 3,
               activation_params: Dict[str, Any] = None,
               name: str = 'FusionNet'):
    """Initialization function.

    Args:
      mode: mode for fusion.
      num_filter: number of filters for each conv at each level.
      kernel_size: kernel size for each conv at each level.
      activation_params: parameters for activation layers.
      name: name of the model.
    """
    super(FusionNet, self).__init__(name=name)

    self._mode = mode

    if isinstance(num_filter, int):
      num_filter = [[num_filter]]

    if isinstance(kernel_size, int):
      kernel_size = [[kernel_size]]

    if isinstance(activation_params, dict) and 'type' in activation_params:
      activation_params = dict(activation_params)
      activation_type = activation_params.pop('type')
    else:
      activation_type = None

    conv_layer_class = tf.keras.layers.Conv3D

    num_level = len(num_filter)
    self._layers = [[] for _ in range(num_level)]
    for level in range(num_level):
      for i in range(len(num_filter[level])):
        conv = conv_layer_class(
            filters=num_filter[level][i],
            kernel_size=kernel_size[level][i],
            strides=1,
            padding='same',
        )
        self._layers[level].append(conv)
        activation = get_activation(activation_type, activation_params)
        self._layers[level].append(activation)

  def call(self, out_dec: Sequence[tf.Tensor], out_enc: Sequence[tf.Tensor],
           dropout_mask: Sequence[tf.Tensor],
           training: bool) -> Sequence[tf.Tensor]:
    """Forward method.

    Args:
      out_dec: outputs from decoder part at each level.
      out_enc: outputs from encoder part at each level.
      dropout_mask: mask for dropout at each level.
      training: flag indicating training phase.

    Returns:
      out: fused outputs at each level.
    """
    num_level = len(out_enc)
    out = [out_dec[0]]
    for level in range(1, num_level):
      if self._mode == 'concat':
        out_i = tf.concat([out_dec[level], out_enc[level]], axis=-1)
      elif self._mode == 'mode1':
        out_i = tf.concat([out_dec[level], out_enc[level]], axis=-1)
        for layer in self._layers[level]:
          out_i = layer(out_i)
      elif self._mode == 'mode2':
        out_i = tf.concat([out_dec[level], out_enc[level]], axis=-1)
        for layer in self._layers[level]:
          out_i = layer(out_i)
        mask_binary = tf.clip_by_value(dropout_mask[level], 0., 1.)
        out_i = out_i * (tf.ones_like(mask_binary) - mask_binary)
        out_i = out_i + out_enc[level]
        out_i = tf.concat([out_dec[level], out_i], axis=-1)
      out.append(out_i)

    return out
