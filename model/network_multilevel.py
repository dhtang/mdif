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

"""Network library defining multi-level modules."""

from typing import Any, Dict, Sequence, Union

import tensorflow as tf

from google3.vr.perception.volume_compression.mdif.model import network_autoencoder
from google3.vr.perception.volume_compression.mdif.model import network_imnet
from google3.vr.perception.volume_compression.mdif.model import network_utils


class MultiLevelImplicitDecoder(tf.keras.Model):
  """Decodes multi-level latent codes into SDF."""

  def __init__(self,
               num_level: int = 3,
               num_filter: Union[int, Sequence[int]] = 128,
               num_out_channel: int = 1,
               implicit_net_type: str = 'imnet',
               share_net_level_groups: Sequence[Sequence[int]] = None,
               activation_params: Dict[str, Any] = None,
               name: str = 'MultiLevelImplicitDecoder'):
    """Initialization function.

    Args:
      num_level: number of hierarchy levels.
      num_filter: base number of filters for implicit decoder at each level.
      num_out_channel: number of output channels.
      implicit_net_type: type of implicit decoder, only support 'imnet' for now.
        'imnet' means using the ImNet implicit decoder.
      share_net_level_groups: groups of levels that share implicit decoder,
        e.g., [[0, 1], [3, 4]] means level 0, 1 share network, and level 3, 4
        share network. Will override network settings for each level.
      activation_params: parameters for activation layers. Must include a 'type'
        keyword to indicate the type of the activation, one of ['leaky_relu',
        'relu', 'sin']. Other parameters are those specifically for the layer.
      name: name of the model.
    """
    super(MultiLevelImplicitDecoder, self).__init__(name=name)
    self.num_level = num_level

    if isinstance(num_filter, int):
      num_filter = [num_filter for i in range(self.num_level)]
    elif isinstance(num_filter, Sequence):
      assert len(num_filter) >= self.num_level
    else:
      raise TypeError('num_filter must be either int or list of int.')
    self.num_filter = num_filter

    self.num_out_channel = num_out_channel
    self.implicit_net_type = implicit_net_type

    if share_net_level_groups is None:
      share_net_level_groups = []
    elif share_net_level_groups == 'all':
      share_net_level_groups = [list(range(self.num_level))]
    self.share_net_level_groups = share_net_level_groups

    # Set up deep implicit functions
    self.implicit_nets = [None for i in range(self.num_level)]
    # Add nets that are shared across levels
    for level_group in self.share_net_level_groups:
      if len(level_group) < 1:
        continue
      level0 = level_group[0]
      if self.implicit_net_type == 'imnet':
        self.implicit_nets[level0] = network_imnet.ImNet(
            num_out_channel=self.num_out_channel,
            num_filter=self.num_filter[level0],
            activation_params=activation_params,
            name='ImNet_' + str(level0))
      else:
        raise ValueError('Unknown implicit_net_type: %s' %
                         self.implicit_net_type)
      for level in level_group[1:]:
        self.implicit_nets[level] = self.implicit_nets[level0]
    # Add nets that are not shared
    for i in range(self.num_level):
      if self.implicit_nets[i] is not None:
        continue
      elif self.implicit_net_type == 'imnet':
        self.implicit_nets[i] = network_imnet.ImNet(
            num_out_channel=self.num_out_channel,
            num_filter=self.num_filter[i],
            activation_params=activation_params,
            name='ImNet_' + str(i))
      else:
        raise ValueError('Unknown implicit_net_type: %s' %
                         self.implicit_net_type)

  def call(self,
           inputs: Sequence[tf.Tensor],
           levels: Sequence[int] = None,
           training: bool = False) -> Sequence[tf.Tensor]:
    """Forward method.

    Args:
      inputs: each with shape [batch_size, num_point, num_in_channel], feature
        vectors to be decoded at each selected level.
      levels: selected levels to be evaluated.
      training: flag indicating training phase.

    Returns:
      out: each with shape [batch_size, num_point, 1], decoded SDF of points at
        selected levels.
    """
    if levels is None:
      levels = list(range(self.num_level))

    # Get SDF at selected levels
    out = []
    for level_i, input_i in zip(levels, inputs):
      out_i = self.implicit_nets[level_i](input_i, training=training)
      out.append(out_i)

    return out


class GeometryNet(tf.keras.Model):
  """Generates latent code from feature vector."""

  def __init__(self,
               num_feature_channel: int = 512,
               num_code_channel: int = 32,
               num_filter: Sequence[int] = None,
               activation_params: Dict[str, Any] = None,
               name: str = 'GeometryNet'):
    """Initialization function.

    Args:
      num_feature_channel: number of feature channels
      num_code_channel: number of code channels
      num_filter: number of units for each FC layer
      activation_params: parameters for activation layers.
      name: name of the model.
    """
    super(GeometryNet, self).__init__(name=name)
    self.num_feature_channel = num_feature_channel
    self.num_code_channel = num_code_channel

    if isinstance(activation_params, dict) and 'type' in activation_params:
      activation_params = dict(activation_params)
      activation_type = activation_params.pop('type')
    else:
      activation_type = None

    if num_filter is None:
      num_filter = [
          self.num_feature_channel // 2, self.num_feature_channel // 2,
          self.num_code_channel
      ]
    else:
      assert num_filter[-1] == self.num_code_channel
    self.num_filter = num_filter

    self.num_dense_layer = len(self.num_filter)

    # Set up layers
    self._layers = []
    for i in range(self.num_dense_layer):
      self._layers.append(
          tf.keras.layers.Dense(self.num_filter[i], name='dense_' + str(i)))
      if i < self.num_dense_layer - 1:
        self._layers.append(
            network_utils.get_activation(activation_type, activation_params))

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    """Forward method.

    Args:
      inputs: [batch_size, num_feature_channel] tensor.
      training: flag indicating training phase.

    Returns:
      output: [batch_size, num_code_channel] tensor.
    """
    output = inputs
    for layer in self._layers:
      output = layer(output)

    return output


class PartitionNet(tf.keras.Model):
  """Partitions a feature vector into children feature vectors."""

  def __init__(self,
               num_in_channel: int = 544,
               num_out_channel: int = 512,
               num_children: int = 4,
               num_filter: Sequence[int] = None,
               activation_params: Dict[str, Any] = None,
               name: str = 'PartitionNet'):
    """Initialization function.

    Args:
      num_in_channel: number of input channels.
      num_out_channel: number of channels for each children feature vector.
      num_children: number of children nodes.
      num_filter: number of units for each FC layer.
      activation_params: parameters for activation layers.
      name: name of the model.
    """
    super(PartitionNet, self).__init__(name=name)
    self.num_in_channel = num_in_channel
    self.num_out_channel = num_out_channel
    self.num_children = num_children

    if isinstance(activation_params, dict) and 'type' in activation_params:
      activation_params = dict(activation_params)
      activation_type = activation_params.pop('type')
    else:
      activation_type = None

    if num_filter is None:
      num_filter = [
          self.num_in_channel, self.num_out_channel * self.num_children
      ]
    else:
      assert num_filter[-1] == self.num_out_channel * self.num_children
    self._num_filter = num_filter

    self._num_dense_layer = len(self._num_filter)

    # Set up layers
    self._layers = []
    for i in range(self._num_dense_layer):
      self._layers.append(
          tf.keras.layers.Dense(self._num_filter[i], name='dense_' + str(i)))
      if i < self._num_dense_layer - 1:
        self._layers.append(
            network_utils.get_activation(activation_type, activation_params))

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    """Forward method.

    Args:
      inputs: [batch_size, num_in_channel] tensor, input feature vector.
      training: flag indicating training phase.

    Returns:
      output: [batch_size, num_out_channel * num_children] tensor, a large
        feature vector consisting of children feature vectors.
    """
    output = inputs
    for layer in self._layers:
      output = layer(output)

    return output


class FeatureToCodeNet(tf.keras.Model):
  """Generates multi-level latent codes from root feature."""

  def __init__(self,
               block_params: Sequence[Dict[str, Any]],
               fusion_params: Dict[str, Any] = None,
               data_type: str = '2d',
               mode: str = 'separate_branch',
               out_pre_upsample_id: Sequence[int] = None,
               dec_only_apply_mask: bool = False,
               unified_mask_config: Dict[str, Any] = None,
               name: str = 'FeatureToCodeNet'):
    """Initialization function.

    Args:
      block_params: parameters for blocks of each level.
      fusion_params: parameters for fusion network.
      data_type: type of data.
      mode: mode of network.
      out_pre_upsample_id: layer id of pre-upsampling output tensors in decoder.
      dec_only_apply_mask: whether apply mask when only using decoder.
      unified_mask_config: configuration for unified mask across levels.
      name: str, name of the model.
    """
    super(FeatureToCodeNet, self).__init__(name=name)
    self.mode = mode
    self.out_pre_upsample_id = out_pre_upsample_id
    self.dec_only_apply_mask = dec_only_apply_mask
    self.unified_mask_config = unified_mask_config

    if self.mode == 'single_branch':
      self.num_level = 1
    else:
      self.num_level = len(block_params)

    if fusion_params is None:
      fusion_params = {'mode': 'concat'}
    self.fusion_net = network_utils.FusionNet(**fusion_params)

    self._blocks = [[] for _ in range(self.num_level)]
    self.subnet_names = [[] for _ in range(self.num_level)]
    for i in range(self.num_level):
      block_params_i = block_params[i]
      if isinstance(block_params_i, dict):
        self._blocks[i].append(
            network_autoencoder.EncoderTemplate(
                **block_params_i, data_type=data_type, name='block_' + str(i)))
        self.subnet_names[i].append('EncoderTemplate')
      else:
        num_subnet = len(block_params_i)
        for ith_subnet in range(num_subnet):
          subnet_name = block_params_i[ith_subnet][0]
          subnet_params = block_params_i[ith_subnet][1]
          if subnet_name == 'EncoderTemplate':
            self._blocks[i].append(
                network_autoencoder.EncoderTemplate(
                    **subnet_params,
                    data_type=data_type,
                    name='block_' + str(i) + '_subnet_' + str(ith_subnet)))
            self.subnet_names[i].append('EncoderTemplate')
          elif subnet_name == 'DecoderConv':
            self._blocks[i].append(
                network_autoencoder.DecoderConv(
                    **subnet_params,
                    data_type=data_type,
                    name='block_' + str(i) + '_subnet_' + str(ith_subnet)))
            self.subnet_names[i].append('DecoderConv')
          elif subnet_name == 'DecoderFC':
            self._blocks[i].append(
                network_autoencoder.DecoderFC(
                    **subnet_params,
                    name='block_' + str(i) + '_subnet_' + str(ith_subnet)))
            self.subnet_names[i].append('DecoderFC')
          elif subnet_name == 'FullyConnectedNet':
            self._blocks[i].append(
                network_utils.FullyConnectedNet(
                    **subnet_params,
                    name='block_' + str(i) + '_subnet_' + str(ith_subnet)))
            self.subnet_names[i].append('FullyConnectedNet')
          elif subnet_name == 'MaskingLayer':
            self._blocks[i].append(
                network_utils.MaskingLayer(
                    **subnet_params,
                    name='block_' + str(i) + '_subnet_' + str(ith_subnet)))
            self.subnet_names[i].append('MaskingLayer')
          else:
            raise ValueError('Unknown subnet_name: %s' % subnet_name)

  def call(self,
           inputs: Union[tf.Tensor, Sequence[tf.Tensor]],
           levels: Sequence[int] = None,
           training: bool = False,
           decoder_only: bool = False) -> Sequence[tf.Tensor]:
    """Forward method.

    Args:
      inputs: input feature map.
      levels: levels to be evaluated.
      training: flag indicating training phase.
      decoder_only: whether only using decoder.

    Returns:
      out: latent code grids for specified levels.
    """
    if levels is None:
      levels = list(range(self.num_level))

    if isinstance(inputs, Sequence):
      batch_size = tf.shape(inputs[0])[0]
    else:
      batch_size = tf.shape(inputs)[0]

    # Get unified mask.
    unified_mask = None
    if self.unified_mask_config is not None:
      mask_spatial_dims = self.unified_mask_config['spatial_dims']
      mask_dropout_rate = self.unified_mask_config['dropout_rate']
      unified_mask = tf.ones((batch_size, *mask_spatial_dims, 1))
      unified_mask = tf.nn.dropout(
          unified_mask,
          mask_dropout_rate,
          noise_shape=(batch_size, *mask_spatial_dims, 1))
      if self.unified_mask_config['dropout_rescale']:
        unified_mask = tf.clip_by_value(unified_mask, 0., 1.)

    if self.mode == 'separate_branch':
      out = self.forward_separate_branch(inputs, unified_mask, levels, training,
                                         decoder_only)
    elif self.mode == 'single_branch':
      out = self.forward_single_branch(inputs, unified_mask, training,
                                       decoder_only)
    elif self.mode == 'single_dec_branch':
      out = self.forward_single_dec_branch(inputs, unified_mask, training,
                                           decoder_only)
    else:
      raise ValueError('Unknown mode: %s' % self.mode)

    return out

  def forward_separate_branch(
      self,
      inputs: Union[tf.Tensor, Sequence[tf.Tensor]],
      unified_mask: tf.Tensor,
      levels: Sequence[int] = None,
      training: bool = False,
      decoder_only: bool = False) -> Sequence[tf.Tensor]:
    """Forward method when mode is 'separate_branch'.

    Args:
      inputs: input feature map.
      unified_mask: unified mask across levels.
      levels: levels to be evaluated.
      training: flag indicating training phase.
      decoder_only: whether only using decoder.

    Returns:
      out: latent code grids for specified levels.
    """
    out = [tf.zeros((0,)) for _ in range(self.num_level)]

    for level in levels:
      if not decoder_only:
        if not isinstance(inputs, tf.Tensor):
          raise ValueError(
              'Input must be a single tensor when decoder_only is False')
        out_i = inputs
      else:
        if not isinstance(inputs, Sequence):
          raise ValueError(
              'Input must be a sequence of tensor when decoder_only is True')
        out_i = inputs[level]
      for subnet, subnet_name in zip(self._blocks[level],
                                     self.subnet_names[level]):
        if subnet_name == 'DecoderConv':
          out_i, _ = subnet(out_i, training=training)
        else:
          if not decoder_only:
            if subnet_name == 'MaskingLayer':
              out_i, _ = subnet(out_i, unified_mask, training=training)
            else:
              out_i = subnet(out_i, training=training)
      out[level] = out_i

    return out

  def forward_single_branch(self,
                            inputs: Union[tf.Tensor, Sequence[tf.Tensor]],
                            unified_mask: tf.Tensor,
                            training: bool = False,
                            decoder_only: bool = False) -> Sequence[tf.Tensor]:
    """Forward method when mode is 'single_branch'.

    Args:
      inputs: input feature map.
      unified_mask: unified mask across levels.
      training: flag indicating training phase.
      decoder_only: whether only using decoder.

    Returns:
      out: latent code grid for level 0.
    """
    level = 0

    if decoder_only:
      out_i = inputs[0]
    else:
      out_i = inputs

    for subnet, subnet_name in zip(self._blocks[level],
                                   self.subnet_names[level]):
      if subnet_name in ['DecoderConv', 'DecoderFC']:
        out_i, out_pre_upsample_i = subnet(out_i, training=training)
        out = [out_pre_upsample_i[x] for x in self.out_pre_upsample_id]
      else:
        if not decoder_only:
          if subnet_name == 'MaskingLayer':
            out_i, _ = subnet(out_i, unified_mask, training=training)
          else:
            out_i = subnet(out_i, training=training)

    return out

  def forward_single_dec_branch(
      self,
      inputs: Union[tf.Tensor, Sequence[tf.Tensor]],
      unified_mask: tf.Tensor,
      training: bool = False,
      decoder_only: bool = False) -> Sequence[tf.Tensor]:
    """Forward method when mode is 'single_dec_branch'.

    Args:
      inputs: input feature map.
      unified_mask: unified mask across levels.
      training: flag indicating training phase.
      decoder_only: whether only using decoder.

    Returns:
      out: latent code grids for each level.
    """
    # Used to store actual dropout mask used in each level (assume at most 1
    #  masking layer in each level).
    dropout_mask = [None for _ in range(self.num_level)]

    if not decoder_only:
      # Get outputs from encoder.
      out_enc = [[] for _ in range(self.num_level)]
      for level in range(self.num_level):
        out_i = inputs
        for subnet, subnet_name in zip(self._blocks[level],
                                       self.subnet_names[level]):
          if subnet_name != 'DecoderConv':
            if subnet_name == 'MaskingLayer':
              out_i, dropout_mask[level] = subnet(
                  out_i, unified_mask, training=training)
            else:
              out_i = subnet(out_i, training=training)
        out_enc[level] = out_i
    else:
      if not self.dec_only_apply_mask:
        # Skip all subnets in encoder.
        out_enc = inputs
      else:
        # Skip all subnets except masking layers in encoder.
        out_enc = [None for _ in range(len(inputs))]
        for level in range(self.num_level):
          out_i = inputs[level]
          for subnet, subnet_name in zip(self._blocks[level],
                                         self.subnet_names[level]):
            if subnet_name == 'MaskingLayer':
              out_i, dropout_mask[level] = subnet(
                  out_i, unified_mask, training=training)
          out_enc[level] = out_i

    # Get outputs from decoder (should be the last subnet of block 0).
    decoder = self._blocks[0][-1]
    _, out_pre_upsample = decoder(out_enc[0], training=training)
    out_dec = [out_pre_upsample[x] for x in self.out_pre_upsample_id]

    # Fuse outputs of encoder and decoder.
    out = self.fusion_net(out_dec, out_enc, dropout_mask, training=training)

    return out
