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

"""Implementation of the ImNet implicit network."""

from typing import Any, Dict, Sequence

import tensorflow as tf
from google3.vr.perception.volume_compression.mdif.model import network_utils


class ImNet(tf.keras.layers.Layer):
  """ImNet layer keras implementation."""

  def __init__(self,
               num_out_channel: int = 1,
               num_filter: int = 128,
               num_filter_multiplier: Sequence[float] = (16, 8, 4, 2, 1),
               num_concat: int = 4,
               activation_params: Dict[str, Any] = None,
               name: str = 'ImNet') -> None:
    """Initialization.

    Args:
      num_out_channel: number of channel for output.
      num_filter: width of the second to last layer.
      num_filter_multiplier: multiplier on number of filters for non-last
        layers.
      num_concat: number of layer that concats its output with the initial
        input.
      activation_params: parameters for activation layers.
      name: name of the layer.
    """
    super(ImNet, self).__init__(name=name)
    self._num_concat = num_concat

    if isinstance(activation_params, dict) and 'type' in activation_params:
      activation_params = dict(activation_params)
      activation_type = activation_params.pop('type')
    else:
      activation_type = None

    self._layers = []
    for i in range(len(num_filter_multiplier)):
      self._layers.append(tf.keras.layers.Dense(
          num_filter * num_filter_multiplier[i], name='dense_' + str(i + 1)))
      self._layers.append(
          network_utils.get_activation(activation_type, activation_params))
    self._layers.append(tf.keras.layers.Dense(
        num_out_channel, name='dense_' + str(len(num_filter_multiplier) + 1)))

  def call(self, inputs: tf.Tensor, training: bool = False) -> tf.Tensor:
    """Forward method.

    Args:
      inputs: [batch_size, num_in_channel] tensor, input.
      training: bool, flag indicating training phase.
    Returns:
      output: [batch_size, num_out_channel] tensor, output.
    """
    output = inputs
    for i in range(self._num_concat):
      dense = self._layers[i * 2]
      activation = self._layers[i * 2 + 1]
      output = activation(dense(output))
      output = tf.concat([output, inputs], axis=-1)

    for layer in self._layers[self._num_concat * 2:]:
      output = layer(output)

    return output
