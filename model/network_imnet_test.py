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

"""Tests for google3.vr.perception.volume_compression.mdif.model.network_imnet."""

import tensorflow as tf

from google3.vr.perception.volume_compression.mdif.model import network_imnet


class NetworkImnetTest(tf.test.TestCase):

  def test_imnet(self):
    batch_size = 2
    num_out_channel = 1
    layer = network_imnet.ImNet(num_out_channel=num_out_channel)
    x = tf.zeros((batch_size, 131), dtype=tf.float32)
    output = layer(x)

    self.assertSequenceEqual(output.shape, (batch_size, num_out_channel))


if __name__ == '__main__':
  tf.test.main()
