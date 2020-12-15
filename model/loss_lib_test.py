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

"""Tests for google3.vr.perception.volume_compression.mdif.model.loss_lib."""

from absl.testing import parameterized
import tensorflow as tf

from google3.vr.perception.volume_compression.mdif.model import loss_lib


class LossLibTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(('loss'), ('metric'))
  def test_l1_loss(self, mode):
    loss = loss_lib.L1Loss()
    self.assertAllEqual(
        loss(
            tf.ones((1, 3, 2, 3), dtype=tf.float32),
            tf.ones((1, 3, 2, 3), dtype=tf.float32),
            mode=mode), 0)

  @parameterized.parameters(('loss'), ('metric'))
  def test_l2_loss(self, mode):
    loss = loss_lib.L2Loss()
    self.assertAllEqual(
        loss(
            tf.ones((1, 3, 2, 3), dtype=tf.float32),
            tf.ones((1, 3, 2, 3), dtype=tf.float32),
            mode=mode), 0)

  @parameterized.parameters(('loss'), ('metric'))
  def test_root_feature_reg_loss(self, mode):
    model_params = {}
    loss_params = {
        'root_feat_reg_l1': {
            'term_weight': [1e0]
        },
        'root_feat_reg_l2': {
            'term_weight': [1e0]
        },
    }
    loss_function = loss_lib.MultiresDeepImplicitLoss(model_params, loss_params)

    model_outputs_and_targets = {
        'training': True,
        'image_size': (8, 8),
        'root_feature': tf.zeros((2, 8, 8, 4), dtype=tf.float32)
    }
    flags = None
    total_loss = loss_function(model_outputs_and_targets, mode,
                               flags)['total_loss']

    self.assertAllClose(total_loss, 0, atol=1e-04)

  @parameterized.parameters(('loss'), ('metric'))
  def test_code_grid_reg_loss(self, mode):
    model_params = {}
    loss_params = {
        'code_grid_reg_l2': {
            'term_weight': [1e0, 1e0]
        },
    }
    loss_function = loss_lib.MultiresDeepImplicitLoss(model_params, loss_params)

    model_outputs_and_targets = {
        'training': True,
        'image_size': (8, 8),
        'code_grid/level0': (tf.zeros((2, 1, 1, 4), dtype=tf.float32), 0),
        'code_grid/level1': (tf.zeros((2, 2, 2, 4), dtype=tf.float32), 1),
    }
    flags = None
    total_loss = loss_function(model_outputs_and_targets, mode,
                               flags)['total_loss']

    self.assertAllClose(total_loss, 0, atol=1e-04)

  @parameterized.parameters(('loss'), ('metric'))
  def test_sdf_loss(self, mode):
    model_params = {}
    loss_params = {
        'sdf_l1': {
            'term_weight': [1e0, 1e0]
        },
        'sdf_reg_l1': {
            'term_weight': [1e0, 1e0]
        },
    }
    loss_function = loss_lib.MultiresDeepImplicitLoss(model_params, loss_params)

    model_outputs_and_targets = {
        'training':
            True,
        'image_size': (8, 8),
        'points_sdf/level0': (tf.zeros(
            (2, 64, 1), dtype=tf.float32), tf.zeros(
                (2, 64, 1), dtype=tf.float32), 0, 'loss/sdf'),
        'points_sdf/level1': (tf.zeros(
            (2, 64, 1), dtype=tf.float32), tf.zeros(
                (2, 64, 1), dtype=tf.float32), 1, 'loss/sdf')
    }
    flags = None
    total_loss = loss_function(model_outputs_and_targets, mode,
                               flags)['total_loss']

    self.assertAllClose(total_loss, 0, atol=1e-04)

  @parameterized.parameters(('loss'), ('metric'))
  def test_sdf_metric(self, mode):
    model_params = {}
    loss_params = {
        'sdf_l1': {
            'term_weight': [1e0, 1e0]
        },
        'sdf_reg_l1': {
            'term_weight': [1e0, 1e0]
        },
    }
    loss_function = loss_lib.MultiresDeepImplicitLoss(model_params, loss_params)

    model_outputs_and_targets = {
        'training':
            True,
        'image_size': (8, 8),
        'points_sdf/all_pixels/level0': (tf.zeros(
            (2, 64, 1), dtype=tf.float32), tf.zeros(
                (2, 64, 1), dtype=tf.float32), 0, 'metric/sdf'),
        'points_sdf/all_pixels/level1': (tf.zeros(
            (2, 64, 1), dtype=tf.float32), tf.zeros(
                (2, 64, 1), dtype=tf.float32), 1, 'metric/sdf'),
        'mask_for_point':
            tf.concat((tf.ones(
                (2, 32), dtype=tf.float32), tf.zeros(
                    (2, 32), dtype=tf.float32)),
                      axis=-1)
    }
    flags = None
    loss_summaries = loss_function(model_outputs_and_targets, mode,
                                   flags)['loss_summaries']

    for _, item in loss_summaries.items():
      self.assertAllClose(item, 0, atol=1e-04)

  @parameterized.parameters(('loss'), ('metric'))
  def test_consistency_loss(self, mode):
    model_params = {}
    loss_params = {
        'sdf_consistency_l1': {
            'stop_grad_ref': True,
            'point_weight_config/dist_to_visible': ['gaussian', 0.1],
            'term_weight': [0e0, 1e0]
        }
    }
    loss_function = loss_lib.MultiresDeepImplicitLoss(model_params, loss_params)

    model_outputs_and_targets = {
        'training':
            True,
        'image_size': (8, 8),
        'points_consistency/level0': (tf.zeros(
            (2, 64, 1), dtype=tf.float32), tf.ones(
                (2, 64, 1),
                dtype=tf.float32), 0, 'gt_residual', 'loss/consistency'),
        'points_consistency/level1': (tf.zeros(
            (2, 64, 1), dtype=tf.float32), tf.ones(
                (2, 64, 1),
                dtype=tf.float32), 1, 'gt_residual', 'loss/consistency'),
    }
    flags = {'consistency_loss': True}
    total_loss = loss_function(model_outputs_and_targets, mode,
                               flags)['total_loss']

    self.assertAllClose(total_loss, 0, atol=1e-04)

  @parameterized.parameters(('loss'), ('metric'))
  def test_symmetry_loss(self, mode):
    model_params = {}
    loss_params = {
        'sdf_symmetry_l1': {
            'stop_grad_ref': True,
            'point_weight_config/dist_to_visible': ['constant'],
            'point_weight_config/global_prior': ['constant'],
            'term_weight': [0e0, 1e0]
        }
    }
    loss_function = loss_lib.MultiresDeepImplicitLoss(model_params, loss_params)

    model_outputs_and_targets = {
        'training':
            True,
        'image_size': (8, 8),
        'points_symmetry/level0': (tf.ones(
            (2, 64, 1), dtype=tf.float32), tf.ones(
                (2, 64, 1), dtype=tf.float32), 0, 'gt_residual', 'gt_residual',
                                   'loss/symmetry'),
        'points_symmetry/level1': (tf.ones(
            (2, 64, 1), dtype=tf.float32), tf.ones(
                (2, 64, 1), dtype=tf.float32), 1, 'gt_residual', 'gt_residual',
                                   'loss/symmetry'),
        'points_residual_sdf/level0': (tf.ones(
            (2, 64, 1), dtype=tf.float32), tf.ones((2, 64, 1),
                                                   dtype=tf.float32), 0, ''),
        'points_residual_sdf/level1': (tf.ones(
            (2, 64, 1), dtype=tf.float32), tf.ones((2, 64, 1),
                                                   dtype=tf.float32), 1, ''),
    }
    flags = {'symmetry_loss': True}
    total_loss = loss_function(model_outputs_and_targets, mode,
                               flags)['total_loss']

    self.assertAllClose(total_loss, 0, atol=1e-04)


if __name__ == '__main__':
  tf.test.main()
