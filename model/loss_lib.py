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

"""Loss functions for multilevel deep implicit function."""

from typing import Any, Dict, Sequence, Tuple

import tensorflow as tf

from google3.vr.perception.volume_compression.mdif.utils import misc_utils

_EPSILON = 1e-6


class L1Loss:
  """An L1 loss class."""

  def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor,
               weight: tf.Tensor = None, mode: str = 'loss') -> tf.Tensor:
    """Calculate the L1 loss.

    Args:
      y_true: Ground truth.
      y_pred: Prediction.
      weight: Per-pixel weight.
      mode: Mode of the loss. Mode 'loss' returns sum over batch for correct
        gradient calculation in tf2. Mode 'metric' returns average over batch.

    Returns:
      L1 loss.

    Raises:
      ValueError: If mode is unknown.
    """
    if weight is None:
      weight = tf.ones_like(y_pred)

    if mode == 'loss':
      reduce_axes = list(range(1, len(y_pred.shape)))
      return tf.reduce_sum(
          tf.reduce_sum(tf.math.abs(y_true - y_pred) * weight,
                        axis=reduce_axes) /
          (tf.reduce_sum(weight, axis=reduce_axes) + _EPSILON))
    elif mode == 'metric':
      return tf.reduce_mean(tf.math.abs(y_true - y_pred) * weight) / (
          tf.reduce_mean(weight) + _EPSILON)
    else:
      raise ValueError('Unknown loss mode: %s' % mode)


class L2Loss:
  """An L2 loss class."""

  def __call__(self, y_true: tf.Tensor, y_pred: tf.Tensor,
               weight: tf.Tensor = None, mode: str = 'loss') -> tf.Tensor:
    """Calculate the L2 loss.

    Args:
      y_true: Ground truth.
      y_pred: Prediction.
      weight: Per-pixel weight.
      mode: Mode of the loss. Mode 'loss' returns sum over batch for correct
        gradient calculation in tf2. Mode 'metric' returns average over batch.

    Returns:
      L2 loss.

    Raises:
      ValueError: If mode is unknown.
    """
    if weight is None:
      weight = tf.ones_like(y_pred)

    if mode == 'loss':
      reduce_axes = list(range(1, len(y_pred.shape)))
      return tf.reduce_sum(
          tf.reduce_sum((y_true - y_pred)**2 * weight, axis=reduce_axes) /
          (tf.reduce_sum(weight, axis=reduce_axes) + _EPSILON))
    elif mode == 'metric':
      return tf.reduce_mean((y_true - y_pred) ** 2 * weight) / (
          tf.reduce_mean(weight) + _EPSILON)
    else:
      raise ValueError('Unknown loss mode: %s' % mode)


class MultiresDeepImplicitLoss:
  """The loss function class for the MultiresDeepImplicitFunction."""

  def __init__(self, model_params: Dict[str, Any],
               loss_params: Dict[str, Any]):
    """Initialize the loss function.

    Args:
      model_params: The parameters for network model.
      loss_params: The parameters of loss functions.
    """
    self._l1_loss = L1Loss()
    self._l2_loss = L2Loss()
    self.loss_params = loss_params

    if 'summary_config' in loss_params:
      self.summary_config = loss_params['summary_config']
    else:
      self.summary_config = None

    if 'eval_data_mode' in model_params:
      self.eval_data_mode = model_params['eval_data_mode']
      self.summary_config = model_params['summary_config']
    else:
      self.eval_data_mode = 'all'

  def compute_point_weight(self, pred: tf.Tensor, gt: tf.Tensor,
                           config: Dict[str, Any]) -> tf.Tensor:
    """Calculate the weight for each point.

    Args:
      pred: Prediction.
      gt: Ground truth.
      config: A dictionary containing configurations.

    Returns:
      The weight for each point.
    """
    if config is None:
      point_weight = tf.ones_like(pred)
    else:
      if config['gt_gaussian']['apply']:
        sigma = config['gt_gaussian']['sigma']
        point_weight_gt = tf.math.exp(- gt ** 2 / 2 / sigma ** 2)
      if config['pred_gaussian']['apply']:
        sigma = config['pred_gaussian']['sigma']
        point_weight_pred = tf.math.exp(- pred ** 2 / 2 / sigma ** 2)

      if config['gt_gaussian']['apply'] and config['pred_gaussian']['apply']:
        point_weight = tf.stack([point_weight_gt, point_weight_pred], axis=-1)
        point_weight = tf.reduce_max(point_weight, axis=-1)
      elif config['gt_gaussian']['apply']:
        point_weight = point_weight_gt
      elif config['pred_gaussian']['apply']:
        point_weight = point_weight_pred
      else:
        point_weight = tf.ones_like(pred)

    return point_weight

  def transform_point_weight(self, points_weight: tf.Tensor,
                             config: Sequence[Any]) -> tf.Tensor:
    """Transform the weight for each point."""
    if config[0] == 'gaussian':
      sigma = config[1]
      points_weight = 1.0 - tf.math.exp(-(points_weight ** 2) /
                                        (2 * sigma ** 2))
    elif config[0] == 'gaussian_no_flip':
      sigma = config[1]
      points_weight = tf.math.exp(-(points_weight ** 2) / (2 * sigma ** 2))
    elif config[0] == 'constant':
      points_weight = tf.ones_like(points_weight)

    return points_weight

  def add_root_feature_reg_loss(self,
                                key: str,
                                item: tf.Tensor,
                                loss_summaries: Dict[str, Any],
                                mode: str = 'loss'):
    """Add regularization losses on root feature."""
    total_loss = 0

    if ('root_feat_reg_l2' in self.loss_params and
        self.loss_params['root_feat_reg_l2']['term_weight'][0] != 0):
      loss = (self._l2_loss(tf.zeros_like(item), item,
                            mode=mode) *
              self.loss_params['root_feat_reg_l2']['term_weight'][0])
      loss_summaries['loss-root_feat-reg/' + key + '/l2'] = loss
      total_loss = total_loss + loss

    if ('root_feat_reg_l1' in self.loss_params and
        self.loss_params['root_feat_reg_l1']['term_weight'][0] != 0):
      loss = (self._l1_loss(tf.zeros_like(item), item, mode=mode) *
              self.loss_params['root_feat_reg_l1']['term_weight'][0])
      loss_summaries['loss-root_feat-reg/' + key + '/l1'] = loss
      total_loss = total_loss + loss

    return total_loss

  def add_code_grid_reg_loss(self,
                             key: str,
                             item: Tuple[tf.Tensor, int],
                             loss_summaries: Dict[str, Any],
                             mode: str = 'loss'):
    """Add regularization losses on code grid."""
    total_loss = 0
    code_grid, level = item

    if ('code_reg_l2' in self.loss_params and
        self.loss_params['code_reg_l2']['term_weight'][level] != 0):
      loss = (self._l2_loss(tf.zeros_like(code_grid), code_grid, mode=mode) *
              self.loss_params['code_reg_l2']['term_weight'][level])
      loss_summaries['loss-code-reg/' + key + '/l2'] = loss
      total_loss = total_loss + loss

    if ('code_reg_l1' in self.loss_params and
        self.loss_params['code_reg_l1']['term_weight'][level] != 0):
      loss = (self._l1_loss(tf.zeros_like(code_grid), code_grid, mode=mode) *
              self.loss_params['code_reg_l1']['term_weight'][level])
      loss_summaries['loss-code-reg/' + key + '/l1'] = loss
      total_loss = total_loss + loss

    return total_loss

  def add_sdf_loss(self,
                   key: str,
                   item: Tuple[tf.Tensor, tf.Tensor, int, str],
                   point_weight: tf.Tensor,
                   loss_summaries: Dict[str, Any],
                   mode: str = 'loss'):
    """Add losses on predicted sdf."""
    total_loss = 0
    sdf_pred, sdf_true, level, _ = item

    # Compute L1 loss on sdf.
    if self.loss_params['sdf_l1']['term_weight'][level] != 0:
      loss = (self._l1_loss(
          sdf_true, sdf_pred, weight=point_weight, mode=mode) *
              self.loss_params['sdf_l1']['term_weight'][level])
      total_loss = total_loss + loss
      loss_summaries['loss/' + key + '/l1_loss'] = loss

    # Compute L1 regularization on sdf.
    if self.loss_params['sdf_reg_l1']['term_weight'][level] != 0:
      loss = (self._l1_loss(tf.zeros_like(sdf_pred), sdf_pred, mode=mode) *
              self.loss_params['sdf_reg_l1']['term_weight'][level])
      total_loss = total_loss + loss
      loss_summaries['loss-sdf-reg/' + key + '/l1'] = loss

    return total_loss

  def add_sdf_metric(self,
                     key: str,
                     item: Tuple[tf.Tensor, tf.Tensor, int, str],
                     point_weight: tf.Tensor,
                     spatial_dims: Sequence[int],
                     loss_summaries: Dict[str, Any],
                     image_summaries: Dict[str, Any],
                     model_outputs_and_targets: Dict[str, Any],
                     mode: str = 'metric'):
    """Add metrics on predicted sdf."""
    sdf_pred, sdf_true, level, _ = item

    # Compute overall metric.
    metric = self._l1_loss(
        sdf_true, sdf_pred, weight=point_weight, mode=mode)
    loss_summaries['metric/' + key + '/l1_loss'] = metric

    if key.split('/')[1] == 'all_pixels':
      # Compute metric for visible and invisible parts.
      mask_for_point = model_outputs_and_targets['mask_for_point'][..., None]
      point_weight_visible = mask_for_point * point_weight
      point_weight_invisible = (1 - mask_for_point) * point_weight
      metric = self._l1_loss(
          sdf_true, sdf_pred, weight=point_weight_visible, mode=mode)
      loss_summaries['metric/' + key + '/l1_loss/visible'] = metric
      metric = self._l1_loss(
          sdf_true, sdf_pred, weight=point_weight_invisible, mode=mode)
      loss_summaries['metric/' + key + '/l1_loss/invisible'] = metric

      # Save point weight to image summaries.
      summary_key = 'point_weight/level' + str(level)
      image_summaries_update = misc_utils.get_image_summary(
          summary_key, point_weight, channels_use='first',
          spatial_dims=spatial_dims, normalize=False,
          summary_config=self.summary_config,
          data_mode=self.eval_data_mode)
      image_summaries.update(image_summaries_update)

  def add_consistency_loss(self,
                           key: str,
                           item: Tuple[tf.Tensor, tf.Tensor, int, str, str],
                           loss_summaries: Dict[str, Any],
                           model_outputs_and_targets: Dict[str, Any],
                           mode: str = 'loss',
                           flags: Dict[str, Any] = None):
    """Add consistency loss."""
    total_loss = 0
    points_sdf, points_dist, level, label_type, _ = item

    if level == 0:
      pass
    if label_type == 'gt_residual':
      points_sdf_ref = tf.zeros_like(points_sdf)
    elif label_type == 'gt_full':
      key_ref = key[:-1] + str(level - 1)
      points_sdf_ref = model_outputs_and_targets[key_ref][0]
    else:
      raise NotImplementedError

    if (flags is not None and flags['consistency_loss'] and
        'sdf_consistency_l1' in self.loss_params and
        self.loss_params['sdf_consistency_l1']['term_weight'][level] != 0):
      if self.loss_params['sdf_consistency_l1']['stop_grad_ref']:
        points_sdf_ref = tf.stop_gradient(points_sdf_ref)

      points_weight = tf.reduce_mean(points_dist, axis=-1)[..., None]
      # Tensor with shape [batch_size, num_point, 1].
      points_weight = self.transform_point_weight(
          points_weight, self.loss_params['sdf_consistency_l1']
          ['point_weight_config/dist_to_visible'])

      loss = (
          self._l1_loss(
              points_sdf_ref * points_weight,
              points_sdf * points_weight,
              mode=mode) *
          self.loss_params['sdf_consistency_l1']['term_weight'][level])
      total_loss = total_loss + loss
      loss_summaries['loss/' + key + '/l1_loss'] = loss

    return total_loss

  def add_symmetry_loss(self,
                        key: str,
                        item: Tuple[tf.Tensor, tf.Tensor, int, str, str, str],
                        loss_summaries: Dict[str, Any],
                        model_outputs_and_targets: Dict[str, Any],
                        mode: str = 'loss',
                        flags: Dict[str, Any] = None):
    """Add symmetry loss."""
    total_loss = 0
    points_sdf, points_dist, level, label_type, label_type_level0, _ = item

    if level == 0:
      pass
    if label_type == 'gt_residual':
      key_ref = key.replace('points_symmetry/', 'points_residual_sdf/')
    elif label_type == 'gt_full':
      key_ref = key.replace('points_symmetry/', 'points_sdf/')
    else:
      raise NotImplementedError

    # Different from consistency loss, use gt sdf as reference sdf.
    points_sdf_ref = model_outputs_and_targets[key_ref][1]

    if (flags is not None and flags['symmetry_loss'] and
        'sdf_symmetry_l1' in self.loss_params and
        self.loss_params['sdf_symmetry_l1']['term_weight'][level] != 0):
      if self.loss_params['sdf_symmetry_l1']['stop_grad_ref']:
        points_sdf_ref = tf.stop_gradient(points_sdf_ref)
      points_sdf_ref = tf.tile(
          points_sdf_ref,
          (1, tf.shape(points_sdf)[1] / tf.shape(points_sdf_ref)[1], 1))

      # Compute point weight based on distance to visible points.
      points_weight_dist = tf.reduce_mean(points_dist, axis=-1)[..., None]
      # Tensor with shape [batch_size, num_point, 1].
      points_weight_dist = self.transform_point_weight(
          points_weight_dist, self.loss_params['sdf_symmetry_l1']
          ['point_weight_config/dist_to_visible'])

      # Compute point weight based on global prior from level 0 (how close
      #  level 0 is symmetric at given points).
      if label_type_level0 == 'gt_residual':
        key_ref_level0 = key.replace('points_symmetry/', 'points_residual_sdf/')
      elif label_type_level0 == 'gt_full':
        key_ref_level0 = key.replace('points_symmetry/', 'points_sdf/')
      else:
        raise NotImplementedError
      key_ref_level0 = key_ref_level0[:-1] + str(0)
      key_level0 = key[:-1] + str(0)
      points_sdf_level0 = model_outputs_and_targets[key_level0][0]
      points_sdf_ref_level0 = model_outputs_and_targets[key_ref_level0][0]
      points_sdf_ref_level0 = tf.tile(
          points_sdf_ref_level0,
          (1,
           tf.shape(points_sdf_level0)[1] / tf.shape(points_sdf_ref_level0)[1],
           1))
      points_weight_prior = tf.stop_gradient(
          points_sdf_level0 - points_sdf_ref_level0)
      points_weight_prior = self.transform_point_weight(
          points_weight_prior, self.loss_params['sdf_symmetry_l1']
          ['point_weight_config/global_prior'])

      # Compute total point weight.
      points_weight = points_weight_dist * points_weight_prior

      # Compute loss.
      loss = (self._l1_loss(
          points_sdf_ref * points_weight, points_sdf * points_weight,
          mode=mode) *
              self.loss_params['sdf_symmetry_l1']['term_weight'][level])
      total_loss = total_loss + loss
      loss_summaries['loss/' + key + '/l1_loss'] = loss

    return total_loss

  def __call__(self,
               model_outputs_and_targets: Dict[str, Any],
               mode: str = 'loss',
               flags: Dict[str, Any] = None) -> Dict[str, Any]:
    """Calculate the loss.

    Args:
      model_outputs_and_targets: A dictionary holding prediction and ground
        truth.
      mode: The mode of the loss function, one of ['loss', 'metric'].
      flags: flags for whether including certain losses.

    Returns:
      A dictionary holding the loss, scalar summaries and image summaries.
    """
    training = model_outputs_and_targets['training']
    spatial_dims = model_outputs_and_targets['image_size']

    loss_summaries = {}
    image_summaries = {}
    total_loss = None
    if training:
      total_loss = 0

    for key, item in model_outputs_and_targets.items():
      if not isinstance(item, tuple):
        if training and key == 'root_feature':
          total_loss += self.add_root_feature_reg_loss(
              key, item, loss_summaries, mode)
      elif isinstance(item, tuple) and len(item) == 2:
        if training and key.split('/')[0] == 'code_grid':
          total_loss += self.add_code_grid_reg_loss(
              key, item, loss_summaries, mode)
      elif isinstance(item, tuple) and len(item) == 3:
        # Add metric for contour.
        if item[-1] == 'metric/contour':
          chamfer_eval_data, _, _ = item
          chamfer_l2 = tf.convert_to_tensor(
              [chamfer_eval_data[i][0] for i in range(len(chamfer_eval_data))])
          chamfer_l2_unsquare = tf.convert_to_tensor(
              [chamfer_eval_data[i][1] for i in range(len(chamfer_eval_data))])
          chamfer_l2 = tf.reduce_mean(chamfer_l2)
          chamfer_l2_unsquare = tf.reduce_mean(chamfer_l2_unsquare)
          loss_summaries['metric-contour/' + key + '/chamfer_l2'] = chamfer_l2
          loss_summaries['metric-contour/' + key + '/chamfer_l2_unsquare'] = (
              chamfer_l2_unsquare)

        # Add metric for iou.
        elif item[-1] == 'metric/iou':
          iou, _, _ = item
          iou = tf.reduce_mean(iou)
          loss_summaries['metric-occupancy/' + key] = iou
      elif isinstance(item, tuple) and len(item) == 4:
        if (training and item[-1] == 'loss/sdf') or item[-1] == 'metric/sdf':
          # Compute per-point weight.
          _, _, level, _ = item
          key_full_sdf = ''.join(key.split('residual_', 1))
          sdf_full_pred, sdf_full_true, _, _ = (
              model_outputs_and_targets[key_full_sdf])
          if 'point_weight_config' in self.loss_params:
            point_weight_config = self.loss_params['point_weight_config'][level]
          else:
            point_weight_config = None
          point_weight = self.compute_point_weight(
              sdf_full_pred, sdf_full_true, point_weight_config)

          if training and item[-1] == 'loss/sdf':
            total_loss += self.add_sdf_loss(
                key, item, point_weight, loss_summaries, mode)
          else:
            self.add_sdf_metric(
                key, item, point_weight, spatial_dims, loss_summaries,
                image_summaries, model_outputs_and_targets, mode)
      elif isinstance(item, tuple) and len(item) == 5:
        if training and item[-1] == 'loss/consistency':
          total_loss += self.add_consistency_loss(
              key, item, loss_summaries, model_outputs_and_targets, mode, flags)
      elif isinstance(item, tuple) and len(item) == 6:
        if training and item[-1] == 'loss/symmetry':
          total_loss += self.add_symmetry_loss(
              key, item, loss_summaries, model_outputs_and_targets, mode, flags)

    return {
        'total_loss': total_loss,
        'loss_summaries': loss_summaries,
        'image_summaries': image_summaries,
    }
