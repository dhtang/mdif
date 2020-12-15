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

"""Tests for google3.vr.perception.volume_compression.mdif.model.point_sampler_lib."""

from absl.testing import parameterized
import tensorflow as tf

from google3.vr.perception.volume_compression.mdif.model import point_sampler_lib


class PointSamplerLibTest(parameterized.TestCase, tf.test.TestCase):

  @parameterized.parameters(({
      'mask': {
          'apply': False
      }
  }), ({
      'mask': {
          'apply': True,
          'mode': 'none'
      }
  }), ({
      'mask': {
          'apply': True,
          'mode': 'right_half',
          'offset': (0, 0)
      }
  }))
  def test_point_sampler(self, mask):
    params = {
        'normalize_coordinates': True,
        'all_pixels': True,
        'untruncated': True,
        'untruncated/num_point': 256,
        'untruncated/mode': 'uniform',
        'untruncated/truncate': 5,
        'regular': True,
        'regular/num_point': 256,
        'global': True,
        'global/num_point': 256,
        'global/mode': 'uniform'
    }
    params['mask'] = mask
    sampler = point_sampler_lib.PointSampler(params)

    spatial_dims = (32, 32)
    sdf_map = tf.ones((*spatial_dims, 1), dtype=tf.float32)
    output = sampler(spatial_dims, sdf_map)

    with self.subTest(name='mask_for_point'):
      self.assertSequenceEqual(output['mask_for_point'].shape, (1024,))
    with self.subTest(name='all_pixels'):
      key = 'all_pixels'
      self.assertSequenceEqual(output['points/' + key].shape, (1024, 2))
      self.assertSequenceEqual(output['points_sdf_gt/' + key].shape, (1024, 1))
    with self.subTest(name='untruncated'):
      key = 'untruncated/' + params['untruncated/mode']
      self.assertSequenceEqual(output['points/' + key].shape, (256, 2))
      self.assertSequenceEqual(output['points_sdf_gt/' + key].shape, (256, 1))
    with self.subTest(name='regular'):
      key = 'regular'
      self.assertSequenceEqual(output['points/' + key].shape, (256, 2))
      self.assertSequenceEqual(output['points_sdf_gt/' + key].shape, (256, 1))
    with self.subTest(name='global'):
      key = 'global/' + params['global/mode']
      self.assertSequenceEqual(output['points/' + key].shape, (256, 2))
      self.assertSequenceEqual(output['points_sdf_gt/' + key].shape, (256, 1))

  @parameterized.parameters(({
      'depth_views': None,
      'mask': {
          'apply': True,
          'mode': 'right_half',
          'offset': (0, 0)
      }
  }), ({
      'depth_views': (0,),
      'mask': {
          'apply': False
      }
  }))
  def test_point_sampler_3d(self, depth_views, mask):
    params = {
        'normalize_coordinates': True,
        'all_pixels': True,
        'regular': True,
        'regular/num_point': 256,
        'global': True,
        'global/num_point': 256,
        'near_surface': True,
        'near_surface/num_point': 256,
        'near_surface/mode': 'uniform',
        'symmetry': True,
        'symmetry_loss': True,
        'symmetry/mode': ['reflect_z'],
        'symmetry/visible/point_source': [
            'global', 'near_surface', 'depth_xyz'
        ],
        'symmetry/point_dist/min_k': 10,
        'consistency': True,
        'consistency_loss': True,
        'consistency/invisible/point_source': 'global',
        'consistency/invisible/num_point': 256,
        'consistency/visible/point_source': [
            'global', 'near_surface', 'depth_xyz'
        ],
        'consistency/point_dist/min_k': 10,
    }
    params['depth_views'] = depth_views
    params['mask'] = mask
    sampler = point_sampler_lib.PointSampler3D(params)

    spatial_dims = (16, 16, 16)
    num_view = 1
    point_samples = {}
    point_samples['grid_samples'] = tf.zeros((*spatial_dims, 4),
                                             dtype=tf.float32)
    point_samples['uniform_samples'] = tf.zeros((1000, 4), dtype=tf.float32)
    point_samples['near_surface_samples'] = tf.zeros((1000, 4),
                                                     dtype=tf.float32)
    point_samples['uniform_samples_per_camera'] = tf.zeros((num_view, 1000, 4),
                                                           dtype=tf.float32)
    point_samples['near_surface_samples_per_camera'] = tf.zeros(
        (num_view, 1000, 4), dtype=tf.float32)
    point_samples['depth_xyzn_per_camera'] = tf.zeros((num_view, 1000, 6),
                                                      dtype=tf.float32)

    output = sampler(spatial_dims, point_samples)

    with self.subTest(name='mask_for_point'):
      self.assertSequenceEqual(output['mask_for_point'].shape, (4096,))
    with self.subTest(name='all_pixels'):
      key = 'all_pixels'
      self.assertSequenceEqual(output['points/' + key].shape, (4096, 3))
      self.assertSequenceEqual(output['points_sdf_gt/' + key].shape, (4096, 1))
    with self.subTest(name='regular'):
      key = 'regular'
      self.assertSequenceEqual(output['points/' + key].shape, (256, 3))
      self.assertSequenceEqual(output['points_sdf_gt/' + key].shape, (256, 1))
    with self.subTest(name='global'):
      key = 'global/uniform'
      self.assertSequenceEqual(output['points/' + key].shape, (256, 3))
      self.assertSequenceEqual(output['points_sdf_gt/' + key].shape, (256, 1))
    with self.subTest(name='near_surface'):
      key = 'near_surface/uniform'
      self.assertSequenceEqual(output['points/' + key].shape, (256, 3))
      self.assertSequenceEqual(output['points_sdf_gt/' + key].shape, (256, 1))
    with self.subTest(name='symmetry'):
      key = 'global/uniform'
      self.assertSequenceEqual(output['points_symmetry/' + key].shape, (256, 3))
      self.assertSequenceEqual(output['points_symmetry_dist/' + key].shape,
                               (256, 10))
      key = 'near_surface/uniform'
      self.assertSequenceEqual(output['points_symmetry/' + key].shape, (256, 3))
      self.assertSequenceEqual(output['points_symmetry_dist/' + key].shape,
                               (256, 10))
    with self.subTest(name='consistency'):
      self.assertSequenceEqual(output['points_consistency'].shape, (256, 3))
      self.assertSequenceEqual(output['points_consistency_dist'].shape,
                               (256, 10))


if __name__ == '__main__':
  tf.test.main()
