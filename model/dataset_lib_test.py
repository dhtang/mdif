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

"""Tests for google3.vr.perception.volume_compression.mdif.model.dataset_lib."""

import os
import tempfile
from typing import Any, Dict

import numpy as np
import tensorflow.google as tf

from google3.sstable.python import pywrapsstable
from google3.sstable.python import sstable
from google3.vr.perception.volume_compression.mdif.model import dataset_lib


def _generate_example_2d(params: Dict[str, Any]) -> str:
  shape = params['shape']

  sdf_map = tf.zeros([np.prod(shape)], dtype=tf.float32)

  example = tf.Example()
  example.features.feature['data'].float_list.value.extend(sdf_map)
  example.features.feature['shape'].int64_list.value.extend(shape)

  return example.SerializeToString()


def _generate_sstable_2d(save_fp: str,
                         params: Dict[str, Any],
                         num_data: int = 2):

  with sstable.Builder(save_fp,
                       pywrapsstable.SSTableBuilderOptions()) as builder:
    for i in range(num_data):
      example = _generate_example_2d(params)
      builder.Add('data_{:05}'.format(i), example)


def _generate_example_3d(data_id: int, params: Dict[str, Any]) -> str:
  grid_size = params['grid_size']
  num_samples = params['num_samples']
  num_samples_per_camera = params['num_samples_per_camera']
  num_cameras = params['num_cameras']

  grid_shape = [grid_size, grid_size, grid_size]
  uniform_samples_shape = [num_samples, 4]
  near_surface_samples_shape = [num_samples, 4]
  depth_xyzn_per_camera_shape = [num_cameras, num_samples_per_camera, 6]
  uniform_samples_per_camera_shape = [num_cameras, num_samples_per_camera, 4]
  near_surface_samples_per_camera_shape = [
      num_cameras, num_samples_per_camera, 4
  ]

  data_id = tf.constant([data_id], dtype=tf.int64)
  world2grid = tf.zeros([4 * 4], dtype=tf.float32)
  grid = tf.zeros([np.prod(grid_shape)], dtype=tf.float32)
  uniform_samples = tf.zeros([np.prod(uniform_samples_shape)], dtype=tf.float32)
  near_surface_samples = tf.zeros([np.prod(near_surface_samples_shape)],
                                  dtype=tf.float32)
  depth_xyzn_per_camera = tf.zeros([np.prod(depth_xyzn_per_camera_shape)],
                                   dtype=tf.float32)
  uniform_samples_per_camera = tf.zeros(
      [np.prod(uniform_samples_per_camera_shape)], dtype=tf.float32)
  near_surface_samples_per_camera = tf.zeros(
      [np.prod(near_surface_samples_per_camera_shape)], dtype=tf.float32)

  example = tf.Example()
  example.features.feature['grid_shape'].int64_list.value.extend(grid_shape)
  example.features.feature['uniform_samples_shape'].int64_list.value.extend(
      uniform_samples_shape)
  example.features.feature[
      'near_surface_samples_shape'].int64_list.value.extend(
          near_surface_samples_shape)
  example.features.feature[
      'depth_xyzn_per_camera_shape'].int64_list.value.extend(
          depth_xyzn_per_camera_shape)
  example.features.feature[
      'uniform_samples_per_camera_shape'].int64_list.value.extend(
          uniform_samples_per_camera_shape)
  example.features.feature[
      'near_surface_samples_per_camera_shape'].int64_list.value.extend(
          near_surface_samples_per_camera_shape)

  example.features.feature['data_id'].int64_list.value.extend(data_id)
  example.features.feature['world2grid'].float_list.value.extend(world2grid)
  example.features.feature['grid'].float_list.value.extend(grid)
  example.features.feature['uniform_samples'].float_list.value.extend(
      uniform_samples)
  example.features.feature['near_surface_samples'].float_list.value.extend(
      near_surface_samples)
  example.features.feature['depth_xyzn_per_camera'].float_list.value.extend(
      depth_xyzn_per_camera)
  example.features.feature[
      'uniform_samples_per_camera'].float_list.value.extend(
          uniform_samples_per_camera)
  example.features.feature[
      'near_surface_samples_per_camera'].float_list.value.extend(
          near_surface_samples_per_camera)

  return example.SerializeToString()


def _generate_sstable_3d(save_fp: str,
                         params: Dict[str, Any],
                         num_data: int = 2):
  with sstable.Builder(save_fp,
                       pywrapsstable.SSTableBuilderOptions()) as builder:
    for i in range(num_data):
      example = _generate_example_3d(i, params)
      builder.Add('data_{:05}'.format(i), example)


class DatasetLibTest(tf.test.TestCase):

  def test_preprocess_2d(self):
    params = {'shape': [4, 4, 1]}
    key = 'data_{:05}'.format(0)
    example = _generate_example_2d(params)

    out = dataset_lib._preprocess_2d(key, example, params)

    self.assertAllEqual(out['sdf_map'],
                        tf.zeros(params['shape'], dtype=tf.float32))

  def test_load_dataset_2d(self):
    batch_size = 1
    params = {'shape': [4, 4, 1]}

    sstable_folder = tempfile.mkdtemp()
    data_sources = os.path.join(sstable_folder, 'temp.sst')
    _generate_sstable_2d(data_sources, params, num_data=batch_size)

    dataset = dataset_lib.load_dataset_2d(
        data_sources, batch_size=batch_size, params=params)

    for _, batch in enumerate(dataset):
      self.assertAllEqual(
          batch['sdf_map'],
          tf.zeros((batch_size, *params['shape']), dtype=tf.float32))

  def test_parse_fn_3d(self):
    batch_size = 1
    grid_size = 2
    num_samples = 1
    num_samples_per_camera = 1
    num_cameras = 1
    params = {
        'grid_size': grid_size,
        'num_samples': num_samples,
        'num_samples_per_camera': num_samples_per_camera,
        'num_cameras': num_cameras
    }
    key = 'data_{:05}'.format(0)
    example = _generate_example_3d(0, params)

    batch = dataset_lib._parse_fn_3d(key, example, params, add_batch_dim=True)

    self.assertSequenceEqual(batch['data_key'].shape, [batch_size])
    self.assertSequenceEqual(batch['data_id'].shape, [batch_size])
    self.assertAllEqual(batch['world2grid'],
                        tf.zeros((batch_size, 4, 4), dtype=tf.float32))
    self.assertAllEqual(
        batch['grid_samples'],
        tf.zeros((batch_size, grid_size, grid_size, grid_size, 1),
                 dtype=tf.float32))
    self.assertAllEqual(
        batch['uniform_samples'],
        tf.zeros((batch_size, num_samples, 4), dtype=tf.float32))
    self.assertAllEqual(
        batch['near_surface_samples'],
        tf.zeros((batch_size, num_samples, 4), dtype=tf.float32))
    self.assertAllEqual(
        batch['depth_xyzn_per_camera'],
        tf.zeros((batch_size, num_cameras, num_samples_per_camera, 6),
                 dtype=tf.float32))
    self.assertAllEqual(
        batch['uniform_samples_per_camera'],
        tf.zeros((batch_size, num_cameras, num_samples_per_camera, 4),
                 dtype=tf.float32))
    self.assertAllEqual(
        batch['near_surface_samples_per_camera'],
        tf.zeros((batch_size, num_cameras, num_samples_per_camera, 4),
                 dtype=tf.float32))

  def test_load_dataset_3d(self):
    batch_size = 1
    grid_size = 2
    num_samples = 1
    num_samples_per_camera = 1
    num_cameras = 1
    params = {
        'grid_size': grid_size,
        'num_samples': num_samples,
        'num_samples_per_camera': num_samples_per_camera,
        'num_cameras': num_cameras
    }

    sstable_folder = tempfile.mkdtemp()
    data_sources = os.path.join(sstable_folder, 'temp.sst')
    _generate_sstable_3d(data_sources, params, num_data=batch_size)

    dataset = dataset_lib.load_dataset_3d(
        data_sources, batch_size=batch_size, is_training=True, params=params)

    for _, batch in enumerate(dataset):
      self.assertSequenceEqual(batch['data_key'].shape, [batch_size])
      self.assertSequenceEqual(batch['data_id'].shape, [batch_size])
      self.assertAllEqual(batch['world2grid'],
                          tf.zeros((batch_size, 4, 4), dtype=tf.float32))
      self.assertAllEqual(
          batch['grid_samples'],
          tf.zeros((batch_size, grid_size, grid_size, grid_size, 1),
                   dtype=tf.float32))
      self.assertAllEqual(
          batch['uniform_samples'],
          tf.zeros((batch_size, num_samples, 4), dtype=tf.float32))
      self.assertAllEqual(
          batch['near_surface_samples'],
          tf.zeros((batch_size, num_samples, 4), dtype=tf.float32))
      self.assertAllEqual(
          batch['depth_xyzn_per_camera'],
          tf.zeros((batch_size, num_cameras, num_samples_per_camera, 6),
                   dtype=tf.float32))
      self.assertAllEqual(
          batch['uniform_samples_per_camera'],
          tf.zeros((batch_size, num_cameras, num_samples_per_camera, 4),
                   dtype=tf.float32))
      self.assertAllEqual(
          batch['near_surface_samples_per_camera'],
          tf.zeros((batch_size, num_cameras, num_samples_per_camera, 4),
                   dtype=tf.float32))


if __name__ == '__main__':
  tf.test.main()
