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

"""Dataset library for loading training and testing data."""

from typing import Any, Dict, Iterable, Union

import gin.tf
import numpy as np
import tensorflow.google as tf

from google3.pyglib import gfile


def _preprocess_2d(key: str,
                   example: str,
                   params: Dict[str, Any] = None) -> Dict[str, tf.Tensor]:
  """Preprocesses each example and reshape the sdf data into a 2D image.

  Args:
    key: Key of this entry.
    example: A serialized tf.Example.
    params: Parameters for preprocessing.

  Returns:
    A dictionary containing
      'sdf_map': [dim_h, dim_w, 1] tensor, sdf map.
  """
  del key  # Unused variable.

  if params is None:
    shape = [256, 256, 1]
  else:
    shape = params['shape']

  feature_map = {
      'data': tf.io.FixedLenFeature([np.prod(shape)], dtype=tf.float32),
      'shape': tf.io.FixedLenFeature([3], dtype=tf.int64),
  }

  example_data = tf.io.parse_single_example(example, feature_map)
  output = dict()
  output['sdf_map'] = tf.reshape(example_data['data'], example_data['shape'])

  return output


@gin.configurable
def load_dataset_2d(
    data_sources: str,
    batch_size: int,
    params: Union[Dict[str, Any], None] = None) -> tf.data.Dataset:
  """A 2D SDF dataset wrapper providing iterator-like interface.

  Args:
    data_sources: Path of the sstable.
    batch_size: Size of each batch.
    params: Parameters for preprocessing.

  Raises:
    RuntimeError: The sstable does not exist.

  Returns:
    An iterator of the sstable.
  """
  if not gfile.Glob(data_sources):
    raise RuntimeError('"{}" not found.'.format(data_sources))

  with tf.name_scope('input'):
    dataset = tf.data.SSTableDataset([data_sources])
    dataset = dataset.map(
        lambda key, example: _preprocess_2d(key, example, params))
    dataset = dataset.batch(batch_size)
    dataset = dataset.prefetch(batch_size)

    return dataset


def _parse_fn_3d(key: str,
                 example: str,
                 params: Dict[str, Any] = None,
                 add_batch_dim: bool = False) -> Dict[str, Any]:
  """Preprocesses each example.

  Args:
    key: Key of this entry.
    example: A serialized tf.Example.
    params: Parameters for preprocessing.
    add_batch_dim: Whether add the batch dimension.

  Returns:
    A dictionary containing preprocessed data.
  """
  if params is None:
    grid_size = 128
    num_samples = 100000
    num_samples_per_camera = 10000
    num_cameras = 20
  else:
    grid_size = params['grid_size']
    num_samples = params['num_samples']
    num_samples_per_camera = params['num_samples_per_camera']
    num_cameras = params['num_cameras']

  if not add_batch_dim:
    feature_map = {
        'grid':
            tf.io.FixedLenFeature([grid_size**3], dtype=tf.float32),
        'grid_shape':
            tf.io.FixedLenFeature([3], dtype=tf.int64),
        'uniform_samples':
            tf.io.FixedLenFeature([num_samples * 4], dtype=tf.float32),
        'uniform_samples_shape':
            tf.io.FixedLenFeature([2], dtype=tf.int64),
        'near_surface_samples':
            tf.io.FixedLenFeature([num_samples * 4], dtype=tf.float32),
        'near_surface_samples_shape':
            tf.io.FixedLenFeature([2], dtype=tf.int64),
        'data_id':
            tf.io.FixedLenFeature([1], dtype=tf.int64),
        'world2grid':
            tf.io.FixedLenFeature([4 * 4], dtype=tf.float32),
        'depth_xyzn_per_camera_shape':
            tf.io.FixedLenFeature([3], dtype=tf.int64),
        'depth_xyzn_per_camera':  # Depth pixels back-projected to world space.
            tf.io.FixedLenFeature([num_cameras * num_samples_per_camera * 6],
                                  dtype=tf.float32),
        'uniform_samples_per_camera_shape':
            tf.io.FixedLenFeature([3], dtype=tf.int64),
        'uniform_samples_per_camera':
            tf.io.FixedLenFeature([num_cameras * num_samples_per_camera * 4],
                                  dtype=tf.float32),
        'near_surface_samples_per_camera_shape':
            tf.io.FixedLenFeature([3], dtype=tf.int64),
        'near_surface_samples_per_camera':
            tf.io.FixedLenFeature([num_cameras * num_samples_per_camera * 4],
                                  dtype=tf.float32),
    }
  else:
    key = tf.constant([key], tf.string)
    feature_map = {
        'grid':
            tf.io.FixedLenFeature([1, grid_size**3], dtype=tf.float32),
        'grid_shape':
            tf.io.FixedLenFeature([1, 3], dtype=tf.int64),
        'uniform_samples':
            tf.io.FixedLenFeature([1, num_samples * 4], dtype=tf.float32),
        'uniform_samples_shape':
            tf.io.FixedLenFeature([1, 2], dtype=tf.int64),
        'near_surface_samples':
            tf.io.FixedLenFeature([1, num_samples * 4], dtype=tf.float32),
        'near_surface_samples_shape':
            tf.io.FixedLenFeature([1, 2], dtype=tf.int64),
        'data_id':
            tf.io.FixedLenFeature([1, 1], dtype=tf.int64),
        'world2grid':
            tf.io.FixedLenFeature([1, 4 * 4], dtype=tf.float32),
        'depth_xyzn_per_camera_shape':
            tf.io.FixedLenFeature([1, 3], dtype=tf.int64),
        'depth_xyzn_per_camera':  # Depth pixels back-projected to world space.
            tf.io.FixedLenFeature([1, num_cameras * num_samples_per_camera * 6],
                                  dtype=tf.float32),
        'uniform_samples_per_camera_shape':
            tf.io.FixedLenFeature([1, 3], dtype=tf.int64),
        'uniform_samples_per_camera':
            tf.io.FixedLenFeature([1, num_cameras * num_samples_per_camera * 4],
                                  dtype=tf.float32),
        'near_surface_samples_per_camera_shape':
            tf.io.FixedLenFeature([1, 3], dtype=tf.int64),
        'near_surface_samples_per_camera':
            tf.io.FixedLenFeature([1, num_cameras * num_samples_per_camera * 4],
                                  dtype=tf.float32),
    }

  example_data = tf.io.parse_example(example, feature_map)
  grid_shape = example_data['grid_shape']
  grid = example_data['grid']
  uniform_samples_shape = example_data['uniform_samples_shape']
  uniform_samples = example_data['uniform_samples']
  near_surface_samples_shape = example_data['near_surface_samples_shape']
  near_surface_samples = example_data['near_surface_samples']
  data_id = example_data['data_id'][:, 0]  # Tensor with shape [batch_size].
  world2grid = example_data['world2grid']
  depth_xyzn_per_camera_shape = example_data['depth_xyzn_per_camera_shape']
  depth_xyzn_per_camera = example_data['depth_xyzn_per_camera']
  uniform_samples_per_camera_shape = example_data[
      'uniform_samples_per_camera_shape']
  uniform_samples_per_camera = example_data['uniform_samples_per_camera']
  near_surface_samples_per_camera_shape = example_data[
      'near_surface_samples_per_camera_shape']
  near_surface_samples_per_camera = example_data[
      'near_surface_samples_per_camera']

  batch_size = grid_shape.shape[0]
  grid_shape = tf.concat(
      [tf.constant([batch_size], dtype=tf.int64), grid_shape[0, :]], axis=0)
  uniform_samples_shape = tf.concat(
      [tf.constant([batch_size], dtype=tf.int64), uniform_samples_shape[0, :]],
      axis=0)
  near_surface_samples_shape = tf.concat([
      tf.constant([batch_size], dtype=tf.int64),
      near_surface_samples_shape[0, :]
  ],
                                         axis=0)

  grid = tf.reshape(grid, grid_shape)[..., None]
  # Tensor with shape [batch_size, dim_d, dim_h, dim_w, 1].

  uniform_samples = tf.reshape(uniform_samples, uniform_samples_shape)
  # Tensor with shape [batch_size, num_point, 4].

  near_surface_samples = tf.reshape(near_surface_samples,
                                    near_surface_samples_shape)
  # Tensor with shape [batch_size, num_point, 4].

  world2grid = tf.reshape(world2grid, [batch_size, 4, 4])

  depth_xyzn_per_camera_shape = tf.concat([
      tf.constant([batch_size], dtype=tf.int64),
      depth_xyzn_per_camera_shape[0, :]
  ],
                                          axis=0)
  uniform_samples_per_camera_shape = tf.concat([
      tf.constant([batch_size], dtype=tf.int64),
      uniform_samples_per_camera_shape[0, :]
  ],
                                               axis=0)
  near_surface_samples_per_camera_shape = tf.concat([
      tf.constant([batch_size], dtype=tf.int64),
      near_surface_samples_per_camera_shape[0, :]
  ],
                                                    axis=0)

  depth_xyzn_per_camera = tf.reshape(depth_xyzn_per_camera,
                                     depth_xyzn_per_camera_shape)
  # Tensor with shape [batch_size, num_view, num_point, 6].

  uniform_samples_per_camera = tf.reshape(uniform_samples_per_camera,
                                          uniform_samples_per_camera_shape)
  # Tensor with shape [batch_size, num_view, num_point, 4].

  near_surface_samples_per_camera = tf.reshape(
      near_surface_samples_per_camera, near_surface_samples_per_camera_shape)
  # Tensor with shape [batch_size, num_view, num_point, 4].

  output = {
      'data_key': key,
      'data_id': data_id,
      'world2grid': world2grid,
      'grid_samples': grid,
      'uniform_samples': uniform_samples,
      'near_surface_samples': near_surface_samples,
      'depth_xyzn_per_camera': depth_xyzn_per_camera,
      'uniform_samples_per_camera': uniform_samples_per_camera,
      'near_surface_samples_per_camera': near_surface_samples_per_camera
  }

  return output


def load_dataset_3d(data_sources: Union[str, Iterable[str]],
                    batch_size: int,
                    is_training: bool,
                    params: Dict[str, Any] = None) -> tf.data.Dataset:
  """Loads the dataset.

  Args:
    data_sources: List of files that make up the dataset.
    batch_size: Batch size.
    is_training: If True, random shuffling and non-deterministic reading are
      used.
    params: Parameters for preprocessing.

  Raises:
    RuntimeError: The sstable does not exist.
  Returns:
    dataset: tf.data.Dataset containing input data for training and testing.
  """
  if isinstance(data_sources, str):
    data_sources = [data_sources]

  for data_source in data_sources:
    if not gfile.Glob(data_source):
      raise RuntimeError('"{}" not found.'.format(data_source))

  filenames_dataset = tf.data.Dataset.list_files(
      data_sources, shuffle=is_training)

  dataset = filenames_dataset.interleave(
      tf.data.SSTableDataset,
      cycle_length=8,
      block_length=2,
      num_parallel_calls=tf.data.experimental.AUTOTUNE)

  dataset = dataset.batch(batch_size, drop_remainder=True)
  dataset = dataset.map(
      lambda key, example: _parse_fn_3d(key, example, params),
      num_parallel_calls=tf.data.experimental.AUTOTUNE)
  dataset = dataset.prefetch(tf.data.experimental.AUTOTUNE)

  if is_training:
    # Set deterministic to false for training.
    options = tf.data.Options()
    options.experimental_deterministic = False
    dataset = dataset.with_options(options)

  return dataset
