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

"""Utility functions."""

import struct
from typing import Any, Dict, Tuple, Sequence, Union

import numpy as np
import tensorflow as tf

from google3.pyglib import gfile


def get_image_summary(summary_key: str,
                      raw_data: tf.Tensor,
                      channels_use: str,
                      spatial_dims: Union[Sequence[int], None],
                      normalize: bool,
                      summary_config: Dict[str, Any],
                      extra_axis: int = 0,
                      data_mode: str = 'all') -> Dict[str, tf.Tensor]:
  """Processes raw data to be tensorboard image summaries.

  Args:
    summary_key: base name of output summaries.
    raw_data: raw data tensor to be processed.
    channels_use: channels that are used for image summary, one of ['first',
      'last']. 'first' means to use the first 1 or 3 channels, 'last'
      means to use the last 1 or 3 channels.
    spatial_dims: spatial dimensions of raw data. When it is None, raw data will
      not be reshaped.
    normalize: whether normalize the raw data to be within [0, 1].
    summary_config: configurations on output image summaries.
    extra_axis: number of extra axes.
    data_mode: the mode of input raw data, one of ['all', 'slices'].

  Returns:
    image_summary: dict containing image summaries.
  """
  batch_size = tf.shape(raw_data)[0]

  # Choose channels to be used for image summary.
  if tf.shape(raw_data)[-1] >= 3:
    channels_count = 3
  else:
    channels_count = 1

  if channels_use == 'first':
    summary_data = raw_data[..., :channels_count]
  elif channels_use == 'last':
    summary_data = raw_data[..., -channels_count:]
  else:
    raise ValueError('Unknown channels_use: %s' % channels_use)

  # Normalize data to [0, 1].
  if normalize:
    summary_data = (summary_data - tf.reduce_min(summary_data)) / (
        tf.reduce_max(summary_data) - tf.reduce_min(summary_data) + 1e-08)

  image_summary = {}

  if data_mode == 'all':
    if spatial_dims is not None:
      if extra_axis == 0:
        summary_data = tf.reshape(
            summary_data, (-1, *spatial_dims, channels_count))
      else:
        summary_data = tf.reshape(
            summary_data, (batch_size, -1, *spatial_dims, channels_count))

    if len(tf.shape(summary_data)) - extra_axis == 4:  # 2D data.
      image_summary[summary_key] = get_image_summary_helper(
          summary_data, extra_axis)
    elif len(tf.shape(summary_data)) - extra_axis == 5:  # 3D data.
      summary_volume = summary_data
      # Tensor with shape [batch_size, [extra_axis], dim_d, dim_h, dim_w,
      #  dim_c].

      dim_d = tf.shape(summary_volume)[-4]
      dim_h = tf.shape(summary_volume)[-3]
      dim_w = tf.shape(summary_volume)[-2]

      if summary_config['slice_idx_z']:
        for i in summary_config['slice_idx_z']:
          summary_data = summary_volume[
              ...,
              tf.cast(tf.math.round(i *
                                    tf.cast(dim_d, tf.float32)), tf.int32) -
              1, :, :, :]
          image_summary[summary_key + '/slice_z/' +
                        str(i)] = get_image_summary_helper(
                            summary_data, extra_axis)

      if summary_config['slice_idx_y']:
        for i in summary_config['slice_idx_y']:
          summary_data = summary_volume[
              ...,
              tf.cast(tf.math.round(i *
                                    tf.cast(dim_h, tf.float32)), tf.int32) -
              1, :, :]
          image_summary[summary_key + '/slice_y/' +
                        str(i)] = get_image_summary_helper(
                            summary_data, extra_axis)

      if summary_config['slice_idx_x']:
        for i in summary_config['slice_idx_x']:
          summary_data = summary_volume[
              ...,
              tf.cast(tf.math.round(i *
                                    tf.cast(dim_w, tf.float32)), tf.int32) -
              1, :]
          image_summary[summary_key + '/slice_x/' +
                        str(i)] = get_image_summary_helper(
                            summary_data, extra_axis)
    else:
      raise ValueError('Unknown summary_data dimension')
  elif data_mode == 'slices':
    summary_slices = summary_data
    dim_c = tf.shape(summary_slices)[-1]
    dim_d, dim_h, dim_w = spatial_dims
    start_idx = 0
    if summary_config['slice_idx_z']:
      for i in summary_config['slice_idx_z']:
        end_idx = start_idx + dim_h * dim_w
        summary_data = tf.reshape(summary_slices[..., start_idx:end_idx, :],
                                  (batch_size, -1, dim_h, dim_w, dim_c))
        start_idx = end_idx
        image_summary[summary_key + '/slice_z/' +
                      str(i)] = get_image_summary_helper(
                          summary_data, extra_axis=1)
    if summary_config['slice_idx_y']:
      for i in summary_config['slice_idx_y']:
        end_idx = start_idx + dim_d * dim_w
        summary_data = tf.reshape(summary_slices[..., start_idx:end_idx, :],
                                  (batch_size, -1, dim_d, dim_w, dim_c))
        start_idx = end_idx
        image_summary[summary_key + '/slice_y/' +
                      str(i)] = get_image_summary_helper(
                          summary_data, extra_axis=1)
    if summary_config['slice_idx_x']:
      for i in summary_config['slice_idx_x']:
        end_idx = start_idx + dim_d * dim_h
        summary_data = tf.reshape(summary_slices[..., start_idx:end_idx, :],
                                  (batch_size, -1, dim_d, dim_h, dim_c))
        start_idx = end_idx
        image_summary[summary_key + '/slice_x/' +
                      str(i)] = get_image_summary_helper(
                          summary_data, extra_axis=1)
  else:
    raise ValueError('Unknown data_mode: %s' % data_mode)

  return image_summary


def get_image_summary_helper(summary_data: tf.Tensor,
                             extra_axis: int) -> tf.Tensor:
  """Helper function that reshapes tensor if needed.

  Args:
    summary_data: data tensor to be processed.
    extra_axis: number of extra axes.

  Returns:
    summary_data: processed data tensor.
  """
  if extra_axis == 1:
    data_shape = tf.shape(summary_data)
    summary_data = tf.reshape(summary_data,
                              (data_shape[0], -1, data_shape[3], data_shape[4]))
  return summary_data


def read_grd(path: str) -> Tuple[np.ndarray, np.ndarray]:
  """Reads a GAPS .grd file into a (tx, grd) pair."""
  with gfile.Open(path, 'rb') as f:
    content = f.read()
  res = struct.unpack('iii', content[:4 * 3])
  vcount = res[0] * res[1] * res[2]
  content = content[4 * 3:]
  tx = struct.unpack('f' * 16, content[:4 * 16])
  tx = np.array(tx).reshape([4, 4]).astype(np.float32)
  content = content[4 * 16:]
  grd = struct.unpack('f' * vcount, content[:4 * vcount])
  grd = np.array(grd).reshape(res).astype(np.float32)
  return tx, grd


def write_grd(path: str,
              volume: np.ndarray,
              world2grid: np.ndarray = None) -> None:
  """Writes a GAPS .grd file containing a voxel grid and world2grid matrix."""
  volume = np.squeeze(volume)
  assert len(volume.shape) == 3
  header = [int(s) for s in volume.shape]
  if world2grid is not None:
    header += [x.astype(np.float32) for x in np.reshape(world2grid, [16])]
  else:
    header += [1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1, 0, 0, 0, 0, 1]
  header = struct.pack(3 * 'i' + 16 * 'f', *header)
  content = volume.astype('f').tostring()
  if path is not None:
    with gfile.Open(path, 'wb') as f:
      f.write(header)
      f.write(content)


def write_grd_batch(
    path: Sequence[str],
    volume: np.ndarray,
    world2grid: np.ndarray = None) -> None:
  """Writes batched voxel grids and world2grid matrices to GAPS .grd files."""
  batch_size = volume.shape[0]

  for i in range(batch_size):
    if world2grid is None:
      world2grid_i = None
    else:
      world2grid_i = world2grid[i, ...]
    write_grd(path[i], volume[i, ...], world2grid_i)
