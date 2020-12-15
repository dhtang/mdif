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

"""Main function for local inference."""

import glob
import os

from absl import app
from absl import flags
from absl import logging
import gin.tf
import tensorflow as tf

from google3.vr.perception.volume_compression.mdif import train_eval_lib

FLAGS = flags.FLAGS

flags.DEFINE_enum('mode', None, ['cpu', 'gpu'],
                  'Distributed strategy approach.')
flags.DEFINE_string('base_folder', None, 'Path to checkpoints/summaries.')
flags.DEFINE_string('job_name', '', 'Name of the job.')
flags.DEFINE_string('checkpoint_step', None, 'Checkpoint step.')
flags.DEFINE_string('checkpoint_step_next', None, 'Next checkpoint step.')
flags.DEFINE_multi_string('gin_bindings', None, 'Gin parameter bindings.')
flags.DEFINE_multi_string('gin_configs', None, 'Gin config files.')


def main(argv):
  del argv  # Unused.

  gin_configs = FLAGS.gin_configs

  gin.parse_config_files_and_bindings(
      config_files=gin_configs,
      bindings=FLAGS.gin_bindings,
      skip_unknown=True,
      finalize_config=False)

  params_update = gin.query_parameter('inference_pipeline.params_update')
  if params_update is not None:
    for key, item in params_update.items():
      if key == 'latent_optim_sampling_params/depth_views':
        bind_key = 'MultiresDeepImplicitFunction.latent_optim_sampling_params'
        param_orig = gin.query_parameter(bind_key)
        param_orig['depth_views'] = item
        gin.bind_parameter(bind_key, param_orig)
        logging.info(bind_key)
        logging.info(gin.query_parameter(bind_key))
      elif key == 'sdf_symmetry_l1/point_weight_config/dist_to_visible':
        bind_key = 'get_training_elements.loss_params_update'
        param_orig = gin.query_parameter(bind_key)
        param_orig['sdf_symmetry_l1'][
            'point_weight_config/dist_to_visible'] = item
        gin.bind_parameter(bind_key, param_orig)
        logging.info(bind_key)
        logging.info(gin.query_parameter(bind_key))
      elif key == 'sdf_symmetry_l1/point_weight_config/global_prior':
        bind_key = 'get_training_elements.loss_params_update'
        param_orig = gin.query_parameter(bind_key)
        param_orig['sdf_symmetry_l1']['point_weight_config/global_prior'] = item
        gin.bind_parameter(bind_key, param_orig)
        logging.info(bind_key)
        logging.info(gin.query_parameter(bind_key))
      elif key == 'sdf_symmetry_l1/term_weight':
        bind_key = 'get_training_elements.loss_params_update'
        param_orig = gin.query_parameter(bind_key)
        param_orig['sdf_symmetry_l1']['term_weight'] = item
        gin.bind_parameter(bind_key, param_orig)
        logging.info(bind_key)
        logging.info(gin.query_parameter(bind_key))
      elif key == 'sdf_consistency_l1/point_weight_config/dist_to_visible':
        bind_key = 'get_training_elements.loss_params_update'
        param_orig = gin.query_parameter(bind_key)
        param_orig['sdf_consistency_l1'][
            'point_weight_config/dist_to_visible'] = item
        gin.bind_parameter(bind_key, param_orig)
        logging.info(bind_key)
        logging.info(gin.query_parameter(bind_key))
      elif key == 'sdf_consistency_l1/term_weight':
        bind_key = 'get_training_elements.loss_params_update'
        param_orig = gin.query_parameter(bind_key)
        param_orig['sdf_consistency_l1']['term_weight'] = item
        gin.bind_parameter(bind_key, param_orig)
        logging.info(bind_key)
        logging.info(gin.query_parameter(bind_key))
      else:
        raise ValueError('Unknown key', key)

  base_folder = FLAGS.base_folder
  train_base_folder = os.path.join(base_folder, 'train')

  if FLAGS.checkpoint_step == 'latest':
    model_path = tf.train.latest_checkpoint(train_base_folder)
  else:
    model_path = os.path.join(train_base_folder,
                              'ckpt-' + FLAGS.checkpoint_step)

  if not os.path.isfile(model_path + '.index'):
    logging.info('Checkpoint not found at %s', model_path)
    logging.info('Searching for the closest next checkpoint')

    # Search for the closest next checkpoint.
    ckpt_list = glob.glob(os.path.join(train_base_folder, 'ckpt-*.index'))
    temp = [int(i.split('-')[-1].split('.')[0]) for i in ckpt_list]
    temp = [i for i in temp if i >= int(FLAGS.checkpoint_step)]
    if not temp:
      raise ValueError('No checkpoint after', FLAGS.checkpoint_step)
    ckpt_next = min(temp)

    if (FLAGS.checkpoint_step_next == 'none' or
        ckpt_next < int(FLAGS.checkpoint_step_next)):
      model_path = os.path.join(train_base_folder, 'ckpt-' + str(ckpt_next))
      logging.info('Use the closest next checkpoint at %s', model_path)
    else:
      raise ValueError(
          'The closest next checkpoint is after the pre-defined next at',
          FLAGS.checkpoint_step_next)

  logging.info('data_sources %s',
               gin.query_parameter('inference_pipeline.data_sources'))
  logging.info('data_sources_ref %s',
               gin.query_parameter('inference_pipeline.data_sources_ref'))
  exp_suffix = gin.query_parameter('inference_pipeline.exp_suffix')
  data_sources = gin.query_parameter('inference_pipeline.data_sources')

  if isinstance(data_sources, list) and len(data_sources) > 1:
    category_name = 'mixed'
  else:
    if isinstance(data_sources, list):
      data_sources = data_sources[0]
    category_name = data_sources.split('.sst')[0].split('/')[-1]
  ckpt_suffix = 'ckpt-' + model_path.split('-')[-1]
  inference_base_folder = os.path.join(base_folder, FLAGS.job_name, exp_suffix,
                                       category_name, ckpt_suffix)
  logging.info('inference_base_folder %s', inference_base_folder)

  os.system('mkdir -p ' + inference_base_folder)
  os.system('cp -f ' + gin_configs[-2] + ' ' + inference_base_folder)
  os.system('cp -f ' + gin_configs[-1] + ' ' + inference_base_folder)

  train_eval_lib.inference_pipeline(
      eval_mode=FLAGS.mode,
      data_sources=gin.REQUIRED,
      data_sources_ref=gin.REQUIRED,
      data_filter=gin.REQUIRED,
      model_path=model_path,
      output_path=inference_base_folder,
      optim_mode=gin.REQUIRED,
      n_iterations_per_batch=gin.REQUIRED,
      learning_rate=gin.REQUIRED,
      save_summaries_frequency_latent_optim=gin.REQUIRED,
      timing_frequency=gin.REQUIRED,
      data_sources_type=gin.REQUIRED,
      save_mode_sdf_grid=gin.REQUIRED,
      override_save=gin.REQUIRED)


if __name__ == '__main__':
  app.run(main)
