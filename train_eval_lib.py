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

"""Functions for training/evaluation."""

import functools
import os
from typing import Any, Callable, Dict, Iterable, Optional, Sequence, Text, Tuple, Union

from absl import logging
import gin.tf
import tensorflow as tf
from google3.pyglib import gfile

from google3.vr.perception.volume_compression.mdif.model import dataset_lib
from google3.vr.perception.volume_compression.mdif.model import loss_lib
from google3.vr.perception.volume_compression.mdif.model import network_pipeline
from google3.vr.perception.volume_compression.mdif.utils import misc_utils


def get_strategy(mode: str) -> Any:
  """Creates a distributed strategy."""
  strategy = None
  if mode == 'cpu':
    strategy = tf.distribute.OneDeviceStrategy('/cpu:0')
  elif mode == 'gpu':
    strategy = tf.distribute.MirroredStrategy()
  else:
    raise ValueError('Unsupported distributed mode.')
  return strategy


@tf.function
def _distributed_train_step(
    strategy: tf.distribute.Strategy,
    batch: Dict[str, tf.Tensor],
    models: Sequence[tf.keras.Model],
    loss_function: Callable[..., Dict[Text, Any]],
    optimizers: Sequence[tf.keras.optimizers.Optimizer],
    global_batch_size: int,
    do_eval: bool,
    optim_mode: str = 'full',
    latent_code_type: str = 'train',
    flags: Dict[str, Any] = None,
    params: Dict[str, Any] = None) -> Dict[str, Any]:
  """Distributed training step.

  Args:
    strategy: A Tensorflow distribution strategy.
    batch: A batch of training examples.
    models: The Keras models to train.
    loss_function: The loss used to train the model.
    optimizers: The Keras optimizers used to train the model.
    global_batch_size: The global batch size used to scale the loss.
    do_eval: Whether do evaluation on training data.
    optim_mode: Optimization mode.
    latent_code_type: Type of latent code.
    flags: Flags for this training step.
    params: Other parameters.

  Returns:
    A dictionary of train step outputs.
  """
  model = models[0]
  optimizer = optimizers[0]

  if params is None:
    eval_data_mode = None
  elif 'eval_data_mode' in params:
    eval_data_mode = params['eval_data_mode']
  else:
    eval_data_mode = None

  def _train_step(
      batch: Dict[str, tf.Tensor],
  ) -> Tuple[Dict[str, tf.Tensor], Dict[str, tf.Tensor], Dict[str, tf.Tensor],
             Dict[str, Any]]:
    """Train for one step."""
    with tf.GradientTape() as tape:
      # Copy data to prevent the complaint about changing input.
      model_output = model(batch.copy(),
                           training=True,
                           do_eval=do_eval,
                           optim_mode=optim_mode,
                           latent_code_type=latent_code_type,
                           eval_data_mode=eval_data_mode,
                           flags=flags)
      loss_full = loss_function(model_output['model_outputs_and_targets'],
                                flags=flags)
      loss = loss_full['total_loss']
      loss /= global_batch_size
      loss = tf.debugging.check_numerics(loss, message='image loss is nan')

    # Compute gradients.
    if optim_mode == 'full':
      vars_optim = model.trainable_variables
    elif optim_mode == 'latent_optim':
      if latent_code_type == 'train':
        vars_optim = model.codes_train_data.trainable_variables
      elif latent_code_type == 'test':
        vars_optim = model.codes_test_data.trainable_variables
    grads = tape.gradient(loss, vars_optim)

    hist_summaries = {}
    scalar_summaries = {}

    # Back propagate gradients.
    optimizer.apply_gradients(zip(grads, vars_optim))

    # Post process for vizualization.
    if 'loss_summaries' in loss_full.keys():
      loss_summaries = loss_full['loss_summaries']
      for key in loss_summaries.keys():
        loss_summaries[key] = loss_summaries[key] / global_batch_size
    else:
      loss_summaries = {}
    loss_summaries['loss/total_loss'] = loss
    loss_summaries.update(scalar_summaries)

    image_summaries = model_output['image_summaries']
    if 'image_summaries' in loss_full.keys():
      image_summaries.update(loss_full['image_summaries'])

    # Keep outputs that are used later
    model_output_data = {}
    spatial_dims = model_output['model_outputs_and_targets']['image_size']
    model_output_data['spatial_dims'] = spatial_dims
    model_output_data['num_level'] = model.num_level
    if eval_data_mode == 'all':
      for i in range(model.num_level):
        data_key = 'eval_points_sdf/all_pixels/level' + str(i)
        sdf_map_pred, sdf_map_gt, _, _ = model_output[
            'model_outputs_and_targets'][data_key]
        sdf_map_pred = tf.reshape(
            sdf_map_pred, [tf.shape(sdf_map_pred)[0], *spatial_dims, -1])
        sdf_map_gt = tf.reshape(sdf_map_gt,
                                [tf.shape(sdf_map_gt)[0], *spatial_dims, -1])
        data_save_key = 'sdf_grid_pred/level' + str(i)
        model_output_data[data_save_key] = sdf_map_pred
        data_save_key = 'sdf_grid_gt/level' + str(i)
        model_output_data[data_save_key] = sdf_map_gt

    return (loss_summaries, image_summaries, hist_summaries, model_output_data)

  (loss_summaries, image_summaries, hist_summaries,
   model_output_data) = strategy.run(
       _train_step, args=(batch,))

  # Check whether loss becomes nan
  loss = loss_summaries['loss/total_loss']
  loss = strategy.reduce(tf.distribute.ReduceOp.SUM, loss, axis=None)
  loss = tf.debugging.check_numerics(
      loss, message='loss is nan after strategy.reduce')

  scalar_summaries = {}
  for loss_key in loss_summaries:
    scalar_summaries[loss_key] = strategy.reduce(
        tf.distribute.ReduceOp.SUM, loss_summaries[loss_key], axis=None)

  hist_summaries_reduce = {}
  for hist_key in hist_summaries:
    hist_summaries_reduce[hist_key] = strategy.reduce(
        tf.distribute.ReduceOp.SUM, hist_summaries[hist_key], axis=None)

  for image_key in image_summaries:
    image_summaries[image_key] = tf.concat(
        strategy.experimental_local_results(image_summaries[image_key]), axis=0)

  return {
      'scalar_summaries': scalar_summaries,
      'image_summaries': image_summaries,
      'hist_summaries': hist_summaries_reduce,
      'model_output_data': model_output_data,
  }


def _summary_writer(summaries_dict: Dict[str, Any],
                    step: Optional[int] = None,
                    prefix: Optional[str] = '') -> None:
  """Adds summaries."""
  # Adds scalar summaries.
  if 'scalar_summaries' in summaries_dict.keys():
    for key, scalars in summaries_dict['scalar_summaries'].items():
      tf.summary.scalar(prefix + key, scalars, step=step)
  # Adds image summaries.
  if 'image_summaries' in summaries_dict.keys():
    for key, images in summaries_dict['image_summaries'].items():
      tf.summary.image(prefix + key, images, step=step)
  # Adds histogram summaries.
  if 'hist_summaries' in summaries_dict.keys():
    for key, hists in summaries_dict['hist_summaries'].items():
      tf.summary.histogram(prefix + key, hists, step=step)


def train_loop(
    strategy: tf.distribute.Strategy,
    train_set: Union[tf.data.Dataset, Sequence[Any]],
    train_data_filter: Sequence[Union[str, int]],
    create_model_fns: Sequence[Callable[..., tf.keras.Model]],
    create_loss_fn: Callable[..., Callable[..., Dict[str, Any]]],
    create_optimizer_fns: Sequence[Callable[...,
                                            tf.keras.optimizers.Optimizer]],
    distributed_train_step_fn: Callable[[
        tf.distribute.Strategy, Dict[str, tf.Tensor], tf.keras.Model,
        tf.keras.losses.Loss, tf.keras.optimizers.Optimizer, int, bool
    ], Dict[str, Any]],
    summary_writer_fn: Any,
    global_batch_size: int,
    base_folder: str,
    num_iterations: int,
    num_iterations_per_batch: int = 100,
    save_summaries_frequency: int = 100,
    save_checkpoint_frequency: int = 100,
    checkpoint_max_to_keep: int = 10,
    checkpoint_save_every_n_hours: float = 2.,
    timing_frequency: int = 100):
  """A Tensorflow 2 eager mode training loop.

  Args:
    strategy: A Tensorflow distributed strategy.
    train_set: A dataset to loop through for training.
    train_data_filter: Filtering mode on training data.
    create_model_fns: A list of callable that returns a tf.keras.Model.
    create_loss_fn: A callable that returns a tf.keras.losses.Loss.
    create_optimizer_fns: A list of callable that returns a
      tf.keras.optimizers.Optimizer.
    distributed_train_step_fn: A callable that takes a distribution strategy, a
      Dict[Text, tf.Tensor] holding the batch of training data, a
      tf.keras.Model, a tf.keras.losses.Loss, a tf.keras.optimizers.Optimizer,
      and the global batch size, and returns a dictionary to be passed to the
      summary_writer_fn.
    summary_writer_fn: A callable that takes the output of
      distributed_train_step_fn and writes summaries to be visualized in
      Tensorboard.
    global_batch_size: The global batch size, typically used to scale losses in
      distributed_train_step_fn.
    base_folder: Path where the summaries event files and checkpoints will be
      saved.
    num_iterations: Number of iterations to train for.
    num_iterations_per_batch: Number of optimization steps per data batch.
    save_summaries_frequency: The iteration frequency with which summaries are
      saved.
    save_checkpoint_frequency: The iteration frequency with which model
      checkpoints are saved.
    checkpoint_max_to_keep: The maximum number of checkpoints to keep.
    checkpoint_save_every_n_hours: The frequency in hours to keep checkpoints.
    timing_frequency: The iteration frequency with which to log timing.
  """
  logging.info('Creating summaries ...')
  summary_writer = tf.summary.create_file_writer(base_folder)
  summary_writer.set_as_default()

  if isinstance(train_set, tf.data.Dataset):
    train_set = filter_dataset(train_data_filter, train_set)
    train_set = strategy.experimental_distribute_dataset(train_set)

  with strategy.scope():
    logging.info('Building models ...')

    models = [create_model_fn() for create_model_fn in create_model_fns]
    loss_func = create_loss_fn()
    optimizers = [
        create_optimizer_fn() for create_optimizer_fn in create_optimizer_fns
    ]

    # Create model and optimizer checkpoint variables.
    model_kwargs = {
        'model_%d' % idx: model for (idx, model) in enumerate(models)
    }
    optimizer_kwargs = {
        'optimizer_%d' % idx: optimizer
        for (idx, optimizer) in enumerate(optimizers)
    }

    logging.info('Creating checkpoint ...')
    checkpoint = tf.train.Checkpoint(
        epoch=tf.Variable(0, dtype=tf.int64),
        training_finished=tf.Variable(False, dtype=tf.bool),
        step=optimizers[0].iterations,
        **model_kwargs,
        **optimizer_kwargs)

  logging.info('Restoring old model (if exists) ...')
  checkpoint_manager = tf.train.CheckpointManager(
      checkpoint,
      directory=base_folder,
      max_to_keep=checkpoint_max_to_keep,
      keep_checkpoint_every_n_hours=checkpoint_save_every_n_hours)

  if checkpoint_manager.latest_checkpoint:
    with strategy.scope():
      checkpoint.restore(checkpoint_manager.latest_checkpoint)

  optimizer = optimizers[0]
  logging.info('Creating Timer ...')
  timer = tf.estimator.SecondOrStepTimer(every_steps=timing_frequency)
  timer.update_last_triggered_step(optimizer.iterations.numpy())

  logging.info('Training ...')
  while optimizer.iterations.numpy() < num_iterations:
    logging.info('epoch %d', checkpoint.epoch.numpy())
    for i_batch, batch in enumerate(train_set):
      batch['batch_id'] = tf.constant(i_batch, dtype=tf.int32)

      optim_iters_start = optimizer.iterations.numpy()
      optim_iters_end = optim_iters_start + num_iterations_per_batch

      # Optimize on same batch for a number of steps
      while optimizer.iterations.numpy() < optim_iters_end:
        if (optimizer.iterations.numpy() >= optim_iters_end or
            optimizer.iterations.numpy() >= num_iterations):
          break

        # Log epoch, total iterations and batch index.
        if (optimizer.iterations.numpy() == 1 or
            optimizer.iterations.numpy() % 100 == 0):
          logging.info('epoch %d; iterations %d; i_batch %d',
                       checkpoint.epoch.numpy(), optimizer.iterations.numpy(),
                       i_batch)

        save_summary = False
        if (optimizer.iterations.numpy() + 1) % save_summaries_frequency == 0:
          save_summary = True

        # Compute distributed step outputs.
        distributed_step_outputs = distributed_train_step_fn(
            strategy, batch, models, loss_func, optimizers, global_batch_size,
            save_summary)

        # Save checkpoint.
        if optimizer.iterations.numpy() % save_checkpoint_frequency == 0:
          checkpoint_manager.save(
              checkpoint_number=optimizer.iterations.numpy())

        # Write summaries.
        if save_summary:
          tf.summary.experimental.set_step(step=optimizer.iterations.numpy())
          summary_writer_fn(distributed_step_outputs, prefix='train-')

        # Log steps/sec.
        if timer.should_trigger_for_step(optimizer.iterations.numpy()):
          elapsed_time, elapsed_steps = timer.update_last_triggered_step(
              optimizer.iterations.numpy())
          if elapsed_time is not None:
            steps_per_second = elapsed_steps / elapsed_time
            tf.summary.scalar(
                'steps/sec', steps_per_second, step=optimizer.iterations)

      # Break if the number of iterations exceeds the max.
      if optimizer.iterations.numpy() >= num_iterations:
        break

    # Increment epoch.
    checkpoint.epoch.assign_add(1)

  # Assign training_finished variable to True after training is finished and
  # save the last checkpoint.
  checkpoint.training_finished.assign(True)
  checkpoint_manager.save(checkpoint_number=optimizer.iterations.numpy())


@gin.configurable
def get_training_elements(
    model_component: str,
    model_params: Dict[str, Any],
    loss_params: Dict[str, Any],
    learning_rate: float = 0.0,
    model_params_update: Dict[str, Any] = None,
    loss_params_update: Dict[str,
                             Any] = None) -> Tuple[Any, Any, Any, Any, Any]:
  """Get model architecture, loss, optimizer, and train/eval step functions.

  Args:
    model_component: The model type.
    model_params: The model hyper parameters.
    loss_params: The hyper parameters for the loss.
    learning_rate: The learning rate for setting up optimizer.
    model_params_update: Update for the model hyper parameters.
    loss_params_update: Update for the loss hyper parameters.

  Returns:
    A tuple with functions for:
      A list of models,
      A loss function,
      A list of optimizer,
      A train step function,
      An eval step function.
  """
  if model_params_update is not None:
    model_params.update(model_params_update)
  if loss_params_update is not None:
    loss_params.update(loss_params_update)

  if model_component == 'MultiresDeepImplicitFunction':
    create_model_fn = [
        functools.partial(
            network_pipeline.MultiresDeepImplicitFunction,
            general_params=model_params,
            loss_params=loss_params,
            input_encoder_params=gin.REQUIRED,
            feature_to_code_net_params=gin.REQUIRED,
            decoder_params=gin.REQUIRED,
            train_sampling_params=gin.REQUIRED,
            eval_sampling_params=gin.REQUIRED,
            latent_optim_sampling_params=gin.REQUIRED)
    ]
    create_loss_fn = functools.partial(
        loss_lib.MultiresDeepImplicitLoss,
        model_params=model_params,
        loss_params=loss_params)
    if 'lr_schedule' not in model_params or model_params['lr_schedule'] is None:
      lr_schedule = learning_rate
    else:
      lr_schedule_params = model_params['lr_schedule'].copy()
      lr_schedule_type = lr_schedule_params.pop('type')
      if lr_schedule_type == 'ExponentialDecay':
        lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
            **lr_schedule_params)
      elif lr_schedule_type == 'CosineDecay':
        lr_schedule = tf.keras.experimental.CosineDecay(**lr_schedule_params)
      else:
        raise ValueError('Unknown lr_schedule_type: %s' % lr_schedule_type)
    create_optimizer_fn = [
        functools.partial(
            tf.keras.optimizers.Adam, learning_rate=lr_schedule)
    ]
    distributed_train_step_fn = _distributed_train_step
    distributed_eval_step_fn = _distributed_eval_step
  else:
    raise ValueError('Unknown model_component: %s' % model_component)

  return (create_model_fn, create_loss_fn, create_optimizer_fn,
          distributed_train_step_fn, distributed_eval_step_fn)


@gin.configurable
def train_pipeline(training_mode: str,
                   base_folder: str,
                   data_sources: Union[str, Iterable[str], Sequence[Any]],
                   train_data_filter: Sequence[Union[str, int]],
                   batch_size: int,
                   n_iterations: int,
                   n_iterations_per_batch: int = 100,
                   learning_rate: float = 0.001,
                   save_summaries_frequency: int = 100,
                   save_checkpoint_frequency: int = 100,
                   time_every_n_steps: int = 100,
                   data_sources_type: str = 'load_2d'):
  """A training function that is strategy agnostic.

  Args:
    training_mode: Distributed strategy approach, one from 'cpu', 'gpu'.
    base_folder: Path where the summaries event files and checkpoints will be
      saved.
    data_sources: List of files that make up the training dataset.
    train_data_filter: Filtering mode on training data.
    batch_size: Batch size.
    n_iterations: Number of iterations to train for.
    n_iterations_per_batch: Number of optimization steps per data batch.
    learning_rate: Learning rate.
    save_summaries_frequency: Save summaries every X iterations.
    save_checkpoint_frequency: Save checkpoint every X iterations.
    time_every_n_steps: Report timing every X iterations.
    data_sources_type: Type of data sources.
  """
  logging.info('Loading training data ...')

  if data_sources_type == 'directly_use':
    # data_sources is already a dataset.
    train_set = data_sources
  elif data_sources_type == 'load_2d':
    train_set = dataset_lib.load_dataset_2d(
        data_sources=data_sources, batch_size=batch_size)
  elif data_sources_type == 'load_3d':
    train_set = dataset_lib.load_dataset_3d(
        data_sources=data_sources, batch_size=batch_size, is_training=True)
  else:
    raise ValueError('Unknown data_sources_type: %s' % data_sources_type)

  (create_model_fn, create_loss_fn, create_optimizer_fn,
   distributed_train_step_fn, _) = get_training_elements(
       model_component=gin.REQUIRED,
       model_params=gin.REQUIRED,
       loss_params=gin.REQUIRED,
       learning_rate=learning_rate)

  train_loop(
      strategy=get_strategy(training_mode),
      train_set=train_set,
      train_data_filter=train_data_filter,
      create_model_fns=create_model_fn,
      create_loss_fn=create_loss_fn,
      create_optimizer_fns=create_optimizer_fn,
      distributed_train_step_fn=distributed_train_step_fn,
      summary_writer_fn=_summary_writer,
      global_batch_size=batch_size,
      base_folder=base_folder,
      num_iterations=n_iterations,
      num_iterations_per_batch=n_iterations_per_batch,
      save_summaries_frequency=save_summaries_frequency,
      save_checkpoint_frequency=save_checkpoint_frequency,
      timing_frequency=time_every_n_steps)


def filter_dataset(data_filter: Sequence[Any],
                   dataset: tf.data.Dataset) -> tf.data.Dataset:
  """Filters dataset.

  Args:
    data_filter: Mode for filtering.
    dataset: Dataset to be filtered.

  Returns:
    Filtered dataset.
  """
  if data_filter[0] == 'all':
    return dataset
  elif data_filter[0] == 'first':
    return dataset.take(data_filter[1])
  elif data_filter[0] == 'after':
    return dataset.skip(data_filter[1])
  elif data_filter[0] == 'between':
    return dataset.skip(data_filter[1]).take(data_filter[2])
  else:
    raise ValueError('Unknown data_filter: %s' % data_filter)


@tf.function
def _distributed_eval_step(strategy: tf.distribute.Strategy,
                           batch: Dict[str, tf.Tensor],
                           models: tf.keras.Model,
                           loss_function: Callable[..., Dict[str, Any]],
                           params: Dict[str, Any] = None) -> Dict[str, Any]:
  """Distributed eval step.

  Args:
    strategy: A Tensorflow distribution strategy.
    batch: A batch of examples.
    models: The Keras models to evaluate.
    loss_function: The loss used to evaluate the model.
    params: Other parameters.

  Returns:
    A dictionary holding summaries.
  """
  def _eval_step(batch: Dict[str, tf.Tensor]):
    """Eval for one step."""
    # Copy data to prevent the complaint about changing input.
    model_output = models[0](batch.copy(),
                             training=False, do_eval=True,
                             eval_data_mode=params['eval_data_mode'])
    loss_full = loss_function(
        model_output['model_outputs_and_targets'], mode='metric')

    if loss_full['total_loss'] is not None:
      loss_summaries = {'loss/total_loss': loss_full['total_loss']}
    else:
      loss_summaries = {}

    image_summaries = model_output['image_summaries']

    # Keep outputs that are used later
    model_output_data = {}
    spatial_dims = model_output['model_outputs_and_targets']['image_size']
    model_output_data['spatial_dims'] = spatial_dims
    model_output_data['num_level'] = models[0].num_level
    for i in range(models[0].num_level):
      data_key = 'eval_points_sdf/all_pixels/level' + str(i)
      sdf_map_pred, sdf_map_gt, _, _ = model_output[
          'model_outputs_and_targets'][data_key]
      if params['eval_data_mode'] == 'all':
        sdf_map_pred = tf.reshape(
            sdf_map_pred, [tf.shape(sdf_map_pred)[0], *spatial_dims, -1])
        sdf_map_gt = tf.reshape(sdf_map_gt,
                                [tf.shape(sdf_map_gt)[0], *spatial_dims, -1])
      data_save_key = 'sdf_grid_pred/level' + str(i)
      model_output_data[data_save_key] = sdf_map_pred
      data_save_key = 'sdf_grid_gt/level' + str(i)
      model_output_data[data_save_key] = sdf_map_gt

    if 'loss_summaries' in loss_full.keys():
      loss_summaries.update(loss_full['loss_summaries'])

    if 'image_summaries' in loss_full.keys():
      image_summaries.update(loss_full['image_summaries'])

    return (loss_summaries, image_summaries, model_output_data)

  (loss_summaries, image_summaries, model_output_data) = strategy.run(
      _eval_step, args=(batch,))
  scalar_summaries = {}
  for scalar_key in loss_summaries:
    scalar_summaries[scalar_key] = strategy.reduce(
        tf.distribute.ReduceOp.MEAN, loss_summaries[scalar_key], axis=None)

  for image_key in image_summaries:
    image_summaries[image_key] = tf.concat(
        strategy.experimental_local_results(image_summaries[image_key]), axis=0)

  return {
      'scalar_summaries': scalar_summaries,
      'image_summaries': image_summaries,
      'model_output_data': model_output_data,
  }


@gin.configurable
def eval_pipeline(eval_mode: str,
                  data_sources: Dict[str, Any],
                  eval_data_filter: Sequence[Union[str, int]],
                  train_base_folder: str,
                  eval_base_folder: str,
                  batch_size: int,
                  eval_name: str,
                  optim_mode: str = 'feed_forward',
                  n_iterations_per_batch: int = 100,
                  learning_rate: float = 0.001,
                  data_sources_type: str = 'load_2d',
                  save_mode_sdf_grid: Sequence[Union[str, int]] = None,
                  only_eval_one_ckpt: bool = False):
  """A eval function that is strategy agnostic.

  Args:
    eval_mode: Distributed strategy approach, one from 'cpu', 'gpu'.
    data_sources: Dictionary of files that make up the dataset for experiments.
    eval_data_filter: Filtering mode of evaluation data.
    train_base_folder: Path to the training checkpoints to be loaded.
    eval_base_folder: Path where the evaluation summaries event files will be
      saved.
    batch_size: Batch size.
    eval_name: Experiment name.
    optim_mode: Optimization mode, one of [`feed_forward`, `latent_optim`].
    n_iterations_per_batch: Number of optimization steps per data batch.
    learning_rate: Learning rate for optimization.
    data_sources_type: Type of data sources.
    save_mode_sdf_grid: Mode for saving output SDF grid.
    only_eval_one_ckpt: Whether only evaluate one checkpoint.
  """
  strategy = get_strategy(eval_mode)

  logging.info('Creating summaries ...')
  summary_writer = tf.summary.create_file_writer(eval_base_folder)
  summary_writer.set_as_default()

  logging.info('Loading testing data ...')
  data_sources = data_sources[eval_name]
  if data_sources_type == 'directly_use':
    # data_sources is already a dataset.
    test_set = data_sources
  elif data_sources_type == 'load_2d':
    test_set = dataset_lib.load_dataset_2d(
        data_sources=data_sources, batch_size=batch_size)
  elif data_sources_type == 'load_3d':
    test_set = dataset_lib.load_dataset_3d(
        data_sources=data_sources, batch_size=batch_size, is_training=False)
  else:
    raise ValueError('Unknown data_sources_type: %s' % data_sources_type)

  if isinstance(test_set, tf.data.Dataset):
    test_set = filter_dataset(eval_data_filter, test_set)
    test_set = strategy.experimental_distribute_dataset(test_set)

  (create_model_fns, create_loss_fn, create_optimizer_fns,
   distributed_train_step_fn,
   distributed_eval_step_fn) = get_training_elements(
       model_component=gin.REQUIRED,
       model_params=gin.REQUIRED,
       loss_params=gin.REQUIRED,
       learning_rate=learning_rate)

  with strategy.scope():
    logging.info('Building models ...')
    models = []
    for create_model_fn in create_model_fns:
      models.append(create_model_fn())
    loss_func = create_loss_fn()
    optimizers = [
        create_optimizer_fn() for create_optimizer_fn in create_optimizer_fns
    ]

    model_kwargs = {
        'model_%d' % idx: model for (idx, model) in enumerate(models)
    }

  checkpoint = tf.train.Checkpoint(
      step=tf.Variable(-1, dtype=tf.int64),
      training_finished=tf.Variable(False, dtype=tf.bool),
      **model_kwargs,
  )

  for checkpoint_path in tf.train.checkpoints_iterator(
      train_base_folder,
      min_interval_secs=10,
      timeout=None,
      timeout_fn=lambda: checkpoint.training_finished):

    try:
      status = checkpoint.restore(checkpoint_path)
      status.expect_partial()
    except (tf.errors.NotFoundError, AssertionError) as err:
      logging.info('Failed to restore checkpoint from %s. Error:\n%s',
                   checkpoint_path, err)
      continue

    logging.info('Restoring checkpoint %s @ step %d.', checkpoint_path,
                 checkpoint.step)

    # Determine whether save SDF grid for this checkpoint.
    save_sdf_grid = False
    if (save_mode_sdf_grid[0] == 'every' and
        checkpoint.step % save_mode_sdf_grid[1] == 0):
      save_sdf_grid = True
    elif save_mode_sdf_grid[0] == 'always':
      save_sdf_grid = True

    eval_data_mode = None
    if save_sdf_grid:
      eval_data_mode = 'all'
    params = {'eval_data_mode': eval_data_mode}

    logging.info('Evaluating ...')

    eval_record = {}
    eval_batch_scalar = {}
    for i_batch, batch in enumerate(test_set):
      batch['batch_id'] = tf.constant(i_batch, dtype=tf.int32)

      if optim_mode == 'feed_forward':
        logging.info('iterations %d; i_batch %d', checkpoint.step, i_batch)

        distributed_step_outputs = distributed_eval_step_fn(
            strategy, batch, models, loss_func, params=params)
      elif optim_mode == 'latent_optim':
        # Reset latent codes.
        with strategy.scope():
          models[0].reset_latent_codes(latent_code_type='test')

        optim_iters_start = optimizers[0].iterations.numpy()
        optim_iters_end = optim_iters_start + n_iterations_per_batch
        do_eval = False

        while optimizers[0].iterations.numpy() < optim_iters_end:
          if optimizers[0].iterations.numpy() == optim_iters_end - 1:
            do_eval = True
          elif optimizers[0].iterations.numpy() >= optim_iters_end:
            break

          logging.info('iterations %d; i_batch %d; i_optim_iter %d',
                       checkpoint.step, i_batch,
                       optimizers[0].iterations.numpy() - optim_iters_start)

          distributed_step_outputs = distributed_train_step_fn(
              strategy, batch, models, loss_func, optimizers, batch_size,
              do_eval, optim_mode, latent_code_type='test', params=params)
      else:
        raise ValueError('Unknown optim_mode: %s' % optim_mode)

      if i_batch == 0:
        eval_record['image_summaries'] = distributed_step_outputs[
            'image_summaries']
      for key, scalar in distributed_step_outputs['scalar_summaries'].items():
        if key in eval_batch_scalar:
          eval_batch_scalar[key].append(scalar)
        else:
          eval_batch_scalar[key] = [scalar]

      # Save SDF grid to file.
      model_output_data = distributed_step_outputs['model_output_data']
      save_base_folder = os.path.join(eval_base_folder,
                                      'ckpt-' + str(checkpoint.step.numpy()))
      if save_sdf_grid:
        save_sdf_grid_to_file(batch, model_output_data, save_base_folder)

    # Average scalar items over all samples.
    eval_record['scalar_summaries'] = {}
    for key, record in eval_batch_scalar.items():
      eval_record['scalar_summaries'][key] = tf.reduce_mean(
          tf.boolean_mask(record, tf.math.is_finite(record)))

    if eval_data_mode == 'all':
      summary_prefix = 'test_full-'
    else:
      summary_prefix = 'test-'
    _summary_writer(eval_record, step=checkpoint.step, prefix=summary_prefix)

    if only_eval_one_ckpt:
      return


@gin.configurable
def inference_pipeline(eval_mode: str,
                       data_sources: Any,
                       data_sources_ref: Any,
                       data_filter: Sequence[Union[str, int]],
                       model_path: str,
                       output_path: str,
                       optim_mode: Text = 'feed_forward',
                       n_iterations_per_batch: int = 100,
                       learning_rate: float = 0.001,
                       save_summaries_frequency_latent_optim: int = 100,
                       timing_frequency: int = 100,
                       data_sources_type: str = 'load_2d',
                       save_mode_sdf_grid: Sequence[Union[str, int]] = None,
                       override_save: bool = False,
                       max_num_batch_for_summary: int = 20,
                       exp_suffix: str = None,
                       params_update: Dict[str, Any] = None):
  """Inference model and save result SDF grid.

  Args:
    eval_mode: Distributed strategy approach, one from 'cpu', 'gpu'.
    data_sources: List of files that make up the dataset for experiments.
    data_sources_ref: List of files that make up the reference dataset for
      experiments.
    data_filter: Filtering mode of testing data.
    model_path: Path to the model to be used for inference.
    output_path: Path to save results.
    optim_mode: Optimization mode, one of [`feed_forward`, `latent_optim`].
    n_iterations_per_batch: Number of optimization steps per data batch.
    learning_rate: Learning rate for optimization.
    save_summaries_frequency_latent_optim: Save summaries every X iterations of
      latent optimization.
    timing_frequency: The iteration frequency with which to log timing.
    data_sources_type: Type of data sources.
    save_mode_sdf_grid: Mode for saving output SDF grid during latent
      optimization.
    override_save: Whether override existing SDF grid files.
    max_num_batch_for_summary: Maximum number of batches to record summaries.
    exp_suffix: Suffix for this inference experiment, used by gin config.
    params_update: Updates for parameters, used by gin config.
  """
  del exp_suffix
  del params_update

  if not gfile.IsDirectory(output_path):
    gfile.MakeDirs(output_path)

  strategy = get_strategy(eval_mode)

  logging.info('Loading testing data ...')
  if data_sources_type == 'directly_use':
    # data_sources is already a dataset.
    test_set = data_sources
  elif data_sources_type == 'load_2d':
    test_set = dataset_lib.load_dataset_2d(
        data_sources=data_sources, batch_size=1)
  elif data_sources_type == 'load_3d':
    test_set = dataset_lib.load_dataset_3d(
        data_sources=data_sources, batch_size=1, is_training=False)
  else:
    raise ValueError('Unknown data_sources_type: %s' % data_sources_type)

  assert (isinstance(test_set, tf.data.Dataset) or
          (isinstance(test_set, Sequence) and isinstance(test_set[0], Dict)))

  if isinstance(test_set, tf.data.Dataset):
    test_set = filter_dataset(data_filter, test_set)
    test_set = strategy.experimental_distribute_dataset(test_set)

  logging.info('Loading reference data ...')
  if data_sources_type == 'directly_use':
    # data_sources is already a dataset.
    ref_set = data_sources_ref
  elif data_sources_type == 'load_2d':
    ref_set = dataset_lib.load_dataset_2d(
        data_sources=data_sources_ref, batch_size=1)
  elif data_sources_type == 'load_3d':
    ref_set = dataset_lib.load_dataset_3d(
        data_sources=data_sources_ref, batch_size=1, is_training=False)
  else:
    raise ValueError('Unknown data_sources_type: %s' % data_sources_type)

  # Extract the first batch in reference dataset as reference data
  for i_batch, batch in enumerate(ref_set):
    ref_batch = batch
    break

  assert isinstance(ref_batch, Dict)

  (create_model_fns, create_loss_fn, create_optimizer_fns,
   distributed_train_step_fn, distributed_eval_step_fn) = get_training_elements(
       model_component=gin.REQUIRED,
       model_params=gin.REQUIRED,
       loss_params=gin.REQUIRED,
       learning_rate=learning_rate,
       model_params_update=gin.REQUIRED,
       loss_params_update=gin.REQUIRED,
   )

  with strategy.scope():
    logging.info('Building models ...')
    models = []
    for create_model_fn in create_model_fns:
      models.append(create_model_fn())
    loss_func = create_loss_fn()
    optimizers = [
        create_optimizer_fn() for create_optimizer_fn in create_optimizer_fns
    ]

    model_kwargs = {
        'model_%d' % idx: model for (idx, model) in enumerate(models)
    }
    optimizer_kwargs = {
        'optimizer_%d' % idx: optimizer
        for (idx, optimizer) in enumerate(optimizers)
    }

  model = models[0]
  optimizer = optimizers[0]

  checkpoint = tf.train.Checkpoint(
      step=tf.Variable(-1, dtype=tf.int64),
      training_finished=tf.Variable(False, dtype=tf.bool),
      **model_kwargs,
  )

  status = checkpoint.restore(model_path)
  status.expect_partial()
  logging.info('Restoring checkpoint %s @ step %d.', model_path,
               checkpoint.step)

  logging.info('Creating Timer ...')
  iter_id_cumul = 0
  timer = tf.estimator.SecondOrStepTimer(every_steps=timing_frequency)
  timer.update_last_triggered_step(iter_id_cumul)

  logging.info(
      'Creating summaries to record latent optimization for each data...')
  summary_writer = tf.summary.create_file_writer(output_path)
  summary_writer.set_as_default()

  logging.info('Evaluating ...')

  eval_record = {}
  eval_batch_scalar = {}
  for i_batch, batch in enumerate(test_set):
    # Check whether output SDF grid files already exist.
    exist_out_files = True
    for i in range(model.num_level):
      for ith_data in range(batch['data_key'].shape[0]):
        save_fp = os.path.join(
            output_path, 'sdf_grid_level' + str(i),
            batch['data_key'][ith_data].numpy().decode('utf8') + '.grd')
        if not gfile.Exists(save_fp):
          exist_out_files = False
          break
    if exist_out_files and not override_save:
      continue

    batch['batch_id'] = tf.constant(i_batch, dtype=tf.int32)

    if optim_mode == 'feed_forward':
      logging.info('iterations %d; i_batch %d', checkpoint.step, i_batch)

      params = {'eval_data_mode': 'all'}
      distributed_step_outputs = distributed_eval_step_fn(
          strategy, batch, models, loss_func, params=params)
    elif optim_mode == 'latent_optim':
      # Reset optimizer.
      logging.info('Reset optimizers ...')
      for var in optimizer.variables():
        var.assign(tf.zeros_like(var))
      logging.info('optimizer.iterations %d', optimizer.iterations.numpy())

      # Reset latent codes.
      with strategy.scope():
        if model.codes_init_from_encoder:
          if model.codes_init_from_ref:
            batch_use = ref_batch.copy()
          else:
            batch_use = batch.copy()

          # Run one feed forward to obtain initial latent codes.
          reset_data = model(
              batch_use, training=True, do_eval=False,
              optim_mode='full')['model_outputs_and_targets']
        else:
          reset_data = None
        init_data = model.reset_latent_codes(
            latent_code_type='test', reset_data=reset_data)

      # Save initial latent codes to summaries.
      if i_batch < max_num_batch_for_summary:
        init_summary = {'image_summaries': {}}
        for key, item in init_data.items():
          summary_key = 'misc/' + key
          image_summaries_update = misc_utils.get_image_summary(
              summary_key,
              item,
              channels_use='first',
              spatial_dims=None,
              normalize=True,
              summary_config=model.summary_config)
          init_summary['image_summaries'].update(image_summaries_update)
        _summary_writer(
            init_summary,
            step=0,
            prefix='infer_lopt-data_' + str(i_batch) + '-')

      optim_iters_start = optimizer.iterations.numpy()
      optim_iters_end = optim_iters_start + n_iterations_per_batch

      # Start latent optimization.
      while optimizer.iterations.numpy() < optim_iters_end:
        iter_id_cumul += 1
        iter_id = optimizer.iterations.numpy() - optim_iters_start
        do_eval = False

        # Determine whether compute consistency loss and symmetry loss for this
        #  iteration.
        flags = {'consistency_loss': False, 'symmetry_loss': False}
        if ('sdf_consistency_l1' in model.loss_params and
            'mode' in model.loss_params['sdf_consistency_l1']):
          loss_mode = model.loss_params['sdf_consistency_l1']['mode']
          if loss_mode[0] == 'every' and (iter_id + 1) % loss_mode[1] == 0:
            flags['consistency_loss'] = True
          elif loss_mode[0] == 'after' and (iter_id + 1) > loss_mode[1]:
            flags['consistency_loss'] = True
        if ('sdf_symmetry_l1' in model.loss_params and
            'mode' in model.loss_params['sdf_symmetry_l1']):
          loss_mode = model.loss_params['sdf_symmetry_l1']['mode']
          if loss_mode[0] == 'every' and (iter_id + 1) % loss_mode[1] == 0:
            flags['symmetry_loss'] = True
          elif loss_mode[0] == 'after' and (iter_id + 1) > loss_mode[1]:
            flags['symmetry_loss'] = True

        # Determine whether save summary for this iteration.
        save_summary = False
        if (iter_id + 1) % save_summaries_frequency_latent_optim == 0:
          save_summary = True
        if optimizer.iterations.numpy() == optim_iters_end - 1:
          save_summary = True

        # Determine whether save SDF grid for this iteration.
        save_sdf_grid = False
        if save_mode_sdf_grid is None:
          pass
        elif save_mode_sdf_grid[0] == 'every':
          if (iter_id + 1) % save_mode_sdf_grid[1] == 0:
            save_sdf_grid = True
        if optimizer.iterations.numpy() == optim_iters_end - 1:
          save_sdf_grid = True

        # Update do_eval.
        if save_summary or save_sdf_grid:
          do_eval = True

        # Update eval_data_mode.
        eval_data_mode = None
        if save_sdf_grid:
          eval_data_mode = 'all'
        params = {'eval_data_mode': eval_data_mode}

        if iter_id % 100 == 0:
          logging.info('iterations %d; i_batch %d; i_optim_iter %d',
                       checkpoint.step, i_batch, iter_id)

        batch_size = 1
        distributed_step_outputs = distributed_train_step_fn(
            strategy,
            batch,
            models,
            loss_func,
            optimizers,
            batch_size,
            do_eval,
            optim_mode,
            latent_code_type='test',
            flags=flags,
            params=params)

        iter_id_new = optimizer.iterations.numpy() - optim_iters_start

        # Write summaries for this latent optimization iteration.
        if save_summary and i_batch < max_num_batch_for_summary:
          summary_prefix = 'infer_lopt-data_' + str(i_batch) + '-'
          if eval_data_mode == 'all':
            summary_prefix = 'infer_lopt_full-data_' + str(i_batch) + '-'
          _summary_writer(
              distributed_step_outputs, step=iter_id_new, prefix=summary_prefix)

        # Save SDF grid for this latent optimization iteration.
        save_base_folder = os.path.join(output_path,
                                        'optim-' + str(iter_id_new))
        if save_sdf_grid:
          save_sdf_grid_to_file(batch,
                                distributed_step_outputs['model_output_data'],
                                save_base_folder)

        # Log steps/sec.
        if timer.should_trigger_for_step(iter_id_cumul):
          elapsed_time, elapsed_steps = timer.update_last_triggered_step(
              iter_id_cumul)
          if elapsed_time is not None:
            steps_per_second = elapsed_steps / elapsed_time
            tf.summary.scalar('steps/sec', steps_per_second, step=iter_id_cumul)
    else:
      raise ValueError('Unknown optim_mode: %s' % optim_mode)

    # Write summaries for this data batch.
    _summary_writer(
        distributed_step_outputs, step=i_batch, prefix='infer_final-')

    if i_batch == 0:
      eval_record['image_summaries'] = distributed_step_outputs[
          'image_summaries']
    for key, scalar in distributed_step_outputs['scalar_summaries'].items():
      if key in eval_batch_scalar:
        eval_batch_scalar[key].append(scalar)
      else:
        eval_batch_scalar[key] = [scalar]

    # Save final SDF grid of this batch.
    save_sdf_grid_to_file(batch, distributed_step_outputs['model_output_data'],
                          output_path)

  # Average scalar items over all samples.
  eval_record['scalar_summaries'] = {}
  for key, record in eval_batch_scalar.items():
    eval_record['scalar_summaries'][key] = tf.reduce_mean(
        tf.boolean_mask(record, tf.math.is_finite(record)))

  logging.info('Creating summaries to record mean stats...')
  output_mean_path = os.path.join(output_path.rpartition('/')[0], 'mean')
  if not gfile.IsDirectory(output_mean_path):
    gfile.MakeDirs(output_mean_path)
  summary_writer_mean = tf.summary.create_file_writer(output_mean_path)
  summary_writer_mean.set_as_default()

  _summary_writer(eval_record, step=checkpoint.step, prefix='infer_final_mean-')


def save_sdf_grid_to_file(batch: Dict[str, Any],
                          model_output_data: Dict[str, Any],
                          save_base_folder: str):
  """Saves SDF grid to file."""
  logging.info('Saving SDF grid to file')
  if 'world2grid' in batch:
    world2grid = batch['world2grid']
  else:
    world2grid = None
  for i in range(model_output_data['num_level']):
    save_folder = os.path.join(save_base_folder, 'sdf_grid_level' + str(i))
    if not gfile.IsDirectory(save_folder):
      gfile.MakeDirs(save_folder)

    save_fp = [
        save_folder + '/' + batch['data_key'][ith_data].numpy().decode('utf8') +
        '.grd' for ith_data in range(batch['data_key'].shape[0])
    ]
    sdf_grid = model_output_data['sdf_grid_pred/level' + str(i)]
    misc_utils.write_grd_batch(save_fp, sdf_grid, world2grid)
