#!/bin/bash

source gbash.sh || exit

DEFINE_string exp_folder "mdif" "Folder of the experiment."
DEFINE_string name "test_0" "Name of the experiment."

DEFINE_string xm_pool "vr" ""
DEFINE_string cell "tp" "Cell to run."
DEFINE_string gin_folder "vr/perception/volume_compression/mdif/gin_config" "Folder of gin config files."
DEFINE_string model_gin "${FLAGS_gin_folder}/model_config.gin" "Gin configuration for model."
DEFINE_string train_eval_gin "${FLAGS_gin_folder}/train_eval_config.gin" "Gin configuration for train_eval."
DEFINE_string xm_configs "${FLAGS_gin_folder}/xm_gpu.gin" "Gin for xm launch."

# Exit if anything fails.
set -e

gbash::init_google "$@"

DATETIME=$(date +'%Y-%m-%d-%H-%M-%S')
EXPERIMENT_NAME="${DATETIME}-${FLAGS_name}"
ROOT_DIR="/cns/${FLAGS_cell}-d/home/${USER}/${FLAGS_exp_folder}/${EXPERIMENT_NAME}"

fileutil mkdir -p "$ROOT_DIR"
fileutil mkdir -p "$ROOT_DIR/gin_config"

fileutil cp -f "${FLAGS_model_gin}" "$ROOT_DIR/gin_config"
fileutil cp -f "${FLAGS_train_eval_gin}" "$ROOT_DIR/gin_config"
fileutil cp -f "${FLAGS_xm_configs}" "$ROOT_DIR/gin_config"

if [[ -r /google/data/ro/teams/dmgi/configs/google_xm_bashrc ]] ; then
  source /google/data/ro/teams/dmgi/configs/google_xm_bashrc
fi

google_xmanager launch vr/perception/volume_compression/mdif/xmanager/xm_launch.py -- \
  --xm_skip_launch_confirmation \
  --noxm_monitor_on_launch \
  --xm_resource_pool="${FLAGS_xm_pool}"\
  --xm_resource_alloc=user:"${FLAGS_xm_pool}/${USER}" \
  --xm_configs="${FLAGS_xm_configs}" \
  --gin_configs="${FLAGS_model_gin}" \
  --gin_configs="${FLAGS_train_eval_gin}" \
  --xm_bindings="experiment.experiment_name = '${EXPERIMENT_NAME}'" \
  --xm_bindings="experiment.base_folder = '${ROOT_DIR}'" \
  --xm_bindings="train/borg.cell = '${FLAGS_cell}'" \
  --gfs_user='vr-beaming'
