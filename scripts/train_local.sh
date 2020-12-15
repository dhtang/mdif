#!/bin/bash

source gbash.sh || exit

DEFINE_string mode "gpu" "Training mode."
DEFINE_string base_folder "${HOME}/mdif/test_0" "Base folder of model."
DEFINE_string gin_folder "${PWD%%google3*}google3/vr/perception/volume_compression/mdif/gin_config" "Folder of gin config files."
DEFINE_string model_gin "${FLAGS_gin_folder}/model_config.gin" "Gin configuration for model."
DEFINE_string train_eval_gin "${FLAGS_gin_folder}/train_eval_config_local.gin" "Gin configuration for train_eval."

# Exit if anything fails.
set -e

mkdir -p "$FLAGS_base_folder"
mkdir -p "$FLAGS_base_folder/gin_config"

model_gin="$FLAGS_base_folder/gin_config/model_config.gin"
train_eval_gin="$FLAGS_base_folder/gin_config/train_eval_config_local.gin"

cp -f "${FLAGS_model_gin}" "${model_gin}"
cp -f "${FLAGS_train_eval_gin}" "${train_eval_gin}"

blaze run -c opt --copt=-mavx --define cuda_target_sm75=1 --config=cuda \
  vr/perception/volume_compression/mdif:train_main_local -- \
  --base_folder="$FLAGS_base_folder" \
  --mode="$FLAGS_mode" \
  --gin_configs="${model_gin}" \
  --gin_configs="${train_eval_gin}" \
  --alsologtostderr
