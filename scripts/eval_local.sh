#!/bin/bash

source gbash.sh || exit

DEFINE_string mode "cpu" "Training mode."
DEFINE_string base_folder "${HOME}/mdif/test_0" "Base folder of model."
DEFINE_string job_name "eval_test_set" "The eval job name."

# Exit if anything fails.
set -e

model_gin="$FLAGS_base_folder/gin_config/model_config.gin"
train_eval_gin="$FLAGS_base_folder/gin_config/train_eval_config_local.gin"

blaze run -c opt --copt=-mavx --define cuda_target_sm75=1 --config=cuda \
  vr/perception/volume_compression/mdif:eval_main_local -- \
  --base_folder="$FLAGS_base_folder" \
  --job_name="$FLAGS_job_name" \
  --mode="$FLAGS_mode" \
  --gin_configs="${model_gin}" \
  --gin_configs="${train_eval_gin}" \
  --alsologtostderr
