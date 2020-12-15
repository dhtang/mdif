#!/bin/bash

source gbash.sh || exit

DEFINE_string base_folder "${HOME}/mdif/test_0" "Base folder of model."

DEFINE_string categories_base_folder '/cns/tp-d/home/danhangtang/shapenet/sstables/3d-r2n2'
DEFINE_string category "test_03001627"

DEFINE_string checkpoint_step "6" "Checkpoint step."
DEFINE_string checkpoint_step_next "none" "Checkpoint step."
DEFINE_string task "lopt" "Inference tasks to carry out."

category_fp="${FLAGS_categories_base_folder}/${FLAGS_category}.sst-000??-of-00020"

DEFINE_string mode "cpu" "Training mode."
DEFINE_string model_gin "${FLAGS_base_folder}/gin_config/model_config.gin" "Gin configuration for model."
DEFINE_string train_eval_gin "${FLAGS_base_folder}/gin_config/train_eval_config_local.gin" "Gin configuration for train_eval."

DEFINE_string gin_folder "${PWD%%google3*}google3/vr/perception/volume_compression/mdif/gin_config" "Folder of gin config files."
DEFINE_string infer_config_share "${FLAGS_gin_folder}/infer_config_share.gin" "Shared gin configuration for inference."
DEFINE_string infer_config_task_base "${FLAGS_gin_folder}/infer_config_" "Gin configuration for task."

# Exit if anything fails.
set -e

if  [[ $FLAGS_task == forward* ]] ;
then
  job_name="forward"
elif [[ $FLAGS_task == lopt* ]] ;
then
  job_name="lopt"
elif [[ $FLAGS_task == compl_dp* ]] ;
then
  job_name="compl_dp"
fi

blaze run -c opt --copt=-mavx --define cuda_target_sm75=1 --config=cuda \
  vr/perception/volume_compression/mdif:inference_main_local -- \
  --mode="$FLAGS_mode" \
  --base_folder="$FLAGS_base_folder" \
  --job_name="$job_name" \
  --checkpoint_step="$FLAGS_checkpoint_step" \
  --checkpoint_step_next="$FLAGS_checkpoint_step_next" \
  --gin_configs="${FLAGS_model_gin}" \
  --gin_configs="${FLAGS_train_eval_gin}" \
  --gin_configs="${FLAGS_infer_config_share}" \
  --gin_configs="${FLAGS_infer_config_task_base}${FLAGS_task}.gin" \
  --gin_bindings="inference_pipeline.data_sources = ['${category_fp}',]" \
  --gin_bindings="inference_pipeline.data_sources_ref = ['${category_fp}',]" \
  --alsologtostderr
