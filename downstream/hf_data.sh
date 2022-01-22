function fulldata() {
DATA=$1
SAVING_PATH=$2
METRIC=$3
PRETRAIN_MODEL_PATH=$4
TRAIN_BATCH_SIZE=$5
EVAL_BATCH_SIZE=$6
TRAIN_EPOCH=$7
GRAD_ACCU_STEPS=$8
LOGGING_STEPS=${9}
LR=${10}
SAVING_STEPS=${11}
EVAL_STEPS=${12}
EVAL_STRATEGY=${13}
SAVE_STRATEGY=${14}
LOAD_FROM_FORMAT_TASK_ID=${15}

mkdir -p ${SAVING_PATH}

python -m torch.distributed.launch --nproc_per_node=8 run_t5_softprompt_yujia.py \
  --load_from_format_task_id ${LOAD_FROM_FORMAT_TASK_ID} \
  --model_name_or_path ${PRETRAIN_MODEL_PATH} \
  --output_dir ${SAVING_PATH} \
  --do_train \
  --do_eval \
  --dataset_name ${DATA} \
  --per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
  --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
  --overwrite_output_dir \
  --gradient_accumulation_steps ${GRAD_ACCU_STEPS} \
  --num_train_epochs ${TRAIN_EPOCH} \
  --warmup_ratio 0.1 \
  --logging_steps ${LOGGING_STEPS} \
  --learning_rate ${LR} \
  --save_steps ${SAVING_STEPS} \
  --eval_steps ${EVAL_STEPS} \
  --evaluation_strategy ${EVAL_STRATEGY} \
  --save_strategy ${SAVE_STRATEGY} \
  --predict_with_generate \
  --num_beams 4 \
  --weight_decay 1e-2 \
  --max_source_length 512 \
  --label_smoothing_factor 0.1 \
  --do_lowercase True \
  --metric_for_best_model ${METRIC}  \
  --load_best_model_at_end True \
  --greater_is_better True \
  --save_total_limit 10 \
  --ddp_find_unused_parameters False 2>&1 | tee ${SAVING_PATH}/log
}


function fewshot() {
DATA=$1
SAVING_PATH=$2
METRIC=$3
PRETRAIN_MODEL_PATH=$4
TRAIN_BATCH_SIZE=$5
EVAL_BATCH_SIZE=$6
TRAIN_EPOCH=$7
GRAD_ACCU_STEPS=$8
LOGGING_STEPS=${9}
LR=${10}
SAVING_STEPS=${11}
EVAL_STEPS=${12}
EVAL_STRATEGY=${13}
SAVE_STRATEGY=${14}
LOAD_FROM_FORMAT_TASK_ID=${15}

mkdir -p ${SAVING_PATH}

python -m torch.distributed.launch --nproc_per_node=8 run_t5_softprompt_yujia.py \
  --load_from_format_task_id ${LOAD_FROM_FORMAT_TASK_ID} \
  --max_train_samples 32 \
  --model_name_or_path ${PRETRAIN_MODEL_PATH} \
  --output_dir ${SAVING_PATH} \
  --do_train \
  --do_eval \
  --dataset_name ${DATA} \
  --per_device_train_batch_size 1 \
  --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
  --overwrite_output_dir \
  --gradient_accumulation_steps ${GRAD_ACCU_STEPS} \
  --num_train_epochs ${EPOCH_NUM} \
  --warmup_ratio 0.1 \
  --logging_steps ${LOGGING_STEPS} \
  --learning_rate 1e-5 \
  --save_steps `expr ${EPOCH_NUM} / 10` \
  --eval_steps `expr ${EPOCH_NUM} / 10 ` \
  --evaluation_strategy steps \
  --save_strategy steps \
  --predict_with_generate \
  --num_beams 4 \
  --weight_decay 1e-2 \
  --max_source_length 512 \
  --label_smoothing_factor 0.1 \
  --metric_for_best_model ${METRIC}  \
  --load_best_model_at_end True \
  --greater_is_better True \
  --save_total_limit 1 \
  --ddp_find_unused_parameters False 2>&1 | tee ${SAVING_PATH}/log
}

function zeroshot() {
DATA=$1
SAVING_PATH=$2
METRIC=$3
PRETRAIN_MODEL_PATH=$4
TRAIN_BATCH_SIZE=$5
EVAL_BATCH_SIZE=$6
TRAIN_EPOCH=$7
GRAD_ACCU_STEPS=$8
LOGGING_STEPS=${9}
LR=${10}
SAVING_STEPS=${11}
EVAL_STEPS=${12}
EVAL_STRATEGY=${13}
SAVE_STRATEGY=${14}
LOAD_FROM_FORMAT_TASK_ID=${15}


mkdir -p ${SAVING_PATH}

python -m torch.distributed.launch --nproc_per_node=8 run_t5_softprompt_yujia.py \
  --load_from_format_task_id ${LOAD_FROM_FORMAT_TASK_ID} \
  --model_name_or_path ${PRETRAIN_MODEL_PATH} \
  --output_dir ${SAVING_PATH} \
  --do_eval \
  --dataset_name ${DATA} \
  --per_device_eval_batch_size ${EVAL_BATCH_SIZE} \
  --predict_with_generate \
  --num_beams 4 \
  --weight_decay 1e-2 \
  --max_source_length 512 \
  --label_smoothing_factor 0.1 \
  --do_lowercase True \
  --overwrite_cache True \
  --metric_for_best_model ${METRIC}  \
  --greater_is_better True 2>&1 | tee ${SAVING_PATH}/log
}


MODE=$1
DATA=$2
SAVING_PATH=$3
METRIC=$4
PRETRAIN_MODEL_PATH=$5
TRAIN_BATCH_SIZE=$6
EVAL_BATCH_SIZE=$7
TRAIN_EPOCH=$8
GRAD_ACCU_STEPS=$9
LOGGING_STEPS=${10}
LR=${11}
SAVING_STEPS=${12}
EVAL_STEPS=${13}
EVAL_STRATEGY=${14}
SAVE_STRATEGY=${15}
LOAD_FROM_FORMAT_TASK_ID=${16}

if [ ${MODE} == "fulldata" ]
then
  SAVING_PATH=${SAVING_PATH}/${DATA}/${MODE}
  fulldata ${DATA} ${SAVING_PATH} ${METRIC} ${PRETRAIN_MODEL_PATH} ${TRAIN_BATCH_SIZE} ${EVAL_BATCH_SIZE} ${TRAIN_EPOCH} ${GRAD_ACCU_STEPS} ${LOGGING_STEPS} ${LR} ${SAVING_STEPS} ${EVAL_STEPS} ${EVAL_STRATEGY} ${SAVE_STRATEGY} ${SAVE_LIMIT}
elif [ ${MODE} == "fewshot" ]
then
  SAVING_PATH=${SAVING_PATH}/${DATA}/${MODE}
  fewshot ${DATA} ${DATA} ${SAVING_PATH} ${METRIC} ${PRETRAIN_MODEL_PATH} ${TRAIN_BATCH_SIZE} ${EVAL_BATCH_SIZE} ${TRAIN_EPOCH} ${GRAD_ACCU_STEPS} ${LOGGING_STEPS} ${LR} ${SAVING_STEPS} ${EVAL_STEPS} ${EVAL_STRATEGY} ${SAVE_STRATEGY} ${SAVE_LIMIT}
elif [ ${MODE} == "zeroshot" ]
then
  SAVING_PATH=${SAVING_PATH}/${DATA}/${MODE}
  zeroshot ${DATA} ${SAVING_PATH} ${METRIC} ${PRETRAIN_MODEL_PATH} ${TRAIN_BATCH_SIZE} ${EVAL_BATCH_SIZE} ${TRAIN_EPOCH} ${GRAD_ACCU_STEPS} ${LOGGING_STEPS} ${LR} ${SAVING_STEPS} ${EVAL_STEPS} ${EVAL_STRATEGY} ${SAVE_STRATEGY} ${SAVE_LIMIT}
else
  echo "Unknown Mode"
fi



