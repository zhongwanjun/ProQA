
export data=multi
export BASIC_DIR=../../
export TRAIN_BATCH_SIZE=8
export ACCUMULATION_STEPS=10
export EPOCH_NUM=6
export LR=1e-4
export MODEL_OUTPUT=pretrain_qapairs_extractive200w_bsz640_qapairs_abstractive_multirc-v11-5epoch-sametask-format-softprompt
python -m torch.distributed.launch --nproc_per_node=8 run_pretrain_proqa.py \
--model_name_or_path google/t5-v1_1-base \
--output_dir ${BASIC_DIR}/model/${MODEL_OUTPUT} \
--data_dir ${BASIC_DIR}/pretrain_data \
--do_train \
--do_eval \
--qa_task_type ${data} \
--per_device_train_batch_size ${TRAIN_BATCH_SIZE} \
--per_device_eval_batch_size 10 \
--overwrite_output_dir \
--gradient_accumulation_steps ${ACCUMULATION_STEPS} \
--num_train_epochs ${EPOCH_NUM} \
--warmup_ratio 0.1 \
--logging_steps 500 \
--learning_rate ${LR} \
--save_steps 10000 \
--eval_steps 10000 \
--evaluation_strategy steps \
--save_strategy steps \
--num_beams 4 \
--weight_decay 1e-2 \
--max_source_length 512 \
--label_smoothing_factor 0.1 \
--load_best_model_at_end True \
--greater_is_better True \
--predict_with_generate \
--metric_for_best_model exact_match \
--save_total_limit 15
