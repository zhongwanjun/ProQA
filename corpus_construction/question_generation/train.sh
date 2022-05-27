#for DATA_NAME in race_qfirst race_negopt_withq
#prepare training data
python preprocess_for_train_qg.py
#training
export BASIC_DIR=../../../
for DATA_NAME in race_negopt_withq
do
python -m torch.distributed.launch --nproc_per_node=8 run_qg.py \
    --model_name_or_path google/t5-v1_1-base \
    --output_dir ${BASIC_DIR}/qg_model/t5-base-qg-${DATA_NAME} \
    --data_dir ${BASIC_DIR}/qg_data/${DATA_NAME}/ \
    --train_file_path train.jsonl \
    --valid_file_path validation.jsonl \
    --per_device_train_batch_size 4 \
    --per_device_eval_batch_size 16 \
    --gradient_accumulation_steps 2 \
    --learning_rate 1e-4 \
    --num_train_epochs 10 \
    --do_train \
    --remove_unused_columns False \
    --do_eval \
    --max_decoding_length 32 \
    --num_beams 4 \
    --evaluation_strategy epoch \
    --save_strategy epoch \
    --logging_steps 100 \
    --load_best_model_at_end True \
    --predict_with_generate \
    --use_fast_tokenizer True \
    --overwrite_output_dir
done
