#preparing data
export BASIC_DIR=/home/t-wzhong/v-wanzho/promptQA
cd ../question_generation
#extractive qa pairs
export DATA_NAME=extractive
python -m torch.distributed.launch --nproc_per_node=8 run_qg.py \
    --model_name_or_path ${BASIC_DIR}/qg_model/t5-large-qg-squad_qapairs \
    --do_predict \
    --model_type t5 \
    --data_dir ${BASIC_DIR}/qg_data/${DATA_NAME}/ \
    --train_file_path train.jsonl \
    --valid_file_path validation.jsonl \
    --output_dir ${BASIC_DIR}/wikipedia_data/qg_inference_data/${DATA_NAME}/output \
    --test_file_path ${BASIC_DIR}/wikipedia_data/qg_inference_data/${DATA_NAME}/qg_inference_${DATA_NAME}_qapairs.jsonl \
    --num_beams 4 \
    --num_return_seq 4 \
    --max_decoding_length 32 \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 20 \
    --remove_unused_columns False \
    --predict_with_generate \
    --use_fast_tokenizer True \
    --output_path ${BASIC_DIR}/wikipedia_data/qg_inference_data/${DATA_NAME}/result_${DATA_NAME}_qapairs_t5-large.jsonl
#MultiChoice QA pairs
export DATA_NAME=multirc
python -m torch.distributed.launch --nproc_per_node=8 run_qg.py \
    --model_name_or_path ${BASIC_DIR}/qg_model/t5-large-qg-race_qapairs \
    --do_predict \
    --model_type t5 \
    --data_dir ${BASIC_DIR}/qg_data/${DATA_NAME}/ \
    --train_file_path train.jsonl \
    --valid_file_path validation.jsonl \
    --output_dir ${BASIC_DIR}/wikipedia_data/qg_inference_data/${DATA_NAME}/output \
    --test_file_path ${BASIC_DIR}/wikipedia_data/qg_inference_data/${DATA_NAME}/qg_inference_multirc_qapairs.jsonl \
    --num_beams 4 \
    --num_return_seq 2 \
    --max_decoding_length 32 \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 20 \
    --remove_unused_columns False \
    --predict_with_generate \
    --use_fast_tokenizer True \
    --output_path ${BASIC_DIR}/wikipedia_data/qg_inference_data/${DATA_NAME}/result_${DATA_NAME}_qapairs_t5-large.jsonl
#MultiChoice QA: neg options generation with result from generated qa-pairs
export DATA_NAME=multirc
python -m torch.distributed.launch --nproc_per_node=8 run_qg.py \
    --model_name_or_path ${BASIC_DIR}/qg_model/t5-large-qg-race_negopt_withq-2 \
    --do_predict \
    --model_type t5 \
    --data_dir ${BASIC_DIR}/qg_data/${DATA_NAME}/ \
    --train_file_path train.jsonl \
    --valid_file_path validation.jsonl \
    --output_dir ${BASIC_DIR}/wikipedia_data/qg_inference_data/${DATA_NAME}/output \
    --test_file_path ${BASIC_DIR}/wikipedia_data/qg_inference_data/${DATA_NAME}/qg_inference_multirc_qapairs_negopt.jsonl \
    --num_beams 15 \
    --num_return_seq 15 \
    --max_decoding_length 24 \
    --per_device_train_batch_size 3 \
    --per_device_eval_batch_size 3 \
    --remove_unused_columns False \
    --predict_with_generate \
    --use_fast_tokenizer True \
    --output_path ${BASIC_DIR}/wikipedia_data/qg_inference_data/${DATA_NAME}/result_qg_inference_multirc_qapairs_negopt.jsonl


export DATA_NAME=abstractive
python -m torch.distributed.launch --nproc_per_node=8 run_qg.py \
    --model_name_or_path ${BASIC_DIR}/qg_model/t5-large-qg-narrativeqa_qapairs-2 \
    --do_predict \
    --model_type t5 \
    --data_dir ${BASIC_DIR}/qg_data/${DATA_NAME}/ \
    --train_file_path train.jsonl \
    --valid_file_path validation.jsonl \
    --output_dir ${BASIC_DIR}/wikipedia_data/qg_inference_data/${DATA_NAME}/output \
    --test_file_path ${BASIC_DIR}/wikipedia_data/qg_inference_data/abstractive/qg_inference_abstractive_qapairs.jsonl \
    --num_beams 4 \
    --num_return_seq 4 \
    --max_decoding_length 32 \
    --per_device_train_batch_size 20 \
    --per_device_eval_batch_size 20 \
    --remove_unused_columns False \
    --predict_with_generate \
    --use_fast_tokenizer True \
    --output_path ${BASIC_DIR}/wikipedia_data/qg_inference_data/${DATA_NAME}/result_${DATA_NAME}_qapairs_t5-large.jsonl