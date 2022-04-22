export data_process_type=squad
export data=squad
export MODEL_OUTPUT=squad-t5-base
export PROJECT_DIR=ProQA
python -m torch.distributed.launch --nproc_per_node=8 run_t5_filter.py \
--model_name_or_path ${PROJECT_DIR}/model/t5-base-v11-structprompt/${data}/fulldata \
--output_dir ${PROJECT_DIR}/model/${MODEL_OUTPUT} \
--do_predict \
--dataset_name ${data_process_type} \
--per_device_train_batch_size 30 \
--per_device_eval_batch_size 30 \
--max_source_length 512 \
--max_target_length 32 \
--test_file ${PROJECT_DIR}/wikipedia_data/pesudo_qa_data/extractive/paq_wiki400w_data_extractive_qapairs_pesudo_training_data.jsonl \
--output_path ${PROJECT_DIR}/wikipedia_data/pesudo_qa_data/extractive/scored_paq_wiki400w_data_extractive_qapairs_pesudo_training_data.jsonl
