export data_process_type=squad
export data=squad
export MODEL_OUTPUT=squad-t5-base
python -m torch.distributed.launch --nproc_per_node=8 run_t5_filter.py \
--model_name_or_path /home/t-wzhong/promptqa/promptQA/model/t5-base-v11-structprompt/${data}/fulldata \
--output_dir /home/t-wzhong/v-wanzho/promptQA/model/${MODEL_OUTPUT} \
--do_predict \
--dataset_name ${data_process_type} \
--per_device_train_batch_size 30 \
--per_device_eval_batch_size 30 \
--max_source_length 512 \
--max_target_length 32 \
--test_file /home/t-wzhong/v-wanzho/promptQA/wikipedia_data/pesudo_qa_data/extractive/paq_wiki400w_data_extractive_qapairs_pesudo_training_data.jsonl \
--output_path /home/t-wzhong/v-wanzho/promptQA/wikipedia_data/pesudo_qa_data/extractive/scored_paq_wiki400w_data_extractive_qapairs_pesudo_training_data.jsonl

#export data_process_type=squad
#export data=narrativeqa
#export MODEL_OUTPUT=narrativeqa-t5-base
#--model_name_or_path
#python -m torch.distributed.launch --nproc_per_node=8 run_t5_filter.py \
#python run_t5_filter.py \
#python -m torch.distributed.launch --nproc_per_node=8 run_t5_filter.py \
#--model_name_or_path /home/t-wzhong/promptqa/promptQA/model/t5-base-v11-structprompt/${data}/fulldata \
#--output_dir /home/t-wzhong/v-wanzho/promptQA/model/${MODEL_OUTPUT} \
#--do_predict \
#--dataset_name ${data_process_type} \
#--per_device_train_batch_size 2 \
#--per_device_eval_batch_size 20 \
#--max_source_length 512 \
#--max_target_length 32 \
#--test_file /home/t-wzhong/v-wanzho/promptQA/wikipedia_data/pesudo_qa_data/abstractive/paq_wiki400w_data_abstractive_qapairs_pesudo_training_data.jsonl \
#--output_path /home/t-wzhong/v-wanzho/promptQA/wikipedia_data/pesudo_qa_data/abstractive/scored_paq_wiki400w_data_abstractive_qapairs_pesudo_training_data.jsonl
#export data=race
#export MODEL_OUTPUT=${data}-t5-base
#python -m torch.distributed.launch --nproc_per_node=8 run_t5_filter.py \
#--model_name_or_path /home/t-wzhong/promptqa/promptQA/model/google-t5-v1_1-base/race/fulldata \
#--output_dir /home/t-wzhong/v-wanzho/promptQA/model/${MODEL_OUTPUT} \
#--do_predict \
#--dataset_name ${data} \
#--per_device_train_batch_size 2 \
#--per_device_eval_batch_size 20 \
#--max_source_length 512 \
#--max_target_length 32 \
#--test_file /home/t-wzhong/v-wanzho/promptQA/wikipedia_data/pesudo_qa_data/multichoice/wiki100w_multirc_qapair_negopt_pesudo_training_data_filtered_qgmodelv0.jsonl \
#--output_path /home/t-wzhong/v-wanzho/promptQA/wikipedia_data/pesudo_qa_data/multichoice/scored_wiki100w_multirc_qapair_negopt_pesudo_training_data_filtered_qgmodelv0.jsonl