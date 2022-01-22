#generate qapair inference data from wikipedia, for utilizing QA-pair generation model to inference the pesudo data
export BASIC_DIR='../../../'
#extractive qa-pairs
python paq_wiki_psg_to_qg.py ${BASIC_DIR}
#bool
python wikipsg_to_qg_data.py bool ${BASIC_DIR}
#abstractive qa-pairs
python wikipsg_to_qg_data.py abstractive_qapairs ${BASIC_DIR}
#MultiCoiceQA
#Step1: MultiChoiceQA qa-pairs
python wikipsg_to_qg_data.py multirc_all_qapairs ${BASIC_DIR}
#Step2: MultiChoice QA neg-options for generated qa pairs
export QAPAIR_RESULT_PATH=${BASIC_DIR}/qg_inference_data/multirc_qapairs/result_qg_inference_multirc_qapairs.jsonl
python wikipsg_to_qg_data.py multirc_qapair_negopt ${BASIC_DIR} ${QAPAIR_RESULT_PATH}
