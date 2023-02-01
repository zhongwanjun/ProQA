# ProQA
# Citation
Source code for the NAACL 2022 paper [ProQA: Structural Prompt-based Pre-training for Unified Question Answering](https://aclanthology.org/2022.naacl-main.313/). If you find the code useful, please cite our paper:
```
@inproceedings{zhong-etal-2022-proqa,
    title = "{P}ro{QA}: Structural Prompt-based Pre-training for Unified Question Answering",
    author = "Zhong, Wanjun  and
      Gao, Yifan  and
      Ding, Ning  and
      Qin, Yujia  and
      Liu, Zhiyuan  and
      Zhou, Ming  and
      Wang, Jiahai  and
      Yin, Jian  and
      Duan, Nan",
    booktitle = "Proceedings of the 2022 Conference of the North American Chapter of the Association for Computational Linguistics: Human Language Technologies",
    month = jul,
    year = "2022",
    address = "Seattle, United States",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2022.naacl-main.313",
    doi = "10.18653/v1/2022.naacl-main.313",
    pages = "4230--4243",
    abstract = "Question Answering (QA) is a longstanding challenge in natural language processing. Existing QA works mostly focus on specific question types, knowledge domains, or reasoning skills. The specialty in QA research hinders systems from modeling commonalities between tasks and generalization for wider applications. To address this issue, we present ProQA, a unified QA paradigm that solves various tasks through a single model. ProQA takes a unified structural prompt as the bridge and improves the QA-centric ability by structural prompt-based pre-training. Through a structurally designed prompt-based input schema, ProQA concurrently models the knowledge generalization for all QA tasks while keeping the knowledge customization for every specific QA task. Furthermore, ProQA is pre-trained with structural prompt-formatted large-scale synthesized corpus, which empowers the model with the commonly-required QA ability. Experimental results on 11 QA benchmarks demonstrate that ProQA consistently boosts performance on both full data fine-tuning, few-shot learning, and zero-shot testing scenarios. Furthermore, ProQA exhibits strong ability in both continual learning and transfer learning by taking the advantages of the structural prompt.",
}
```
## Prerequest
```
torch == 1.6.0
transformers == 4.12.5
nltk
```
## Pre-training
```angular2html
cd code/pretrain
# the example pretrain script is shown in
bash proqa_pretrain.sh
```
(Note: to modify the number of soft prompts, you need to modify the data_args.prompt_num in the main script and the prompt_num in models/modeling_t5.py)
## Training and evaluating on downstream tasks
The nqopen|newsqa|mctest|social_iqa datasets use local preprocessed data, and the other datasets(squad,quoref,narrativeqa,drop,race,dream,...) use the huggingface datasets API for downloading.
If you want to support more datasets or other tasks, you just need to add a function in the dataset_processors.py, and add corresponding evluation scripts (we support EM/F1/Accuracy/Rouge by now.)
```angular2html
cd code/downstream
```
### Full data fine-tuning
```
example script for nqopen|newsqa|mctest|social_iqa datasets (these datasets utilize local data)
./run.sh fulldata {drop|squad|...} path/for/save/models {f1|em|rouge_l|accuracy} path/to/pretrained/model train_batch_size eval_batch_size epoch_num gradient_accumulation_step logging_steps learning_rate saving_steps eval_steps eval_strategy save_strategy load_from_format_task_id [path/to/local/data]

example script for other dataset
./run.sh fulldata {drop|squad|...} path/for/save/models {f1|em|rouge_l|accuracy} path/to/pretrained/model train_batch_size eval_batch_size epoch_num gradient_accumulation_step logging_steps learning_rate saving_steps eval_steps eval_strategy save_strategy load_from_format_task_id
```
### Few-shot Learning
```
example script for nqopen|newsqa|mctest|social_iqa datasets (these datasets utilize local data)
./run.sh fewshot {drop|squad|...} path/for/save/models {f1|em|rouge_l|accuracy} path/to/pretrained/model train_batch_size eval_batch_size epoch_num gradient_accumulation_step logging_steps learning_rate saving_steps eval_steps eval_strategy save_strategy load_from_format_task_id [path/to/local/data]

example script for other dataset (squad,quoref,narrativeqa,drop,race,dream,...)
./run.sh fewshot {drop|squad|...} path/for/save/models {f1|em|rouge_l|accuracy} path/to/pretrained/model train_batch_size eval_batch_size epoch_num gradient_accumulation_step logging_steps learning_rate saving_steps eval_steps eval_strategy save_strategy load_from_format_task_id
```
### Zero-shot Learning
```
#some arguments are used for placeholder only
example script for nqopen|newsqa|mctest|social_iqa datasets (these datasets utilize local data)
./run.sh zeroshot {drop|squad|...} path/for/save/models {f1|em|rouge_l|accuracy} path/to/pretrained/model train_batch_size eval_batch_size epoch_num gradient_accumulation_step logging_steps learning_rate saving_steps eval_steps eval_strategy save_strategy load_from_format_task_id [path/to/local/data]

example script for other dataset
./run.sh zeroshot {drop|squad|...} path/for/save/models {f1|em|rouge_l|accuracy} path/to/pretrained/model train_batch_size eval_batch_size epoch_num gradient_accumulation_step logging_steps learning_rate saving_steps eval_steps eval_strategy save_strategy load_from_format_task_id
```
## Corpus Construction

### Question-Answer Pair Generation
```angular2html
cd corpus_construction/question_generation

1. Generate data for training the qa-pair generation model
python preprocess_for_train_qg.py

2. training the QA-pair generation model. The example script is:
bash train.sh
```
### Building Corpus from Wikipedia
```angular2html
cd corpus_construction/wikipedia_pretraining_construction

1. Generate QA-pair inference data from Wikipedia corpus
bash generate_qg_inference_data_from_wiki.sh 

2. Using QA-pair generation model to generate pesudo QA-pairs
bash inference_wikipedia_data.sh

3. Generate pesudo QA-pair for pretraining from the inferred results
python generate_pesudo_data.py [extractive_qapairs/abstractive_qapairs/multirc_qapairs_negopt/bool]
```
### Filtering
```
cd corpus_construction/filtering

1. The example script for calculating the score for the generated QA-pairs using the trained QA model (lower score is better)
bash t5_filter.sh

```
### Pre-trained Model
1. [ProQA (20 prompts+paq)](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155102332_link_cuhk_edu_hk/ElA6k4awhEBNidd4JO_gl8gBG1B27ZQ5U9wYMqCZ5HCWkg?e=FUx8tZ)
2. [ProQA (20 prompts+qapair)](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155102332_link_cuhk_edu_hk/EjHZZk8URW1JugNmdp2WbKsBou13ePwxF0h6KmPgU92t7Q?e=bADXrv)
3. [ProQA (100 prompts)](https://mycuhk-my.sharepoint.com/:f:/g/personal/1155102332_link_cuhk_edu_hk/Elf3RD-l6IFNvlvFRRzulN0B_YgOtCxnnZVuDWgPEYZcIg?e=UjaxSr)
