# ProQA
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
## Download Source
### Download Pre-trained Model

The pretrained QA model with training epochs as 5

- ProQA-Base (20 prompt)
- ProQA-Base (100 prompt)
- ProQA-Large (100 prompt)

### Download Synthesized Pre-training Corpus
- Download generated QA-pairs corpus:
  
- Download the preprocessed corpus from PAQ:

- Download source Wikipedia data
```angular2html

```
### Download Preprocessed Downstream Data
For the nqopen | mctest | social_iqa datasets, we utilize preprocessed datasets from UnifiedQA, we can download it via this link.


