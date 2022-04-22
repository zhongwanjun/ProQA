# coding=utf-8
# Copyright The HuggingFace Team and The HuggingFace Inc. team. All rights reserved.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""
Fine-tuning the library models for sequence to sequence.
"""
# You can also adapt this script on your own sequence to sequence task. Pointers for this are left as comments.
# from data import QAData
import logging
import os
import sys
from functools import partial
from dataclasses import dataclass, field
from typing import Optional
from typing import List, Optional, Tuple
# from data import QAData
from trainer import QuestionAnsweringTrainer
import datasets
import numpy as np
from datasets import load_dataset, load_metric
import os
os.environ["WANDB_DISABLED"] = "true"
import transformers
from transformers import (
    AutoConfig,
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    DataCollatorForSeq2Seq,
    HfArgumentParser,
    M2M100Tokenizer,
    MBart50Tokenizer,
    EvalPrediction,
    MBart50TokenizerFast,
    MBartTokenizer,
    MBartTokenizerFast,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    default_data_collator,
    set_seed,
)
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version
sys.path.append('../')
from models.modeling_t5 import T5ForConditionalGeneration as PromptT5
from dataset_processors import *

# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.12.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")

logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]
format2id = {'extractive':0,'abstractive':1,'bool':2,'multichoice':3}
format2dataset = {
    'extractive':['squad','extractive'],
    'abstractive':['narrativeqa','abstractive'],
    'multichoice':['race','multichoice'],
    'bool':['boolq','bool']
}
seed_datasets = ['race','narrativeqa','squad']
dataset2format= {}
task2id = {}
for k,vs in format2dataset.items():
    for v in vs:
        dataset2format[v] = k
        # task2id[v] = len(task2id.keys())
# task2id = {v:k for k,v in enumerate(task_list)}
task2id = {'squad': 0, 'extractive': 1, 'narrativeqa': 2, 'abstractive':3 , 'race': 4, 'multichoice': 5, 'boolq': 6, 'bool': 7}
print(task2id)

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    config_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained config name or path if not the same as model_name"}
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Where to store the pretrained models downloaded from huggingface.co"},
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    model_revision: str = field(
        default="main",
        metadata={"help": "The specific model version to use (can be a branch name, tag name or commit id)."},
    )
    use_auth_token: bool = field(
        default=False,
        metadata={
            "help": "Will use the token generated when running `transformers-cli login` (necessary to use this script "
            "with private models)."
        },
    )


@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """

    source_lang: str = field(default=None, metadata={"help": "Source language id for translation."})
    target_lang: str = field(default=None, metadata={"help": "Target language id for translation."})

    dataset_name: Optional[str] = field(
        default=None, metadata={"help": "The name of the dataset to use (via the datasets library)."}
    )
    dataset_config_name: Optional[str] = field(
        default=None, metadata={"help": "The configuration name of the dataset to use (via the datasets library)."}
    )
    train_file: Optional[str] = field(default=None, metadata={"help": "The input training data file (a jsonlines)."})
    validation_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input evaluation data file to evaluate the metrics (sacreblue) on "
            "a jsonlines file."

        },
    )
    test_file: Optional[str] = field(
        default=None,
        metadata={
            "help": "An optional input test data file to evaluate the metrics (sacreblue) on " "a jsonlines file."
        },
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=None,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    max_source_length: Optional[int] = field(
        default=1024,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    max_target_length: Optional[int] = field(
        default=128,
        metadata={
            "help": "The maximum total sequence length for target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded."
        },
    )
    val_max_target_length: Optional[int] = field(
        default=None,
        metadata={
            "help": "The maximum total sequence length for validation target text after tokenization. Sequences longer "
            "than this will be truncated, sequences shorter will be padded. Will default to `max_target_length`."
            "This argument is also used to override the ``max_length`` param of ``model.generate``, which is used "
            "during ``evaluate`` and ``predict``."
        },
    )
    max_seq_length: int = field(
        default=384,
        metadata={
            "help": "The maximum total input sequence length after tokenization. Sequences longer "
                    "than this will be truncated, sequences shorter will be padded."
        },
    )
    pad_to_max_length: bool = field(
        default=False,
        metadata={
            "help": "Whether to pad all samples to model maximum sentence length. "
            "If False, will pad the samples dynamically when batching to the maximum length in the batch. More "
            "efficient on GPU but very bad for TPU."
        },
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of training examples to this "
            "value if set."
        },
    )
    max_eval_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of evaluation examples to this "
            "value if set."
        },
    )
    max_predict_samples: Optional[int] = field(
        default=None,
        metadata={
            "help": "For debugging purposes or quicker training, truncate the number of prediction examples to this "
            "value if set."
        },
    )
    num_beams: Optional[int] = field(
        default=None,
        metadata={
            "help": "Number of beams to use for evaluation. This argument will be passed to ``model.generate``, "
            "which is used during ``evaluate`` and ``predict``."
        },
    )
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    source_prefix: Optional[str] = field(
        default=None, metadata={"help": "A prefix to add before every source text (useful for T5 models)."}
    )
    forced_bos_token: Optional[str] = field(
        default=None,
        metadata={
            "help": "The token to force as the first generated token after the :obj:`decoder_start_token_id`."
            "Useful for multilingual models like :doc:`mBART <../model_doc/mbart>` where the first generated token "
            "needs to be the target language token.(Usually it is the target language token)"
        },
    )
    do_lowercase: bool = field(
        default=False,
        metadata={
            'help':'Whether to process input into lowercase'
        }
    )
    append_another_bos: bool = field(
        default=False,
        metadata={
            'help':'Whether to add another bos'
        }
    )
    context_column: Optional[str] = field(
        default="context",
        metadata={"help": "The name of the column in the datasets containing the contexts (for question answering)."},
    )
    question_column: Optional[str] = field(
        default="question",
        metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
    )
    answer_column: Optional[str] = field(
        default="answers",
        metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
    )
    qa_task_type: Optional[str] = field(
        default="bool",
        metadata={"help": "the type of the qa task"},
    )
    max_task_num: Optional[int] = field(
        default=30,
        metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
    )
    qa_task_type_num: Optional[int] = field(
        default=4,
        metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
    )
    prompt_number: Optional[int] = field(
        default=100,
        metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
    )
    add_task_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
    )
    add_seed_dataset: Optional[bool] = field(
        default=True,
        metadata={"help": "whether to add seed datasets for training"},
    )
    data_dir: Optional[str] = field(
        default="../../",
        metadata={"help": "the data path for storing the pre-training file"},
    )



import copy
def main():
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    training_args.load_best_model_at_end = True
    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout)],
    )

    log_level = training_args.get_process_log_level()
    logger.setLevel(log_level)
    datasets.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.set_verbosity(log_level)
    transformers.utils.logging.enable_default_handler()
    transformers.utils.logging.enable_explicit_format()
    question_answering_column_name_mapping = {
        "squad_v2": ("question", "context", "answer"),
    }
    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        + f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    logger.info(f"Training/evaluation parameters {training_args}")

    if data_args.source_prefix is None and model_args.model_name_or_path in [
        "t5-small",
        "t5-base",
        "t5-large",
        "t5-3b",
        "t5-11b",
    ]:
        logger.warning(
            "You're running a t5 model but didn't provide a source prefix, which is expected, e.g. with "
            "`--source_prefix 'translate English to German: ' `"
        )

    # Detecting last checkpoint.
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    # Set seed before initializing model.
    set_seed(training_args.seed)

    qa_task_to_file_names = {

        'abstractive':{
            'train':os.path.join(data_args.data_dir,'abstractive/scored_paq_wiki400w_data_abstractive_qapairs_pesudo_training_data_top200w.jsonl'),
            'validation':os.path.join(data_args.data_dir,'abstractive/scored_paq_wiki400w_data_abstractive_qapairs_pesudo_val_data.jsonl')
        },
        'extractive':{

            'train':os.path.join(data_args.data_dir,'PAQ_data/preprocessed_paq_pesudo_qa_data_train.jsonl'),
            'validation':os.path.join(data_args.data_dir,'PAQ_data/preprocessed_paq_pesudo_qa_data_val.jsonl')
        },
        'multichoice':{
            'train':os.path.join(data_args.data_dir,'multichoice/wiki100w_multirc_qapair_negopt_pesudo_training_data_filtered_qgmodelv0.jsonl'),
            'validation':os.path.join(data_args.data_dir,'multichoice/wiki100w_multirc_qapair_negopt_pesudo_val_data_filtered_qgmodelv0.jsonl')
        }
    }
    qa_tasks = ['extractive','abstractive','multichoice']
    if data_args.qa_task_type == 'multi':
        all_data_files = [{
            'train':qa_task_to_file_names[task]['train'],
            'validation':qa_task_to_file_names[task]['validation'],
        } for task in qa_tasks]
        task2datasets = {}
        for i,data_files in enumerate(all_data_files):
            raw_datasets = load_dataset('json', data_files=data_files, cache_dir=model_args.cache_dir)
            def add_task_name(example):
                example.update({'task_type':qa_tasks[i]})
                return example
            raw_datasets = {k:v.map(add_task_name) for k,v in raw_datasets.items()}
            task2datasets[qa_tasks[i]] = raw_datasets
    else:
        if data_args.dataset_name is not None:
            # Downloading and loading a dataset from the hub.
            raw_datasets = load_dataset(
                data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
            )
        else:
            data_files = {}
            if data_args.train_file is not None:
                data_files["train"] = data_args.train_file
                extension = data_args.train_file.split(".")[-1]
            if data_args.validation_file is not None:
                data_files["validation"] = data_args.validation_file
                extension = data_args.validation_file.split(".")[-1]
            if data_args.test_file is not None:
                data_files["test"] = data_args.test_file
                extension = data_args.test_file.split(".")[-1]

            raw_datasets = load_dataset('json',data_files=data_files,cache_dir=model_args.cache_dir)
            def add_task_name(example):
                example.update({'task_type':data_args.qa_task_type})
                return example
            raw_datasets = {k:v.map(add_task_name) for k,v in raw_datasets.items()}
        task2datasets = {}
        task2datasets[data_args.qa_task_type] = raw_datasets

    dname_to_funcs = {
        'race':transfer_race_to_pretrain_data,
        'narrativeqa':transfer_narrativeqa_to_pretrain_data,
        'squad':transfer_squad_to_pretrain_data
    }

    if data_args.add_seed_dataset:
        for dataset in seed_datasets:
            logger.info(f'Loading {dataset} dataset')
            config_name = 'all' if dataset=='race' else None
            seed_dataset = load_dataset(dataset,config_name,cache_dir=model_args.cache_dir)
            transfer_func = dname_to_funcs[dataset]
            map_func = partial(transfer_func,task_type=dataset)
            column_names = seed_dataset['validation'].features.keys()#[key for key in seed_dataset['validation'].features.keys() if key not in ['question','answers','context','options']]
            seed_dataset = {k: v.map(map_func,remove_columns=column_names,load_from_cache_file=False,
                                     ) for k, v in seed_dataset.items()}
            task2datasets[dataset] = seed_dataset

    config = AutoConfig.from_pretrained(
        model_args.config_name if model_args.config_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )

    model = PromptT5.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
    )
    special_tokens_dict = {'additional_special_tokens': ['[TASK]', '[QUESTION]', '[CONTEXT]',
                                                         '[OPTIONS]']}
    # tokenizer.add_tokens(['[TASK]', '[ABSTRACTIVE]','[QUESTION]','[CONTEXT]','[BOOL]','[EXTRACTIVE]','[MultiChoice]',
    #                       '[OPTIONS]'])
    tokenizer.add_tokens(['[ABSTRACTIVE]', '[BOOL]', '[EXTRACTIVE]', '[MultiChoice]'])
    num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
    added_tokens = tokenizer.get_added_vocab()
    logger.info('Added tokens: {}'.format(added_tokens))
    model.resize_token_embeddings(len(tokenizer))

    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""

    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["test"].column_names
    else:
        logger.info("There is nothing to do. Please pass `do_train`, `do_eval` and/or `do_predict`.")
        return

    # Temporarily set max_target_length for training.
    max_target_length = data_args.max_target_length
    padding = "max_length" if data_args.pad_to_max_length else False

    if training_args.label_smoothing_factor > 0 and not hasattr(model, "prepare_decoder_input_ids_from_labels"):
        logger.warning(
            "label_smoothing is enabled but the `prepare_decoder_input_ids_from_labels` method is not defined for"
            f"`{model.__class__.__name__}`. This will lead to loss being calculated twice and will take up more memory"
        )

    dataset_columns = question_answering_column_name_mapping.get(data_args.dataset_name, None)
    if data_args.question_column is None:
        question_column = dataset_columns[0] if dataset_columns is not None else column_names[0]
    else:
        question_column = data_args.question_column
        if question_column not in column_names:
            raise ValueError(
                f"--question_column' value '{data_args.question_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.context_column is None:
        context_column = dataset_columns[1] if dataset_columns is not None else column_names[1]
    else:
        context_column = data_args.context_column
        if context_column not in column_names:
            raise ValueError(
                f"--context_column' value '{data_args.context_column}' needs to be one of: {', '.join(column_names)}"
            )
    if data_args.answer_column is None:
        answer_column = dataset_columns[2] if dataset_columns is not None else column_names[2]
    else:
        answer_column = data_args.answer_column
        if answer_column not in column_names:
            raise ValueError(
                f"--answer_column' value '{data_args.answer_column}' needs to be one of: {', '.join(column_names)}"
            )


    import random
    if data_args.max_seq_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_seq_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    max_seq_length = min(data_args.max_seq_length, tokenizer.model_max_length)

    def dataset_name_to_func(dataset_name):
        mapping = {
            'extractive': preprocess_sqaud_batch,
            'bool': preprocess_boolq_batch_pretrain,
            'abstractive': preprocess_narrativeqa_batch_pretrain,
            'multichoice': preprocess_multirc_batch_pretrain,
            'race': preprocess_multirc_batch_new,
            'narrativeqa': preprocess_narrativeqa_batch_pretrain_new,
            'squad':preprocess_sqaud_batch_new
        }
        return mapping[dataset_name]

    def preprocess_function(examples):
        preprocess_fn = dataset_name_to_func(examples['task_type'][0])
        inputs, targets = preprocess_fn(examples, question_column, context_column, answer_column)
        #preprocess_sqaud_batch(examples, question_column, context_column, answer_column)

        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        with tokenizer.as_target_tokenizer():
            labels = tokenizer(targets, max_length=data_args.max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        format_id = format2id[dataset2format[examples['task_type'][0]]]
        format_prompt_ids = [- (i + 1) for i in range(format_id*data_args.prompt_number,(format_id+1)*data_args.prompt_number)]#list(range(-(format_id * data_args.prompt_number+1), -((format_id + 1) * data_args.prompt_number+1)))
        task_prompt_id_start = len(format2id.keys()) * data_args.prompt_number
        task_id = task2id[examples['task_type'][0]]
        task_prompt_ids = [- (i + 1) for i in range(task_prompt_id_start+task_id*data_args.prompt_number,task_prompt_id_start+(task_id+1)*data_args.prompt_number)]

        input_ids = copy.deepcopy([format_prompt_ids+task_prompt_ids+input_ids for input_ids in model_inputs['input_ids']])
        model_inputs['input_ids'] = input_ids#[format_prompt_ids+input_ids for input_ids in model_inputs['input_ids']]
        model_inputs['attention_mask'] = [[1]*data_args.prompt_number*2+attention_mask for attention_mask in model_inputs['attention_mask']]

        return model_inputs
    def preprocess_validation_function(examples):
        preprocess_fn = dataset_name_to_func(examples['task_type'][0])
        inputs, targets = preprocess_fn(examples, question_column, context_column, answer_column)
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        # model_inputs["example_id"] = []
        for i in range(len(model_inputs["input_ids"])):
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = i#sample_mapping[i]
            # model_inputs["example_id"].append(examples["id"][sample_index])
        with tokenizer.as_target_tokenizer():

            labels = tokenizer(targets, max_length=data_args.max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]

        model_inputs["labels"] = labels["input_ids"]
        # model_inputs['format'] = [
        #     list(range(format_id * data_args.prompt_number, (format_id + 1) * data_args.prompt_number)) for input in
        #     inputs]
        format_id = format2id[dataset2format[examples['task_type'][0]]]
        format_prompt_ids = [- (i + 1) for i in range(format_id * data_args.prompt_number, (
                    format_id + 1) * data_args.prompt_number)]  # list(range(-(format_id * data_args.prompt_number+1), -((format_id + 1) * data_args.prompt_number+1)))
        task_prompt_id_start = len(format2id.keys()) * data_args.prompt_number
        task_id = task2id[examples['task_type'][0]]
        task_prompt_ids = [- (i + 1) for i in range(task_prompt_id_start + task_id * data_args.prompt_number,
                                                    task_prompt_id_start + (task_id + 1) * data_args.prompt_number)]

        input_ids = copy.deepcopy(
            [format_prompt_ids + task_prompt_ids + input_ids for input_ids in model_inputs['input_ids']])
        # input_ids = copy.deepcopy([format_prompt_ids + input_ids for input_ids in model_inputs['input_ids']])
        model_inputs['input_ids'] = input_ids
        model_inputs['attention_mask'] = [[1] * data_args.prompt_number*2 + attention_mask for attention_mask in
                                          model_inputs['attention_mask']]
        # print(model_inputs['input_ids'][0])
        return model_inputs
    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        # if data_args.qa_task_type=='multi':
        train_datasets = []
        for task_type, ds in task2datasets.items():
            ds = ds['train']
            if data_args.max_train_samples is not None:
                ds = ds.select(range(min(len(ds), data_args.max_train_samples)))
            column_names = ds.column_names
            with training_args.main_process_first(desc="train dataset map pre-processing"):
                ds = ds.map(
                    preprocess_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on train dataset",
                )
                train_datasets.append(ds)
        train_dataset = datasets.concatenate_datasets(train_datasets)


 
    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        # if data_args.qa_task_type == 'multi':
        eval_datasets = []
        eval_examples = []

        def add_id(example, index):
            example.update({'id': f'{index}'})
            return example

        def add_dataset_id(example, index):
            example.update({'example_id': f'{index}'})
            return example

        for task_type, ds in task2datasets.items():
            ds = ds['validation']
            if data_args.max_eval_samples is not None:
                ds = ds.select(range(min(len(ds), data_args.max_eval_samples)))

            if (data_args.add_seed_dataset and task_type in seed_datasets):# or not data_args.add_seed_dataset:
                eval_examples.append(ds)
            column_names = ds.column_names
            with training_args.main_process_first(desc="validation dataset map pre-processing"):
                ds = ds.map(
                    preprocess_validation_function,
                    batched=True,
                    num_proc=data_args.preprocessing_num_workers,
                    remove_columns=column_names,
                    load_from_cache_file=not data_args.overwrite_cache,
                    desc="Running tokenizer on validation dataset",
                )

            if data_args.add_seed_dataset and task_type in seed_datasets:
                eval_datasets.append(ds)
            logger.info(ds.features.keys())

        logger.info([eval.features.keys() for eval in eval_examples])
        eval_examples = datasets.concatenate_datasets(eval_examples)
        if 'id' not in eval_examples.features.keys():
            eval_examples = eval_examples.map(add_id, with_indices=True)
        eval_dataset = datasets.concatenate_datasets(eval_datasets)
        eval_dataset = eval_dataset.map(add_dataset_id, with_indices=True)



    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on prediction dataset",
            )

    # Data collator
    label_pad_token_id = -100 if data_args.ignore_pad_token_for_loss else tokenizer.pad_token_id
    if data_args.pad_to_max_length:
        data_collator = default_data_collator
    else:
        data_collator = DataCollatorForSeq2Seq(
            tokenizer,
            model=model,
            label_pad_token_id=label_pad_token_id,
            pad_to_multiple_of=8 if training_args.fp16 else None,
        )

    # Metric
    # metric = load_metric("squad")
    metric = load_metric("squad")
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)

    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        answer_column_name=answer_column,
        dataset_name='squad_new',
        tokenizer=tokenizer,
        # report_to=None,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )

    # Training
    if training_args.do_train:
        checkpoint = None
        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)
        trainer.save_model()  # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        max_train_samples = (
            data_args.max_train_samples if data_args.max_train_samples is not None else len(train_dataset)
        )
        metrics["train_samples"] = min(max_train_samples, len(train_dataset))

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()

        def save_prompt_embedding(model):
            prompt_embedding = model.state_dict['encoder.prompt_embeddings.weight']
            save_prompt_info = {'encoder.prompt_embeddings.weight': copy.deepcopy(prompt_embedding), 'task2id': task2id,
                                'format2id': format2id}
            prompt_path = os.path.join(training_args.output_dir, 'prompt_embedding_info')
            torch.save(save_prompt_info, prompt_path)
            logger.info(f'Saving prompt embedding information to {prompt_path}')

        save_prompt_embedding(model)

    # Evaluation
    results = {}
    max_length = (
        training_args.generation_max_length
        if training_args.generation_max_length is not None
        else data_args.val_max_target_length
    )
    num_beams = data_args.num_beams if data_args.num_beams is not None else training_args.generation_num_beams
    if training_args.do_eval:
        logger.info("*** Evaluate ***")

        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval",tokenizer=tokenizer)
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        logger.info("*** Predict ***")

        predict_results = trainer.predict(
            predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        )
        metrics = predict_results.metrics
        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        if trainer.is_world_process_zero():
            if training_args.predict_with_generate:
                predictions = tokenizer.batch_decode(
                    predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
                )
                predictions = [pred.strip() for pred in predictions]
                output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
                with open(output_prediction_file, "w", encoding="utf-8") as writer:
                    writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "translation"}
    if data_args.dataset_name is not None:
        kwargs["dataset_tags"] = data_args.dataset_name
        if data_args.dataset_config_name is not None:
            kwargs["dataset_args"] = data_args.dataset_config_name
            kwargs["dataset"] = f"{data_args.dataset_name} {data_args.dataset_config_name}"
        else:
            kwargs["dataset"] = data_args.dataset_name

    languages = [l for l in [data_args.source_lang, data_args.target_lang] if l is not None]
    if len(languages) > 0:
        kwargs["language"] = languages

    if training_args.push_to_hub:
        trainer.push_to_hub(**kwargs)
    else:
        trainer.create_model_card(**kwargs)

    return results


def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()