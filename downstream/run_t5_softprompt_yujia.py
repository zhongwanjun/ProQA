#!/usr/bin/env python
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
import torch
import copy,random
import sys
from dataclasses import dataclass, field
from typing import Optional
from typing import List, Optional, Tuple
# from data import QAData
from models.modeling_t5 import T5ForConditionalGeneration as PromptT5
# from models.promptT5 import PromptT5
from dataset_processors import *
from trainer import QuestionAnsweringTrainer
import datasets
import numpy as np
from datasets import load_dataset, load_metric
import os

from functools import partial
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
from pathlib import Path
from transformers.trainer_utils import get_last_checkpoint
from transformers.utils import check_min_version
from transformers.utils.versions import require_version


# Will error if the minimal version of Transformers is not installed. Remove at your own risks.
# check_min_version("4.12.0.dev0")

require_version("datasets>=1.8.0", "To fix: pip install -r examples/pytorch/translation/requirements.txt")
logger = logging.getLogger(__name__)

# A list of all multilingual tokenizer which require src_lang and tgt_lang attributes.
MULTILINGUAL_TOKENIZERS = [MBartTokenizer, MBartTokenizerFast, MBart50Tokenizer, MBart50TokenizerFast, M2M100Tokenizer]
format2id = {'extractive':0,'abstractive':1,'bool':2,'multichoice':3}
format2dataset = {
    'extractive':['squad','extractive','newsqa','quoref','ropes'],
    'abstractive':['narrativeqa','abstractive','nqopen','drop'],
    'multichoice':['race','multichoice','openbookqa','mctest','social_iqa','dream'],
    'bool':['boolq','bool','boolq_np']
}
seed_datasets = ['race','narrativeqa','squad']
dataset2format= {}
task2id = {}
for k,vs in format2dataset.items():
    for v in vs:
        dataset2format[v] = k
        # task2id[v] = len(task2id.keys())
# task2id = {v:k for k,v in enumerate(task_list)}
task2id = {'squad': 0, 'extractive': 1, 'narrativeqa': 2, 'abstractive': 3, 'race': 4, 'multichoice': 5, 'boolq': 6, 'bool': 7, 'newsqa':8,'quoref':9,'ropes':10,'drop':11,'nqopen':12,'boolq_np':13,'openbookqa':14,'mctest':15,'social_iqa':16,'dream':17}

# task2id = {'squad': 0, 'extractive': 0, 'narrativeqa': 1, 'abstractive':1 , 'race': 2, 'multichoice': 2, 'boolq': 3, 'bool': 3, 'newsqa':8,'quoref':9,'ropes':10,'drop':11,'nqopen':12,'boolq_np':13,'openbookqa':14,'mctest':15,'social_iqa':16,'dream':17}
# print(task2id)

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
    fix_t5: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to fix the main parameters of t5"}
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
    max_task_num: Optional[int] = field(
        default=30,
        metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
    )
    qa_task_type_num: Optional[int] = field(
        default=4,
        metadata={"help": "The name of the column in the datasets containing the questions (for question answering)."},
    )
    prompt_number: Optional[int] = field(
        default=20,
        metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
    )
    add_task_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "The name of the column in the datasets containing the answers (for question answering)."},
    )
    reload_from_trained_prompt: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to reload prompt from trained prompt"},
    )
    trained_prompt_path:Optional[str] = field(
        default = None,
        metadata={
            'help':'the path storing trained prompt embedding'
        }
    )
    load_from_format_task_id: Optional[bool] = field(
        default=False,
        metadata={"help": "whether to reload prompt from format-corresponding task prompt"}
    )
    task_b_name: Optional[str] = field(
        default=None,
        metadata={
            'help': 'task b name for continual learning'
        }
    )
    continual_task_a_name: Optional[str] = field(
        default=None,
        metadata={
            'help': 'task a name for continual learning'
        }
    )


    def __post_init__(self):
        if self.dataset_name is None and self.train_file is None and self.validation_file is None:
            raise ValueError("Need either a dataset name or a training/validation file.")
        # elif self.source_lang is None or self.target_lang is None:
        #     raise ValueError("Need to specify the source language and the target language.")

        if self.train_file is not None:
            extension = self.train_file.split(".")[-1]
            # assert extension == "json", "`train_file` should be a json file."
        if self.validation_file is not None:
            extension = self.validation_file.split(".")[-1]
            # assert extension == "json", "`validation_file` should be a json file."
        if self.val_max_target_length is None:
            self.val_max_target_length = self.max_target_length


# +
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
    if 'same' in model_args.model_name_or_path:
        task2id = {'squad': 0, 'extractive': 0, 'narrativeqa': 1, 'abstractive': 1, 'race': 2, 'multichoice': 2,
                   'boolq': 3, 'bool': 3, 'newsqa': 8, 'quoref': 9, 'ropes': 10, 'drop': 11, 'nqopen': 12,
                   'boolq_np': 13, 'openbookqa': 14, 'mctest': 15, 'social_iqa': 16, 'dream': 17}
    else:
        task2id = {'squad': 0, 'extractive': 1, 'narrativeqa': 2, 'abstractive': 3, 'race': 4, 'multichoice': 5,
                   'boolq': 6, 'bool': 7, 'newsqa': 8, 'quoref': 9, 'ropes': 10, 'drop': 11, 'nqopen': 12,
                   'boolq_np': 13, 'openbookqa': 14, 'mctest': 15, 'social_iqa': 16, 'dream': 17}

    print(task2id)
    if data_args.task_b_name:
        if '/' in data_args.task_b_name:
            data_args.task_b_name = data_args.task_b_name.split('/')[-1]
        print(f'Task b name: {data_args.task_b_name}')
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
    # question_answering_column_name_mapping = {
    #     "squad_v2": ("question", "context", "answer"),
    # }
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

    # Get the datasets: you can either provide your own JSON training and evaluation files (see below)
    # or just provide the name of one of the public datasets available on the hub at https://huggingface.co/datasets/
    # (the dataset will be downloaded automatically from the datasets Hub).
    #
    # For translation, only JSON files are supported, with one field named "translation" containing two keys for the
    # source and target languages (unless you adapt what follows).
    #
    # In distributed training, the load_dataset function guarantee that only one local process can concurrently
    # download the dataset.
    if data_args.dataset_name is not None and data_args.dataset_name not in ['newsqa', 'nqopen', 'multirc', 'boolq_np', 'mctest', 'social_iqa']:
        # Downloading and loading a dataset from the hub.
        raw_datasets = load_dataset(
            data_args.dataset_name, data_args.dataset_config_name, cache_dir=model_args.cache_dir
        )
        if data_args.dataset_name in ['ropes']:
            # add answer_start (not used for squad evaluation but required)
            def add_answer_start(example):
                example['answers'].update({"answer_start": [0]})
                return example
            raw_datasets = raw_datasets.map(add_answer_start)
        elif data_args.dataset_name in ['drop']:
            # add answer_start (not used for squad evaluation but required)
            # add answers (for squad evaluation)
            def add_answers(example):
                answers = []
                answer_start = []
                for _a in example['answers_spans']['spans']:
                    answers.append(_a)
                    answer_start.append(-1)
                example['answers'] = {"text": answers, "answer_start": answer_start}
                return example
            raw_datasets = raw_datasets.map(add_answers)

    else:
        data_files = {}
        if data_args.train_file is not None:
            data_files["train"] = data_args.train_file
            # extension = data_args.train_file.split(".")[-1]
        if data_args.validation_file is not None:
            data_files["validation"] = data_args.validation_file
            # extension = data_args.validation_file.split(".")[-1]
        if data_args.test_file is not None:
            data_files["test"] = data_args.test_file
            # extension = data_args.test_file.split(".")[-1]
        if data_args.dataset_name in ['newsqa', 'nqopen', 'multirc', 'boolq_np', 'mctest', 'social_iqa',]:
            raw_datasets = load_dataset('json', data_files=data_files, cache_dir=model_args.cache_dir)
        else:
            print(f"Unknown dataset {data_args.dataset_name}")
            raise NotImplementedError
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
    # tokenizer.add_tokens(['[TASK]', '[ABSTRACTIVE]','[QUESTION]','[CONTEXT]','[BOOL]','[EXTRACTIVE]','[MultiChoice]',
    #                       '[OPTIONS]'])
    tokens_to_add = ['[ABSTRACTIVE]', '[BOOL]', '[EXTRACTIVE]', '[MultiChoice]']
    special_tokens_dict = {'additional_special_tokens': ['[TASK]', '[QUESTION]', '[CONTEXT]',
                                                             '[OPTIONS]']}
        # tokenizer.add_tokens(['[TASK]', '[ABSTRACTIVE]','[QUESTION]','[CONTEXT]','[BOOL]','[EXTRACTIVE]','[MultiChoice]',
        #                       '[OPTIONS]'])
    if training_args.do_train or not all([token in tokenizer.get_added_vocab() for token in tokens_to_add+special_tokens_dict['additional_special_tokens']]):
        tokenizer.add_tokens(tokens_to_add)
        num_added_toks = tokenizer.add_special_tokens(special_tokens_dict)
        added_tokens = tokenizer.get_added_vocab()
        logger.info('Added tokens: {}'.format(added_tokens))
    else:
        logger.info('Added tokens: {}'.format(tokenizer.get_added_vocab()))
    model = PromptT5.from_pretrained(
        model_args.model_name_or_path,
        from_tf=bool(".ckpt" in model_args.model_name_or_path),
        config=config,
        cache_dir=model_args.cache_dir,
        revision=model_args.model_revision,
        use_auth_token=True if model_args.use_auth_token else None,
        # task_num = data_args.max_task_num,
        # prompt_num = data_args.prompt_number,
        # format_num = data_args.qa_task_type_num,
        # add_task_prompt = False
    )

    model.resize_token_embeddings(len(tokenizer))
    #reload format specific task-prompt for newly involved task
    if data_args.continual_task_a_name is not None:
        task_start_id = data_args.prompt_number * len(format2dataset.keys())
        a_task_id = task_start_id + task2id[data_args.continual_task_a_name] * data_args.prompt_number
        a_format_id = format2id[dataset2format[data_args.continual_task_a_name]]
        b_task_id = task_start_id + task2id[data_args.dataset_name] * data_args.prompt_number
        b_format_id = format2id[dataset2format[data_args.dataset_name]]
        # a_format_task_id = task_start_id + task2id[dataset2format[data_args.continual_task_a_name]] * data_args.prompt_number
        model.state_dict()['encoder.prompt_embeddings.weight'][b_task_id:b_task_id + data_args.prompt_number, :] = \
        model.state_dict()['encoder.prompt_embeddings.weight'][a_task_id:a_task_id + data_args.prompt_number,:]

        model.state_dict()['encoder.prompt_embeddings.weight'][b_format_id * data_args.prompt_number:(b_format_id + 1) * data_args.prompt_number, :] = \
            model.state_dict()['encoder.prompt_embeddings.weight'][a_format_id * data_args.prompt_number:(a_format_id + 1) * data_args.prompt_number,:]
        # model.state_dict()['encoder.prompt_embeddings.weight'][b_task_id:b_task_id + data_args.prompt_number, :] = \  
        #     model.state_dict()['encoder.prompt_embeddings.weight'][a_task_id:a_task_id + data_args.prompt_number, :]
        logger.info(
            f'Successfully initialize format and task prompt {data_args.dataset_name} from task {data_args.continual_task_a_name}, task a id {a_task_id}, task b id {b_task_id}')
    elif data_args.load_from_format_task_id and (data_args.dataset_name not in seed_datasets) and not data_args.reload_from_trained_prompt and (data_args.dataset_name!=data_args.task_b_name) and data_args.task_b_name:
        data_args.trained_prompt_path = f'/home/t-wzhong/v-wanzho/promptQA/model/few-shot/pretrain_paq_extractive_qapairs_abstractive_multirc-v11-5epoch-task-format-softprompt/{data_args.task_b_name}/fewshot/prompt_embedding_info'
        prompt_info = torch.load(data_args.trained_prompt_path)
        task_start_id = data_args.prompt_number * len(format2dataset.keys())

        b_task_id = task_start_id + task2id[data_args.task_b_name] * data_args.prompt_number
        b_format_id = format2id[dataset2format[data_args.task_b_name]]
        # a_format_task_id = task_start_id + task2id[dataset2format[data_args.continual_task_a_name]] * data_args.prompt_number
        model.state_dict()['encoder.prompt_embeddings.weight'][b_task_id:b_task_id + data_args.prompt_number, :] = \
            prompt_info['encoder.prompt_embeddings.weight'][b_task_id:b_task_id + data_args.prompt_number, :]
        model.state_dict()['encoder.prompt_embeddings.weight'][
        b_format_id * data_args.prompt_number:(b_format_id + 1) * data_args.prompt_number, :] = \
            prompt_info['encoder.prompt_embeddings.weight'][
            b_format_id * data_args.prompt_number:(b_format_id + 1) * data_args.prompt_number, :]
        logger.info(
            f'Successfully restore task+format prompt for the task {data_args.task_b_name} from {data_args.trained_prompt_path}')
    if data_args.load_from_format_task_id and (data_args.dataset_name not in seed_datasets) and not data_args.reload_from_trained_prompt and ((data_args.dataset_name==data_args.task_b_name) or not data_args.task_b_name):
        task_start_id = data_args.prompt_number * len(format2dataset.keys())
        task_id = task_start_id + task2id[data_args.dataset_name] * data_args.prompt_number
        format_task_id = task_start_id + task2id[dataset2format[data_args.dataset_name]] * data_args.prompt_number
        model.state_dict()['encoder.prompt_embeddings.weight'][task_id:task_id+data_args.prompt_number,:] =  model.state_dict()['encoder.prompt_embeddings.weight'][format_task_id:format_task_id+data_args.prompt_number,:]
        logger.info(f'Successfully initialize format {dataset2format[data_args.dataset_name]} task prompt for new task {data_args.dataset_name}, task id {task_id}')
            # print(dataset2format[data_args.dataset_name])
            # print(data_args.dataset_name)
    elif data_args.reload_from_trained_prompt:
        assert data_args.trained_prompt_path,'Must specify the path of stored prompt'
        prompt_info = torch.load(data_args.trained_prompt_path)
        assert prompt_info['task2id'][data_args.dataset_name]==task2id[data_args.dataset_name],f'the task id in trained prompt task id is not matched to the current task id for {data_args.dataset_name}'
        assert prompt_info['format2id'].keys()==format2id.keys(),'the format dont match'
        task_start_id = data_args.prompt_number * len(format2dataset.keys())

        task_id = task_start_id + task2id[data_args.dataset_name] * data_args.prompt_number
        logger.info('task id range {} {}'.format(task_id,task_id+data_args.prompt_number))
        # assert torch.sum(model.state_dict()['encoder.prompt_embeddings.weight'][task_id:task_id+data_args.prompt_number,:] - prompt_info['encoder.prompt_embeddings.weight'][task_id:task_id+data_args.prompt_number,:])==0
        model.state_dict()['encoder.prompt_embeddings.weight'][task_id:task_id+data_args.prompt_number,:] = prompt_info['encoder.prompt_embeddings.weight'][task_id:task_id+data_args.prompt_number,:]
        format_id = format2id[dataset2format[data_args.dataset_name]]
        model.state_dict()['encoder.prompt_embeddings.weight'][format_id*data_args.prompt_number:(format_id+1)*data_args.prompt_number, :] = prompt_info['encoder.prompt_embeddings.weight'][format_id*data_args.prompt_number:(format_id+1)*data_args.prompt_number, :]
        logger.info(
            f'Successfully restore task+format prompt for the task {data_args.dataset_name} from {data_args.trained_prompt_path}')

    if model_args.fix_t5:
        for name, param in model.named_parameters():
            if 'prompt' not in name:
                param.requires_grad = False
        logger.info('fixed t5 parameters')
    # Set decoder_start_token_id
    if model.config.decoder_start_token_id is None and isinstance(tokenizer, (MBartTokenizer, MBartTokenizerFast)):
        if isinstance(tokenizer, MBartTokenizer):
            model.config.decoder_start_token_id = tokenizer.lang_code_to_id[data_args.target_lang]
        else:
            model.config.decoder_start_token_id = tokenizer.convert_tokens_to_ids(data_args.target_lang)

    if model.config.decoder_start_token_id is None:
        raise ValueError("Make sure that `config.decoder_start_token_id` is correctly defined")

    prefix = data_args.source_prefix if data_args.source_prefix is not None else ""
    if training_args.local_rank == -1 or training_args.no_cuda:
        device = torch.device("cuda")
        n_gpu = torch.cuda.device_count()
    
#     model.to(device)
#     if training_args.local_rank != -1:
#         model = torch.nn.parallel.DistributedDataParallel(model, device_ids=[training_args.local_rank],
#                                                           output_device=training_args.local_rank,
#                                                           find_unused_parameters=True)
#     elif n_gpu > 1:
#         model = torch.nn.DataParallel(model)
    # Preprocessing the datasets.
    # We need to tokenize inputs and targets.
    if training_args.do_train:
        column_names = raw_datasets["train"].column_names
    elif training_args.do_eval:
        column_names = raw_datasets["validation"].column_names
    elif training_args.do_predict:
        column_names = raw_datasets["validation"].column_names
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
    # def flatten(answers):
    #     new_answers, metadata = [], []
    #     for answer in answers:
    #         assert type(answer)==list
    #         metadata.append((len(new_answers), len(new_answers)+len(answer)))
    #         new_answers += answer
    #     return new_answers, metadata
    # dataset_columns = question_answering_column_name_mapping.get(data_args.dataset_name, None)

    question_column = data_args.question_column
    context_column = data_args.context_column
    answer_column = data_args.answer_column

    # import random
    if data_args.max_source_length > tokenizer.model_max_length:
        logger.warning(
            f"The max_seq_length passed ({data_args.max_source_length}) is larger than the maximum length for the"
            f"model ({tokenizer.model_max_length}). Using max_seq_length={tokenizer.model_max_length}."
        )
    data_args.max_source_length = min(data_args.max_source_length, tokenizer.model_max_length)

    def dataset_name_to_func(dataset_name):
        mapping = {
            'squad': preprocess_sqaud_batch,
            'squad_v2': preprocess_sqaud_batch,
            'boolq': preprocess_boolq_batch,
            'narrativeqa': preprocess_narrativeqa_batch,
            'race': preprocess_race_batch,
            'newsqa': preprocess_newsqa_batch,
            'quoref': preprocess_sqaud_batch,
            'ropes': preprocess_ropes_batch,
            'drop': preprocess_drop_batch,
            'nqopen': preprocess_sqaud_abstractive_batch,
            # 'multirc': preprocess_boolq_batch,
            'boolq_np': preprocess_boolq_batch,
            'openbookqa': preprocess_openbookqa_batch,
            'mctest': preprocess_race_batch,
            'social_iqa': preprocess_social_iqa_batch,
            'dream': preprocess_dream_batch,
        }
        return mapping[dataset_name]

    def preprocess_function(examples):
        preprocess_fn = dataset_name_to_func(data_args.dataset_name)
        inputs, targets = preprocess_fn(examples, question_column, context_column, answer_column)
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
        format_id = format2id[dataset2format[data_args.dataset_name]]
        format_prompt_ids = [- (i + 1) for i in range(format_id * data_args.prompt_number, (
                    format_id + 1) * data_args.prompt_number)]  # list(range(-(format_id * data_args.prompt_number+1), -((format_id + 1) * data_args.prompt_number+1)))
        task_prompt_id_start = len(format2id.keys()) * data_args.prompt_number
        task_id = task2id[data_args.dataset_name]
        task_prompt_ids = [- (i + 1) for i in range(task_prompt_id_start + task_id * data_args.prompt_number,
                                                    task_prompt_id_start + (task_id + 1) * data_args.prompt_number)]

        input_ids = copy.deepcopy(
            [format_prompt_ids + task_prompt_ids + input_ids for input_ids in model_inputs['input_ids']])
        model_inputs['input_ids'] = input_ids  # [format_prompt_ids+input_ids for input_ids in model_inputs['input_ids']]
        model_inputs['attention_mask'] = [[1] * data_args.prompt_number * 2 + attention_mask for attention_mask in
                                          model_inputs['attention_mask']]
        # task_id = task2id[data_args.dataset_name]
        # format_id = format2id[dataset2format[data_args.dataset_name]]
        # # model_inputs['task'] = [list(range(task_id*data_args.prompt_number,(task_id+1)*data_args.prompt_number)) for input in inputs]
        # # model_inputs['format'] = [list(range(format_id*data_args.prompt_number,(format_id+1)*data_args.prompt_number)) for input in inputs]
        # format_prompt_ids = [- (i + 1) for i in range(format_id * data_args.prompt_number, (
        #             format_id + 1) * data_args.prompt_number)]  # list(range(-(format_id * data_args.prompt_number+1), -((format_id + 1) * data_args.prompt_number+1)))
        # input_ids = copy.deepcopy([format_prompt_ids + input_ids for input_ids in model_inputs['input_ids']])
        # model_inputs[
        #     'input_ids'] = input_ids  # [format_prompt_ids+input_ids for input_ids in model_inputs['input_ids']]
        # model_inputs['attention_mask'] = [[1] * data_args.prompt_number + attention_mask for attention_mask in
        #                                   model_inputs['attention_mask']]
        # print(model_inputs['input_ids'][0])
        return model_inputs

    def preprocess_validation_function(examples):
        preprocess_fn = dataset_name_to_func(data_args.dataset_name)
        inputs, targets = preprocess_fn(examples, question_column, context_column, answer_column)
        model_inputs = tokenizer(inputs, max_length=data_args.max_source_length, padding=padding, truncation=True)
        # Setup the tokenizer for targets
        model_inputs["example_id"] = []

        for i in range(len(model_inputs["input_ids"])):
            # One example can give several spans, this is the index of the example containing this span of text.
            sample_index = i #sample_mapping[i]
            model_inputs["example_id"].append(examples["id"][sample_index])
        with tokenizer.as_target_tokenizer():

            labels = tokenizer(targets, max_length=data_args.max_target_length, padding=padding, truncation=True)

        # If we are padding here, replace all tokenizer.pad_token_id in the labels by -100 when we want to ignore
        # padding in the loss.
        if padding == "max_length" and data_args.ignore_pad_token_for_loss:
            labels["input_ids"] = [
                [(l if l != tokenizer.pad_token_id else -100) for l in label] for label in labels["input_ids"]
            ]
        model_inputs["labels"] = labels["input_ids"]
        if data_args.task_b_name is not None and not data_args.reload_from_trained_prompt:
            logger.info(f'Loading task {data_args.task_b_name} prompt')
            format_id = format2id[dataset2format[data_args.task_b_name]]
            task_id = task2id[data_args.task_b_name]
        else:
            logger.info(f'Loading task {data_args.dataset_name} prompt')
            format_id = format2id[dataset2format[data_args.dataset_name]]
            task_id = task2id[data_args.dataset_name]
        format_prompt_ids = [- (i + 1) for i in range(format_id * data_args.prompt_number, (
                format_id + 1) * data_args.prompt_number)]  # list(range(-(format_id * data_args.prompt_number+1), -((format_id + 1) * data_args.prompt_number+1)))
        task_prompt_id_start = len(format2id.keys()) * data_args.prompt_number
        # logger.info('Prompt ids {}: {}'.format(task_prompt_id_start + task_id * data_args.prompt_number,
        #                                             task_prompt_id_start + (task_id + 1) * data_args.prompt_number))
        task_prompt_ids = [- (i + 1) for i in range(task_prompt_id_start + task_id * data_args.prompt_number,
                                                    task_prompt_id_start + (task_id + 1) * data_args.prompt_number)]
        input_ids = copy.deepcopy(
            [format_prompt_ids + task_prompt_ids + input_ids for input_ids in model_inputs['input_ids']])
        # input_ids = copy.deepcopy([format_prompt_ids + input_ids for input_ids in model_inputs['input_ids']])
        model_inputs['input_ids'] = input_ids
        model_inputs['attention_mask'] = [[1] * data_args.prompt_number * 2 + attention_mask for attention_mask in
                                          model_inputs['attention_mask']]
        # task_id = task2id[data_args.dataset_name]
        # format_id = format2id[dataset2format[data_args.dataset_name]]
        # # model_inputs['task'] = [list(range(task_id * data_args.prompt_number, (task_id + 1) * data_args.prompt_number))
        # #                         for input in inputs]
        # # model_inputs['format'] = [
        # #     list(range(format_id * data_args.prompt_number, (format_id + 1) * data_args.prompt_number)) for input in
        # #     inputs]
        # format_prompt_ids = [- (i + 1) for i in range(format_id * data_args.prompt_number, (
        #         format_id + 1) * data_args.prompt_number)]  # list(range(-(format_id * data_args.prompt_number+1), -((format_id + 1) * data_args.prompt_number+1)))
        # input_ids = copy.deepcopy([format_prompt_ids + input_ids for input_ids in model_inputs['input_ids']])
        # model_inputs['input_ids'] = input_ids
        # model_inputs['attention_mask'] = [[1] * data_args.prompt_number + attention_mask for attention_mask in
        #                                   model_inputs['attention_mask']]
        # print(model_inputs['input_ids'][0])
        return model_inputs

    if training_args.do_train:
        if "train" not in raw_datasets:
            raise ValueError("--do_train requires a train dataset")
        train_dataset = raw_datasets["train"]
        if data_args.max_train_samples is not None:
            all_num = list(range(0, len(train_dataset)))
            random.shuffle(all_num)
            selected_indices = all_num[:data_args.max_train_samples]
            train_dataset = train_dataset.select(selected_indices)
            # train_dataset = train_dataset.select(range(data_args.max_train_samples))
        with training_args.main_process_first(desc="train dataset map pre-processing"):
            train_dataset = train_dataset.map(
                preprocess_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on train dataset",
            )

    if training_args.do_eval:
        max_target_length = data_args.val_max_target_length
        if "validation" not in raw_datasets:
            raise ValueError("--do_eval requires a validation dataset")
        eval_examples = raw_datasets["validation"]
        def add_id(example,index):
            example.update({'id':index})
            return example
        if 'id' not in eval_examples.features.keys():
            eval_examples = eval_examples.map(add_id,with_indices=True)
        if data_args.max_eval_samples is not None:
            eval_examples = eval_examples.select(range(data_args.max_eval_samples))
        with training_args.main_process_first(desc="validation dataset map pre-processing"):
            eval_dataset = eval_examples.map(
                preprocess_validation_function,
                batched=True,
                num_proc=data_args.preprocessing_num_workers,
                remove_columns=column_names,
                load_from_cache_file=not data_args.overwrite_cache,
                desc="Running tokenizer on validation dataset",
            )

    if training_args.do_predict:
        max_target_length = data_args.val_max_target_length
        if "test" not in raw_datasets:
            raise ValueError("--do_predict requires a test dataset")
        predict_dataset = raw_datasets["test"]
        def add_id(example,index):
            example.update({'id':index})
            return example
        if 'id' not in predict_dataset.features.keys():
            predict_dataset = predict_dataset.map(add_id,with_indices=True)
        if data_args.max_predict_samples is not None:
            predict_dataset = predict_dataset.select(range(data_args.max_predict_samples))
        with training_args.main_process_first(desc="prediction dataset map pre-processing"):
            predict_dataset = predict_dataset.map(
                preprocess_validation_function,
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
    dataset_name_to_metric = {
        'squad': 'squad',
        'squad_v2': 'metric/squad_v2_local/squad_v2_local.py',
        'newsqa': 'metric/squad_v2_local/squad_v2_local.py',
        'boolq': 'accuracy',
        'narrativeqa': 'rouge',
        'race': 'accuracy',
        'quoref': 'squad',
        'ropes': 'squad',
        'drop': 'squad',
        'nqopen': 'squad',
        # 'multirc': 'accuracy',
        'boolq_np': 'accuracy',
        'openbookqa': 'accuracy',
        'mctest': 'accuracy',
        'social_iqa': 'accuracy',
        'dream': 'accuracy',
    }
    metric = load_metric(dataset_name_to_metric[data_args.dataset_name])
    # metric = load_metric("squad")
    def compute_metrics(p: EvalPrediction):
        return metric.compute(predictions=p.predictions, references=p.label_ids)
    '''
    from metrics import evaluate_squad
    def compute_metrics(eval_preds):
        preds, labels = eval_preds
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
        if data_args.ignore_pad_token_for_loss:
            # Replace -100 in the labels as we can't decode them.
            labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
        decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
 
        # Some simple post-processing
        decoded_preds, decoded_labels = postprocess_text(decoded_preds, decoded_labels)
        # print(decoded_preds,decoded_labels)
        # result = metric.compute(predictions=decoded_preds, references=decoded_labels)
        # result = {"EM": result["exact_match"],"F1":result['f1']}
        result = evaluate_squad(predictions=decoded_preds,labels=decoded_labels)
        # prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        # result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result
    '''
    # Initialize our Trainer
    trainer = QuestionAnsweringTrainer(
        model=model,
        args=training_args,
        train_dataset=train_dataset if training_args.do_train else None,
        eval_dataset=eval_dataset if training_args.do_eval else None,
        eval_examples=eval_examples if training_args.do_eval else None,
        answer_column_name=answer_column,
        dataset_name=data_args.dataset_name,
        tokenizer=tokenizer,
        data_collator=data_collator,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None,
    )
    # trainer.is_model_parallel=True
#     trainer.place_model_on_device=True
    # Training
    def save_prompt_embedding(model):
        prompt_embedding = model.state_dict()['encoder.prompt_embeddings.weight']
        save_prompt_info = {'encoder.prompt_embeddings.weight':copy.deepcopy(prompt_embedding),'task2id':task2id,'format2id':format2id}
        prompt_path = os.path.join(training_args.output_dir,'prompt_embedding_info')
        torch.save(save_prompt_info,prompt_path)
        logger.info(f'Saving prompt embedding information to {prompt_path}')

    if training_args.do_train:
        best_dir = os.path.join(training_args.output_dir, 'best-checkpoint')
        if not os.path.exists(best_dir):
            Path(best_dir).mkdir(exist_ok=True)
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
        torch.cuda.empty_cache()
        logger.info("*** Evaluate ***")
        if training_args.do_train:
            from transformers.file_utils import (
                CONFIG_NAME,
                WEIGHTS_NAME)
            best_model_path = os.path.join(training_args.output_dir, 'best-checkpoint')
            state_dict = torch.load(os.path.join(best_model_path,WEIGHTS_NAME), map_location="cpu")
            trainer._load_state_dict_in_model(state_dict)

        metrics = trainer.evaluate(max_length=max_length, num_beams=num_beams, metric_key_prefix="eval")
        max_eval_samples = data_args.max_eval_samples if data_args.max_eval_samples is not None else len(eval_dataset)
        metrics["eval_samples"] = min(max_eval_samples, len(eval_dataset))
        if data_args.max_train_samples==32 and training_args.do_train:
            pre = 'fewshot'
        elif training_args.do_train:
            pre = 'fulldata'
        elif training_args.do_eval and not training_args.do_train and not data_args.reload_from_trained_prompt:
            pre = 'zeroshot'
        elif training_args.do_eval and not training_args.do_train and data_args.reload_from_trained_prompt:
            pre = 'continual-testa'
        else:
            pre = 'unknown'
        split_model_name = [item for item in model_args.model_name_or_path.strip('/').split('/') if 'pretrain' in item]
        save_model_config = split_model_name[0]#'-'.join(split_model_name[-2:]) if 'checkpoint' in model_args.model_name_or_path else split_model_name[-1]+'-final'
        result_path = os.path.join('/home/t-wzhong/v-wanzho/promptQA/eval_results/','{}_{}_{}_results.json'.format(data_args.dataset_name,pre,save_model_config))
        with open(result_path,'a+',encoding='utf8') as outf:
            import json
            # if training_args.do_train:
            #     metrics = trainer.state.best_metric
            if pre=='continual-testa' or pre=='zeroshot':
                metrics.update({'task_b':data_args.task_b_name})
            metrics.update({'model_name_path': model_args.model_name_or_path})
            json.dump(metrics,outf,indent=4)
            print('Saving results to '+result_path)
        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)

    if training_args.do_predict:
        torch.cuda.empty_cache()
        logger.info("*** Predict ***")

        max_predict_samples = (
            data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        )
        metrics = trainer.predict(predict_dataset=predict_dataset, predict_examples=max_predict_samples, max_length=max_length, num_beams=num_beams, metric_key_prefix="predict")

        metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))

        trainer.log_metrics("predict", metrics)
        trainer.save_metrics("predict", metrics)

        # predict_results = trainer.predict(
        #     predict_dataset, metric_key_prefix="predict", max_length=max_length, num_beams=num_beams
        # )
        # metrics = predict_results.metrics
        # max_predict_samples = (
        #     data_args.max_predict_samples if data_args.max_predict_samples is not None else len(predict_dataset)
        # )
        # metrics["predict_samples"] = min(max_predict_samples, len(predict_dataset))
        #
        # trainer.log_metrics("predict", metrics)
        # trainer.save_metrics("predict", metrics)
        #
        # if trainer.is_world_process_zero():
        #     if training_args.predict_with_generate:
        #         predictions = tokenizer.batch_decode(
        #             predict_results.predictions, skip_special_tokens=True, clean_up_tokenization_spaces=True
        #         )
        #         predictions = [pred.strip() for pred in predictions]
        #         output_prediction_file = os.path.join(training_args.output_dir, "generated_predictions.txt")
        #         with open(output_prediction_file, "w", encoding="utf-8") as writer:
        #             writer.write("\n".join(predictions))

    kwargs = {"finetuned_from": model_args.model_name_or_path, "tasks": "question answering"}
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

    # if training_args.push_to_hub:
    #     trainer.push_to_hub(**kwargs)
    # else:
    #     trainer.create_model_card(**kwargs)

    return results


# -

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()


if __name__ == "__main__":
    main()
