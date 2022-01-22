import dataclasses
import json
import logging
import os
import sys
from dataclasses import dataclass, field
from typing import Dict, List, Optional
from tqdm import tqdm
import numpy as np
import nlp
import torch
import nltk
nltk.download('punkt')
from datasets import load_dataset,load_metric
from transformers import (
    AutoModelForSeq2SeqLM,
    AutoTokenizer,
    T5Tokenizer,
    BartTokenizer,
    HfArgumentParser,
    EvalPrediction,
    DataCollator,
    TrainingArguments,
    Seq2SeqTrainer,
    Seq2SeqTrainingArguments,
    set_seed,
)
from MultipleChoiceTrainer import MultipleSeq2SeqTrainer
from data_collator import T2TDataCollator
from utils import freeze_embeds, assert_not_all_frozen
os.environ["WANDB_DISABLED"] = "true"
MODEL_TYPE_TO_TOKENIZER = {
    "t5": T5Tokenizer,
    "bart": BartTokenizer,
}


logger = logging.getLogger(__name__)


@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to fine-tune from.
    """

    model_name_or_path: str = field(
        metadata={"help": "Path to pretrained model or model identifier from huggingface.co/models"}
    )
    model_type: str = field(metadata={"help": "One of 't5', 'bart'"})
    tokenizer_name_or_path: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )
    cache_dir: Optional[str] = field(
        default=None, metadata={"help": "Where do you want to store the pretrained models downloaded from s3"}
    )
    label_smoothing: Optional[float] = field(
        default=0,
        metadata={"help": "label smoothing rate, set to > 0 if you want to enable lable smoothing"}
    )
    freeze_embeds: bool = field(
        default=False,
        metadata={"help": "Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."}
    )
    use_fast_tokenizer: bool = field(
        default=True,
        metadata={"help": "Whether to use one of the fast tokenizer (backed by the tokenizers library) or not."},
    )
    tokenizer_name: Optional[str] = field(
        default=None, metadata={"help": "Pretrained tokenizer name or path if not the same as model_name"}
    )

@dataclass
class DataTrainingArguments:
    """
    Arguments pertaining to what data we are going to input our model for training and eval.
    """
    ignore_pad_token_for_loss: bool = field(
        default=True,
        metadata={
            "help": "Whether to ignore the tokens corresponding to padded labels in the loss computation or not."
        },
    )
    num_beams: Optional[int] = field(
        default=4,
        metadata={"help": "num_beams to use for decoding"}
    )
    # model_type: Optional[str] = field(
    #     default='t5',
    #     metadata={"help": "Path for cached valid dataset"},
    # )
    train_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path for cached train dataset"},
    )
    max_decoding_length: Optional[int] = field(
        default=32,
        metadata={"help": "maximum length for decoding"}
    )
    valid_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path for cached valid dataset"},
    )

    test_file_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path for cached valid dataset"},
    )
    data_dir: Optional[str] = field(
        default=None,
        metadata={"help": "Path for data files"}, 
    )
    task: Optional[str] = field(
        default=None,
        metadata={"help": "Which task 'qa', 'qg', 'e2e_qg', 'ans_ext', 'multi'. 'multi' means 'qa', 'qg', 'ans_ext' tasks"}, 
    )
    qg_format: Optional[str] = field(
        default='prepend_qg_format',
        metadata={"help": "How to format inputs for que generation, 'highlight_qg_format' or 'prepend_qg_format'"}, 
    )
    max_source_length: Optional[int] = field(
        default=512,
        metadata={"help": "Max input length for the source text"},
    )
    max_target_length: Optional[int] = field(
        default=32,
        metadata={"help": "Max input length for the target text"},
    )
    output_path: Optional[str] = field(
        default=None,
        metadata={"help": "Path for data files"},
    )
    num_return_seq: Optional[int] = field(
        default=4,
        metadata={"help": "Max input length for the target text"},
    )
    max_test_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Max input length for the target text"},
    )
    max_train_samples: Optional[int] = field(
        default=None,
        metadata={"help": "Max input length for the target text"},
    )
    overwrite_cache: bool = field(
        default=False, metadata={"help": "Overwrite the cached training and evaluation sets"}
    )


    # remove_unused_columns: bool = field(
    #     default=False,
    #     metadata={"help": "Freeze token embeddings and positional embeddings for bart, just token embeddings for t5."}
    # )


class DataProcessor:
    def __init__(self, tokenizer, model_type="t5", max_source_length=512, max_target_length=32):
        self.tokenizer = tokenizer
        self.max_source_length = max_source_length
        self.max_target_length = max_target_length
        self.model_type = model_type
        self.hl_token = "<hl>"

        if model_type == "t5":
            self.sep_token = "<sep>"
        elif model_type == "bart":
            self.sep_token = "<sep>"
        else:
            self.sep_token = "[SEP]"

    def process(self, dataset):
        if self.model_type == "t5":
            dataset = dataset.map(self._add_eos_examples)

        dataset = dataset.map(self._add_special_tokens)
        dataset = dataset.map(self._convert_to_features, batched=True)

        return dataset

    def _add_eos_examples(self, example):
        example['source_text'] = example['source_text'].strip('</s>')  # + " </s>"
        example['target_text'] = example['target_text']#.lower().capitalize()  # + " </s>"
        # example['target_text'] = example['answer']
        return example

    def _add_special_tokens(self, example):
        example['source_text'] = example['source_text'].replace("{hl_token}", self.hl_token)
        example['target_text'] = example['target_text'].replace("{sep_token}", self.sep_token)
        return example

    # tokenize the examples
    def _convert_to_features(self, example_batch):
        source_encoding = self.tokenizer.batch_encode_plus(
            example_batch['source_text'],
            max_length=self.max_source_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True,
        )
        target_encoding = self.tokenizer.batch_encode_plus(
            example_batch['target_text'],
            max_length=self.max_target_length,
            padding='max_length',
            pad_to_max_length=True,
            truncation=True,
        ),
        # print(target_encoding)
        encodings = {
            'source_ids': source_encoding['input_ids'],
            'target_ids': target_encoding[0]['input_ids'],
            'attention_mask': source_encoding['attention_mask'],
        }

        return encodings

def main(args_file=None):
    # See all possible arguments in src/transformers/training_args.py
    # or by passing the --help flag to this script.
    # We now keep distinct sets of args, for a cleaner separation of concerns.

    parser = HfArgumentParser((ModelArguments, DataTrainingArguments, Seq2SeqTrainingArguments))

    if (len(sys.argv) == 2 and sys.argv[1].endswith(".json")) or args_file is not None:
        # If we pass only one argument to the script and it's the path to a json file,
        # let's parse it to get our arguments.
        args_file_path = os.path.abspath(sys.argv[1]) if args_file is None else args_file
        model_args, data_args, training_args = parser.parse_json_file(json_file=args_file_path)
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()

    assert model_args.model_type in list(MODEL_TYPE_TO_TOKENIZER.keys()), "model type should be 't5' or 'bart'"

    if (
        os.path.exists(training_args.output_dir)
        and os.listdir(training_args.output_dir)
        and training_args.do_train
        and not training_args.overwrite_output_dir
    ):
        raise ValueError(
            f"Output directory ({training_args.output_dir}) already exists and is not empty. Use --overwrite_output_dir to overcome."
        )

    # Setup logging
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        level=logging.INFO if training_args.local_rank in [-1, 0] else logging.WARN,
    )
    logger.warning(
        "Process rank: %s, device: %s, n_gpu: %s, distributed training: %s, 16-bits training: %s",
        training_args.local_rank,
        training_args.device,
        training_args.n_gpu,
        bool(training_args.local_rank != -1),
        training_args.fp16,
    )
    logger.info("Training/evaluation parameters %s", training_args)

    # Set seed
    set_seed(training_args.seed)

    # Set project name
    os.environ["WANDB_PROJECT"] = "question-generation"

    # Load pretrained model and tokenizer
    #
    # Distributed training:
    # The .from_pretrained methods guarantee that only one local process can concurrently
    # download model & vocab.
    # tokenizer_cls = MODEL_TYPE_TO_TOKENIZER[model_args.model_type]
    # tokenizer = tokenizer_cls.from_pretrained(
    #     model_args.tokenizer_name_or_path if model_args.tokenizer_name_or_path else model_args.model_name_or_path,
    #     cache_dir=model_args.cache_dir,
    # )
    tokenizer = AutoTokenizer.from_pretrained(
        model_args.tokenizer_name if model_args.tokenizer_name else model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
        use_fast=model_args.use_fast_tokenizer,
    )
    if training_args.do_train:
        tokenizer.add_tokens(['[Context]','[question]','[answer]','<hl>'])
        added_tokens = tokenizer.get_added_vocab()
        logger.info('Added tokens: {}'.format(added_tokens))
    # tokenizer.add_tokens(['<sep>', '<hl>']),'[Question]','[Answer]'])
    model = AutoModelForSeq2SeqLM.from_pretrained(
        model_args.model_name_or_path,
        cache_dir=model_args.cache_dir,
    )

    model.resize_token_embeddings(len(tokenizer))

    if model_args.freeze_embeds:
        logger.info("freezing embeddings of the model")
        freeze_embeds(model)
        assert_not_all_frozen(model)

    # Get datasets
    logger.info('loading dataset')
    data_files = {
        'train':os.path.join(data_args.data_dir, data_args.train_file_path),#if training_args.do_train else None,
        'validation': os.path.join(data_args.data_dir, data_args.valid_file_path), #if training_args.do_eval else None,
        'test':data_args.test_file_path, #if training_args.do_predict else None
    }
    # training_args.do_eval = False
    train_dataset = load_dataset('json', data_files=data_files['train'])['train'] if training_args.do_train else None
    valid_dataset = load_dataset('json', data_files=data_files['validation'])['train'] if training_args.do_eval else None
    predict_dataset = load_dataset('json', data_files=data_files['test'])['train'] if training_args.do_predict else None
    # print(predict_dataset.keys())
    if data_args.max_train_samples and training_args.do_train:
        train_dataset = train_dataset.shuffle()
        train_dataset = train_dataset.select(range(data_args.max_train_samples))
    if data_args.max_test_samples and training_args.do_predict:
        predict_dataset = predict_dataset.shuffle()
        predict_dataset = predict_dataset.select(range(data_args.max_test_samples))
    # train_dataset = torch.load(data_args.train_file_path) if training_args.do_train else None
    # valid_dataset = torch.load(data_args.valid_file_path) if training_args.do_eval else None
    # predict_dataset = torch.load(data_args.test_file_path) if training_args.do_predict else None
    # columns = ["source_ids", "target_ids", "attention_mask"]
    # train_dataset.set_format(type='torch', columns=columns)
    # valid_dataset.set_format(type='torch', columns=columns)
    # logger.info(train_dataset.features.keys())
    processor = DataProcessor(
        tokenizer,
        model_type=model_args.model_type,
        max_source_length=data_args.max_source_length,
        max_target_length=data_args.max_target_length
    )
    train_dataset = processor.process(train_dataset) if training_args.do_train else None
    valid_dataset = processor.process(valid_dataset) if training_args.do_eval else None
    processed_predict_dataset = processor.process(predict_dataset) if training_args.do_predict else None
    logger.info('finished loading dataset')

    # Initialize data_collator
    data_collator = T2TDataCollator(
        tokenizer=tokenizer,
        model_type=model_args.model_type,
        mode="training",
        using_tpu=training_args.tpu_num_cores is not None
    )

    # Initialize our Trainer

    # trainer = Seq2SeqTrainer(
    #     model=model,
    #     args=training_args,
    #     train_dataset=train_dataset,
    #     eval_dataset=valid_dataset,
    #     data_collator=data_collator,
    #     # label_smoothing=model_args.label_smoothing
    # )
    metric = load_metric("myrouge")

    def postprocess_text(preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]

        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]

        return preds, labels
    metric = load_metric('myrouge')
    # def compute_metrics(p: EvalPrediction):
    #     return metric.compute(predictions=p.predictions, references=p.label_ids)
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
        # pdb.set_trace()
        result, accuracy = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)
        # Extract a few results from ROUGE
        result = {key: value.mid.fmeasure * 100 for key, value in result.items()}
        result['accuracy'] = accuracy

        prediction_lens = [np.count_nonzero(pred != tokenizer.pad_token_id) for pred in preds]
        result["gen_len"] = np.mean(prediction_lens)
        result = {k: round(v, 4) for k, v in result.items()}
        return result

    trainer = MultipleSeq2SeqTrainer(
        model=model,
        tokenizer = tokenizer,
        args=training_args,
        train_dataset=train_dataset,
        eval_dataset=valid_dataset,
        data_collator=data_collator,
        num_return_seq = data_args.num_return_seq,
        num_beams=data_args.num_beams,
        compute_metrics=compute_metrics if training_args.predict_with_generate else None
        # label_smoothing=model_args.label_smoothing
    )

    # disable wandb console logs
    logging.getLogger('wandb.run_manager').setLevel(logging.WARNING)
    # train_dataset.set_format(type=train_dataset.format["type"], columns=list(train_dataset.features.keys()))
    # valid_dataset.set_format(type=valid_dataset.format["type"], columns=list(valid_dataset.features.keys()))
    # Training
    if training_args.do_train:

        trainer.train(
            model_path=model_args.model_name_or_path if os.path.isdir(model_args.model_name_or_path) else None
        )
        trainer.save_model()
        # For convenience, we also re-save the tokenizer to the same directory,
        # so that you can share your model easily on huggingface.co/models =)
        if trainer.is_world_master():
            tokenizer.save_pretrained(training_args.output_dir)

    # Evaluation
    results = {}
    if training_args.do_eval and training_args.local_rank in [-1, 0]:
        logger.info("*** Evaluate ***")

        eval_output = trainer.evaluate(
            max_length=data_args.max_decoding_length, num_beams=data_args.num_beams, metric_key_prefix="eval"
        )
        trainer.log_metrics("eval", eval_output)
        output_eval_file = os.path.join(training_args.output_dir, "eval_results.txt")
        with open(output_eval_file, "w") as writer:
            logger.info("***** Eval results *****")
            for key in sorted(eval_output.keys()):
                logger.info("  %s = %s", key, str(eval_output[key]))
                writer.write("%s = %s\n" % (key, str(eval_output[key])))
    
        results.update(eval_output)

    if training_args.do_predict:
        logger.info("*** Predict ***")
        split = 5
        for i in tqdm(range(split)):
            select_indices = range(i*int(len(processed_predict_dataset)/split),(i+1)*int(len(processed_predict_dataset)/split))
            split_predict_dataset = processed_predict_dataset.select(select_indices)
            split_ori_dataset = predict_dataset.select(select_indices)
            predict_results = trainer.predict(
                split_predict_dataset, metric_key_prefix="predict", max_length=data_args.max_decoding_length, num_beams=data_args.num_beams
            )
            outputs = predict_results.predictions
            predictions = tokenizer.batch_decode(
                outputs.reshape(outputs.shape[0] * outputs.shape[1], outputs.shape[-1]), skip_special_tokens=True,
                clean_up_tokenization_spaces=True
            )
            predictions = [pred.strip() for pred in predictions]
            output_prediction_file = data_args.output_path+f'_{i}'  # os.path.join(training_args.output_dir, "generated_predictions.txt")
            print(f'output results to {output_prediction_file}')
            with open(output_prediction_file, "w", encoding="utf-8") as writer:
                for j, example in tqdm(enumerate(split_ori_dataset)):
                    preds = predictions[int(j * data_args.num_return_seq):int((j + 1) * data_args.num_return_seq)]
                    if len(preds) == 0:
                        print(example, len(predictions)) 
                    writer.write(json.dumps({'source_text': example['source_text'], 'predictions': preds}) + '\n')
        ''' 
        predict_results = trainer.predict(
            processed_predict_dataset_dataset, metric_key_prefix="predict", max_length=data_args.max_decoding_length,
            num_beams=data_args.num_beams
        )
        outputs = predict_results.predictions
        # print(outputs.device())
        predictions = tokenizer.batch_decode(
            outputs.reshape(outputs.shape[0]*outputs.shape[1],outputs.shape[-1]), skip_special_tokens=True, clean_up_tokenization_spaces=True
        )
        # outputs = outputs.reshape(outputs.shape[0]*outputs.shape[1],outputs.shape[-1])
        # predictions = []
        # split = 10
        # print(outputs)
        # input()
        # print('decoding predictions')
        # for i in tqdm(range(10)):
        #     tmp_results = tokenizer.batch_decode(
        #         outputs[data_args.num_return_seq*i*int(len(outputs)/split):(i+1)*data_args.num_return_seq*int(len(outputs)/split)],skip_special_tokens=True,clean_up_tokenization_spaces=True
        #     )
        #     predictions.extend(tmp_results)
        # print(type(outputs))
        predictions = [pred.strip() for pred in predictions]
        output_prediction_file = data_args.output_path#os.path.join(training_args.output_dir, "generated_predictions.txt")
        print(f'output results to {output_prediction_file}')

        with open(output_prediction_file, "w", encoding="utf-8") as writer:
            for i,example in tqdm(enumerate(predict_dataset)):
                preds = predictions[int(i*data_args.num_return_seq):int((i+1)*data_args.num_return_seq)]
                if len(preds)==0:
                    print(example,len(predictions))
                writer.write(json.dumps({'source_text':example['source_text'],'predictions': preds})+'\n')
            # writer.write("\n".join(predictions))
    '''


    return results

def _mp_fn(index):
    # For xla_spawn (TPUs)
    main()

def run_qg(args_dict):
    with open("args.json", 'w') as f:
        json.dump(args_dict, f)
    
    main(args_file="args.json")

if __name__ == "__main__":
    main()