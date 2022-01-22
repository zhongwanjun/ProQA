from transformers import Seq2SeqTrainer, is_torch_tpu_available, EvalPrediction
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Optional, Tuple, Union
import nltk
import datasets
import re
import numpy as np
from transformers.trainer_utils import PredictionOutput,EvalLoopOutput
if is_torch_tpu_available():
    import torch_xla.core.xla_model as xm
    import torch_xla.debug.metrics as met

def fix_buggy_characters(str):
    return re.sub("[{}^\\\\`\u2047<]", " ", str)
def replace_punctuation(str):
    return str.replace("\"", "").replace("'", "")
def score_string_similarity(str1, str2):
    if str1 == str2:
        return 3.0   # Better than perfect token match
    str1 = fix_buggy_characters(replace_punctuation(str1))
    str2 = fix_buggy_characters(replace_punctuation(str2))
    if str1 == str2:
        return 2.0
    if " " in str1 or " " in str2:
        str1_split = str1.split(" ")
        str2_split = str2.split(" ")
        overlap = list(set(str1_split) & set(str2_split))
        return len(overlap) / max(len(str1_split), len(str2_split))
    else:
        if str1 == str2:
            return 1.0
        else:
            return 0.0

class QuestionAnsweringTrainer(Seq2SeqTrainer):
    def __init__(self, *args, tokenizer,eval_examples=None,answer_column_name='answers',dataset_name='squad',  **kwargs):
        super().__init__(*args, **kwargs)
        self.eval_examples = eval_examples
        self.answer_column_name = answer_column_name
        self.dataset_name = dataset_name
        self.tokenizer = tokenizer
        if dataset_name=='squad':
            self.post_process_function = self._post_process_squad#post_process_function
        elif dataset_name=='boolq':
            self.post_process_function = self._post_process_boolq
        elif dataset_name== 'narrativeqa':
            self.post_process_function = self._post_process_narrative_qa
        elif dataset_name == 'race':
            self.post_process_function = self._post_process_race
        elif dataset_name=='squad_v2':
            self.post_process_function = self._post_process_squad
    def _post_process_squad(
        self,examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, stage="eval"
    ,version_2_with_negative=False):
        # Decode the predicted tokens.
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)

        # Build a map example to its corresponding features.
        example_id_to_index = {k: i for i, k in enumerate(examples["id"])}
        feature_per_example = {example_id_to_index[feature["example_id"]]: i for i, feature in enumerate(features)}
        predictions = {}
        # Let's loop over all the examples!
        for example_index, example in enumerate(examples):
            # This is the index of the feature associated to the current example.
            feature_index = feature_per_example[example_index]
            predictions[example["id"]] = decoded_preds[feature_index]

        # Format the result to the format the metric expects.
        if version_2_with_negative:
            formatted_predictions = [
                {"id": k, "prediction_text": v, "no_answer_probability": 0.0} for k, v in predictions.items()
            ]
        else:
            formatted_predictions = [{"id": k, "prediction_text": v} for k, v in predictions.items()]

        references = [{"id": ex["id"], "answers": ex['answers']} for ex in examples]
        return EvalPrediction(predictions=formatted_predictions, label_ids=references)

    @classmethod
    def postprocess_text(cls,preds, labels):
        preds = [pred.strip() for pred in preds]
        labels = [label.strip() for label in labels]
        # rougeLSum expects newline after each sentence
        preds = ["\n".join(nltk.sent_tokenize(pred)) for pred in preds]
        labels = ["\n".join(nltk.sent_tokenize(label)) for label in labels]
        return preds, labels

    def _post_process_boolq(self,examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, tokenizer, stage="eval"):
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # preds = [" ".join(pred) for pred in decoded_preds]
        preds = decoded_preds
        outputs = []
        for pred in preds:
            if 'True' in pred:
                outputs.append(1)
            elif 'False' in pred:
                outputs.append(0)
            else:
                outputs.append(0)
        references = [1 if ex['answer'] else 0 for ex in examples]
        assert(len(references)==len(outputs))
        formatted_predictions = []
        return EvalPrediction(predictions=outputs, label_ids=references)

    def _post_process_narrative_qa(self,examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, tokenizer, stage="eval"):
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        # preds = [" ".join(pred) for pred in decoded_preds]
        preds = decoded_preds
        references = [exp['answers'][0]['text'].lower() for exp in examples]
        formatted_predictions = []
        preds, references = self.postprocess_text(preds,references)

        assert(len(preds)==len(references))
        return EvalPrediction(predictions=preds, label_ids=references)

    def _post_process_race(self,examples: datasets.Dataset, features: datasets.Dataset, outputs: EvalLoopOutput, tokenizer, stage="eval"):
        preds = outputs.predictions
        if isinstance(preds, tuple):
            preds = preds[0]
        decoded_preds = self.tokenizer.batch_decode(preds, skip_special_tokens=True)
        references = [{"answer": ex['answer'],'options':ex['options']} for ex in examples]
        assert(len(references)==len(decoded_preds))
        gold_ids , pred_ids = [],[]
        for prediction, reference in zip(decoded_preds, references):
            #             reference = json.loads(reference)
            gold = int(ord(reference['answer'].strip()) - ord('A'))
            options = reference['options']
            prediction = prediction.replace("\n", "").strip()
            options = [opt.strip() for opt in options if len(opt) > 0]
            #             print('options',options,type(options))
            #             print('prediction',prediction)
            #             print('answer',gold)
            scores = [score_string_similarity(opt, prediction) for opt in options]
            max_idx = np.argmax(scores)
            gold_ids.append(gold)
            pred_ids.append(max_idx)
            # selected_ans = chr(ord('A') + max_idx)
        # print(len(references),len(decoded_preds))

        return EvalPrediction(predictions=pred_ids,label_ids = gold_ids)

    def evaluate(self, eval_dataset=None, eval_examples=None, tokenizer=None,ignore_keys=None, metric_key_prefix: str = "eval",
                 max_length: Optional[int] = None,num_beams: Optional[int] = None):
        self._memory_tracker.start()
        self._max_length = max_length if max_length is not None else self.args.generation_max_length
        self._num_beams = num_beams if num_beams is not None else self.args.generation_num_beams
        eval_dataset = self.eval_dataset if eval_dataset is None else eval_dataset
        eval_dataloader = self.get_eval_dataloader(eval_dataset)
        eval_examples = self.eval_examples if eval_examples is None else eval_examples

        # Temporarily disable metric computation, we will do it in the loop here.
        compute_metrics = self.compute_metrics
        self.compute_metrics = None
        eval_loop = self.prediction_loop if self.args.use_legacy_prediction_loop else self.evaluation_loop
        try:
            output = eval_loop(
                eval_dataloader,
                description="Evaluation",
                # No point gathering the predictions if there are no metrics, otherwise we defer to
                # self.args.prediction_loss_only
                prediction_loss_only=True if compute_metrics is None else None,
                ignore_keys=ignore_keys,
            )
        finally:
            self.compute_metrics = compute_metrics

        if self.post_process_function is not None and self.compute_metrics is not None:
            eval_preds = self.post_process_function(eval_examples, eval_dataset, output,tokenizer)
            # if self.dataset_name=='boolq':
            #     eval_preds = self._post_process_boolq(eval_examples, eval_dataset, output.predictions,tokenizer)
            # else:
            #     eval_preds = self._post_process_squad(eval_examples, eval_dataset, output.predictions, tokenizer)
            metrics = self.compute_metrics(eval_preds)
            if self.dataset_name=='narrativeqa':
                metrics = {key: value.mid.fmeasure * 100 for key, value in metrics.items()}
                metrics = {k: round(v, 4) for k, v in metrics.items()}
            # Prefix all keys with metric_key_prefix + '_'
            for key in list(metrics.keys()):
                if not key.startswith(f"{metric_key_prefix}_"):
                    metrics[f"{metric_key_prefix}_{key}"] = metrics.pop(key)

            self.log(metrics)
        else:
            metrics = {}

        self.control = self.callback_handler.on_evaluate(self.args, self.state, self.control, metrics)
        self._memory_tracker.stop_and_update_metrics(metrics)
        return metrics


