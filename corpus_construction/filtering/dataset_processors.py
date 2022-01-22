import sys
from typing import List, Optional, Tuple
import sys,re,copy,random
sys.path.append('../../data_process')
from QAInput import StructuralQAInput as QAInput
def clear_context(context):
    filtered_context = re.sub(r"[^a-zA-Z0-9\ ()\.,\!\?\-:;\'/]", "", context)
    return filtered_context
def preprocess_sqaud_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples[question_column]
    contexts = examples[context_column]
    answers = examples[answer_column]

    def generate_input(_question, _context):
        return " ".join(["question:", _question, "context:", _context])

    inputs = [QAInput.qg_input_extractive_qa(context, question) for question, context in zip(questions, contexts)]
    targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    return inputs, targets

def preprocess_sqaud_batch_new(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples['question_new']
    contexts = examples['context_new']
    answers = examples['answers_new']

    def generate_input(_question, _context):
        return " ".join(["question:", _question, "context:", _context])

    inputs = [QAInput.qg_input_extractive_qa(context, question) for question, context in zip(questions, contexts)]
    targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    return inputs, targets

def preprocess_boolq_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    question_column, context_column, answer_column = 'passage','question','answer'
    questions = examples[question_column]
    contexts = examples[context_column]
    answers = examples[answer_column]
    inputs = [QAInput.qg_input_boolqa(context,question) for question, context in zip(questions, contexts)]
    targets = [str(ans) for ans in answers]#[answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    # print(inputs,targets)
    return inputs, targets

def preprocess_boolq_batch_pretrain(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples[question_column]
    contexts = examples[context_column]
    answers = examples[answer_column]
    inputs = [QAInput.qg_input_boolqa(clear_context(context),question) for question, context in zip(questions, contexts)]
    targets = [answer["text"][0].capitalize() if len(answer["text"]) > 0 else "" for answer in answers]
    # print(inputs,targets)
    return inputs, targets
def preprocess_multirc_batch_pretrain(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
)-> Tuple[List[str], List[str]]:
    questions = examples['question']
    contexts = examples['context']
    # answers = [exp.strip() for exp in examples['answer']]
    answers = [ans['text'][0].strip() for ans in examples['answers']]
    all_options = examples['options']
    options_texts = []
    for options in all_options:
        options_copy = copy.deepcopy([opt.strip() for opt in options])
        random.shuffle(options_copy)
        options_texts.append(
            f'options: A. {options_copy[0]}; B. {options_copy[1]}; C. {options_copy[2]}; D. {options_copy[3]}')
    # options_texts = [f'options: A.{options[0]}; B. {options[1]}; C. {options[2]}; D. {options[3]}' for options in
    #                  all_options]
    inputs = [QAInput.qg_input_multirc(context=context,question=question,options=option) for question,context,option in zip(questions,contexts,options_texts)]
    targets = answers
    return inputs,targets

def preprocess_multirc_batch_new(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
)-> Tuple[List[str], List[str]]:
    questions = examples['question_new']
    contexts = examples['context_new']
    # answers = [exp.strip() for exp in examples['answer']]
    answers = [ans['text'][0].strip() for ans in examples['answers_new']]
    all_options = examples['options_new']
    options_texts = []
    for options in all_options:
        options_copy = copy.deepcopy([opt.strip() for opt in options])
        random.shuffle(options_copy)
        options_texts.append(
            f'options: A. {options_copy[0]}; B. {options_copy[1]}; C. {options_copy[2]}; D. {options_copy[3]}')
    # options_texts = [f'options: A.{options[0]}; B. {options[1]}; C. {options[2]}; D. {options[3]}' for options in
    #                  all_options]
    inputs = [QAInput.qg_input_multirc(context=context,question=question,options=option) for question,context,option in zip(questions,contexts,options_texts)]
    targets = answers
    return inputs,targets

def preprocess_narrativeqa_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    contexts = [exp['summary']['text'] for exp in examples['document']]
    questions = [exp['text'] for exp in examples['question']]
    answers = [ans[0]['text'] for ans in examples['answers']]
    inputs = [QAInput.qg_input_abstrativeqa(context=context, question=question) for question, context in zip(questions, contexts)]
    targets = answers#[answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    return inputs, targets

def preprocess_narrativeqa_batch_pretrain(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples['question']
    contexts = examples['context']
    answers = examples['answers']
    inputs = [QAInput.qg_input_abstrativeqa(context, question) for question, context in zip(questions, contexts)]
    targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    return inputs, targets

def preprocess_narrativeqa_batch_pretrain_new(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    questions = examples['question_new']
    contexts = examples['context_new']
    answers = examples['answers_new']
    inputs = [QAInput.qg_input_abstrativeqa(context, question) for question, context in zip(questions, contexts)]
    targets = [answer["text"][0] if len(answer["text"]) > 0 else "" for answer in answers]
    return inputs, targets
def preprocess_race_batch(
        examples,
        question_column: str,
        context_column: str,
        answer_column: str,
) -> Tuple[List[str], List[str]]:
    contexts = examples['article']
    questions = examples['question']
    all_options = examples['options']
    answers = examples['answer']
    options_texts = [f'options: A. {options[0]}; B. {options[1]}; C. {options[2]}; D. {options[3]}' for options in all_options]
    inputs = [QAInput.qg_input_multirc(context, question,ops) for question, context,ops in zip(questions, contexts,options_texts)]
    ans_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    targets = [options[ans_map[answer]] for options,answer in zip(all_options,answers)]
    return inputs, targets
import copy
import numpy as np
import collections
def transfer_race_to_pretrain_data(example,task_type):
    ori_keys = example.keys()
    context = copy.deepcopy(example['article'])
    question = copy.deepcopy(example['question'])
    options = copy.deepcopy(example['options'])
    answer = copy.deepcopy(example['answer'])
    ans_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
    answer_text = options[ans_map[answer]]
    # for key in ori_keys:
    #     del example[key]
    output = collections.OrderedDict(copy.deepcopy({'task_type':task_type,'context_new':context,'question_new':question,'options_new':options,'answers_new':{'text':[answer_text],'answer_start':[np.int32(0)]}}))
    return output#{'task_type':task_type,'context':context,'question':question,'options':options,'answers':{'text':[answer_text],'answer_start':[np.int32(0)]}}

def transfer_narrativeqa_to_pretrain_data(example,task_type):
    ori_keys = example.keys()
    contexts = copy.deepcopy(example['document']['summary']['text'])
    questions = copy.deepcopy(example['question']['text'])
    answers = copy.deepcopy([ans['text'] for ans in example['answers']])
    # for key in ori_keys:
    #     del example[key]
    output = collections.OrderedDict(copy.deepcopy({'task_type':task_type,'context_new':contexts,'question_new':questions,'options_new':['A'],'answers_new':{'text':answers,'answer_start':[np.int32(0)]*len(answers)}}))
    return output
def transfer_squad_to_pretrain_data(example,task_type):
    contexts = example['context']
    questions = example['question']
    answers = example['answers']
    # example['answers']['answer_start'] = [np.int64(x) for x in example['answers']['answer_start']]
    # del example['id']
    # del example['title']
    output = collections.OrderedDict(copy.deepcopy({'task_type':task_type,'context_new':contexts,'question_new':questions,'options_new':['A'],'answers_new':{'text':answers['text'],'answer_start':[np.int32(0)]*len(answers['text'])}}))
    return output