from datasets import load_dataset
from tqdm import tqdm
import os
import json
import sys
import nltk
sys.path.append('../../data_process')
from QGInput import QGInput

QGInput = QGInput()
from fuzzywuzzy import fuzz
def general_dataset_reader(name,example_processor_fn,qatype_processor_fn,prefix=None):
    if name=='race':
        dataset = load_dataset(name,'all')
    else:
        dataset = load_dataset(name)
    if prefix:
        name = name+prefix
    if not os.path.exists(os.path.join(basic_dir, name)):
        os.mkdir(os.path.join(basic_dir, name))
    # split = ['train', 'validation', 'test']
    split = dataset.keys()
    counts = open(os.path.join(basic_dir,f"{name}/counts.json"), "a+")

    for sid, split in enumerate(split):
        fout = open(os.path.join(basic_dir, f'{name}/{split}.jsonl'), 'w+')
        count = 0
        outputs = []
        for example in dataset[split]:
            context,question,answer,options = example_processor_fn(example)
            # print(context,question,answer)
            score = fuzz.partial_ratio(context,question+' '+answer[0])
            cuts = ['following is true','following is not true']
            if any([cut in question.lower() for cut in cuts]):
                score = 0
            if context:
                examples = qatype_processor_fn(context,question,answer,options)
                for exp in examples:
                    exp.update({'psg_qa_score':score})
                    outputs.append(exp)
        # print(outputs)
        outputs = sorted(outputs,key=lambda k:k['psg_qa_score'],reverse=True)
        output_count = len(outputs)
        for exp in outputs[:int(output_count*0.7)]:
            fout.write(json.dumps(exp)+'\n')
            count+=1
        counts.write(f'{split}: {count}\n')

#extractive
def squad():
    name = 'squad'
    def example_processor(example):
        context = example['context']
        question = example['question']
        answers = example['answers']['text']#[ans['text'] for ans in example['answers']['text']]
        return context,question,answers,None
    qatype_processor_fn = QGInput.qg_input_extractive_qa
    example_processor_fn = example_processor
    general_dataset_reader(name,example_processor_fn,qatype_processor_fn)

def squad_qapairs():
    name = 'squad'
    def example_processor(example):
        context = example['context']
        question = example['question']
        answers = example['answers']['text']#[ans['text'] for ans in example['answers']['text']]
        return context,question,answers,None
    qatype_processor_fn = QGInput.qg_input_extractive_qapairs
    example_processor_fn = example_processor
    general_dataset_reader(name,example_processor_fn,qatype_processor_fn,prefix='_qapairs')

#abstractive
def narrative_qa_onlyq():
    name = 'narrativeqa'
    def example_processor(example):
        context = example['document']['summary']['text']
        question = example['question']['text']
        answers = [ans['text'] for ans in example['answers']]
        return context,question,answers,None
    qatype_processor_fn = QGInput.qg_input_summaryqa
    example_processor_fn = example_processor
    general_dataset_reader(name,example_processor_fn,qatype_processor_fn)

def narrative_qa_pairs():
    name = 'narrativeqa'
    def example_processor(example):
        context = example['document']['summary']['text']
        question = example['question']['text'].lower().capitalize()
        answers = [ans['text'].lower() for ans in example['answers']]
        return context,question,answers,None
    qatype_processor_fn = QGInput.qg_input_summaryqa_qapairs
    example_processor_fn = example_processor
    general_dataset_reader(name,example_processor_fn,qatype_processor_fn,'_qapairs')
#bool question
def boolq():
    name = 'boolq'
    def example_processor(example):
        context = example['passage']
        question = example['question']
        answers = [example['answer']]
        return context, question, answers, None
    qatype_processor_fn = QGInput.qg_input_boolqa
    example_processor_fn = example_processor
    general_dataset_reader(name,example_processor_fn,qatype_processor_fn)
#multichoice machine reading comprehension
def race():
    name = 'race'
    def example_processor(example):
        context = example['article']
        question = example['question']
        options = example['options']
        answer = example['answer']
        question = f'{question}'
        options_text = f'options: A.{options[0]}; B. {options[1]}; C. {options[2]}; D. {options[3]}'
        ans_map = {'A':0,'B':1,'C':2,'D':3}
        answer_text = [options[ans_map[answer]]]
        return context,question,answer_text,options_text
    qatype_processor_fn = QGInput.qg_input_multirc
    example_processor_fn = example_processor
    general_dataset_reader(name,example_processor_fn,qatype_processor_fn)

def race_negoption():
    name = 'race'

    def example_processor(example):
        context = example['article']
        question = example['question']
        options = example['options']
        answer = example['answer']
        question = f'{question}'
        options_text = options#f'options: A.{options[0]}; B. {options[1]}; C. {options[2]}; D. {options[3]}'
        ans_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        answer_text = [options[ans_map[answer]]]
        return context, question, answer_text, options_text
    qatype_processor_fn = QGInput.qg_intput_multirc_negoption
    example_processor_fn = example_processor
    general_dataset_reader(name, example_processor_fn, qatype_processor_fn,'_negoption')

def race_qapairs():
    name = 'race'

    def example_processor(example):
        context = example['article']
        question = example['question']
        options = example['options']
        answer = example['answer']
        question = f'{question}'
        options_text = options  # f'options: A.{options[0]}; B. {options[1]}; C. {options[2]}; D. {options[3]}'
        ans_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        answer_text = [options[ans_map[answer]]]
        return context, question, answer_text, options_text
    qatype_processor_fn = QGInput.qg_intput_multirc_qapairs
    example_processor_fn = example_processor
    general_dataset_reader(name, example_processor_fn, qatype_processor_fn, '_qapairs')

def race_qapairs_filter():
    name = 'race'
    def example_processor(example):
        context = example['article']
        question = example['question']
        options = example['options']
        answer = example['answer']
        question = f'{question}'
        options_text = options  # f'options: A.{options[0]}; B. {options[1]}; C. {options[2]}; D. {options[3]}'
        ans_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        answer_text = [options[ans_map[answer]]]
        return context, question, answer_text, options_text
    qatype_processor_fn = QGInput.qg_intput_multirc_qapairs
    example_processor_fn = example_processor
    general_dataset_reader(name, example_processor_fn, qatype_processor_fn, '_qapairs')
def race_qfirst():
    name = 'race'
    def example_processor(example):
        context = example['article']
        question = example['question']
        options = example['options']
        answer = example['answer']
        question = f'{question}'
        options_text = options#f'options: A.{options[0]}; B. {options[1]}; C. {options[2]}; D. {options[3]}'
        ans_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        answer_text = [options[ans_map[answer]]]
        return context, question, answer_text, options_text
    qatype_processor_fn = QGInput.qg_intput_multirc_qfirst
    example_processor_fn = example_processor
    general_dataset_reader(name, example_processor_fn, qatype_processor_fn,'_qfirst')

def race_negoption_withq():
    name = 'race'
    def example_processor(example):
        context = example['article']
        question = example['question']
        options = example['options']
        answer = example['answer']
        question = f'{question}'
        options_text = options#f'options: A.{options[0]}; B. {options[1]}; C. {options[2]}; D. {options[3]}'
        ans_map = {'A': 0, 'B': 1, 'C': 2, 'D': 3}
        answer_text = [options[ans_map[answer]]]
        return context, question, answer_text, options_text
    qatype_processor_fn = QGInput.qg_intput_multirc_negoption_withq
    example_processor_fn = example_processor
    general_dataset_reader(name, example_processor_fn, qatype_processor_fn,'_negopt_withq')
if __name__=='__main__':
    basic_dir = '../../../qg_data/'
    narrative_qa_pairs()
    # race_qfirst()
    # race_qapairs_filter()
    race_negoption()
    # narrative_qa()
    # squad()
    squad_qapairs()
    boolq()
    # race()

