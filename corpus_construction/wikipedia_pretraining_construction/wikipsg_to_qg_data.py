import json
import copy
from tqdm import tqdm
import nltk
from nltk.corpus import stopwords
import sys
from multiprocessing import Pool, cpu_count
from functools import partial
sys.path.append('/home/t-wzhong/v-wanzho/promptQA/code/data_process')
from QGInput import QGInput
stopwords = stopwords.words('english')
def clean_keyword(kws):
    def clean_word(word):
        delete = ['the ']#,',','"','.',')','(']
        replace = ['(',')','[',']','"',';','\'','-']
        for item in delete:
            word = word.replace(item,'')
        for item in replace:
            word = word.replace(item,' '+item+' ')
            # word = word.replace(item+' ', item)
        # word = word.replace(',',' ,').replace('.',' .')
        return word.strip()

    all_keyword = kws
    keep = []
    for kw1 in all_keyword:
        flag = True
        for kw2 in all_keyword:
            if kw1!=kw2 and (kw2 in kw1):
                flag=False
        if flag and kw1.lower() not in stopwords:
            keep.append(clean_word(kw1))
    return keep

def extract_qg_data_bool(wiki_psg_kws):
    context = wiki_psg_kws['context']
    outputs, used = [], []
    for answer in ['true','false']:
        source_text = f'The context: {context}. The answer is: <hl> {answer} <hl>. This is a bool task, generate question: '
        outputs.append({"answer": answer, "source_text": source_text})
    return outputs



def extract_qg_data_abstractive_qapris(line):
    wiki_psg_kws = {'context': line[1], 'title': line[0]}
    context = wiki_psg_kws['context']
    # f'generate question: The context: {context}. The answer is: <hl> {cleaned_title} <hl>. Task: abstractive.'
    if not context:
        return None
    outputs = QGInput.qg_input_summaryqa_qapairs(context=context, question='',answers=[''])  # [{'answer':cleaned_title,'source_text':source_text}]
    return outputs

import re

def extract_qg_data_all_multirc_qapris(line):
    wiki_psg_kws = {'context': line[1], 'title': line[0]}
    context = wiki_psg_kws['context']
    # f'generate question: The context: {context}. The answer is: <hl> {cleaned_title} <hl>. Task: abstractive.'
    if not context:
        return None
    filtered_context = re.sub(r"[^a-zA-Z0-9\ ()\.,\!\?\-:;\'/]","", context)
    words = nltk.word_tokenize(filtered_context)
    #filter shorter passages or cutoff extremely long passages
    if len(words)<150:
        return None
    elif len(words)>450:
        filtered_context = ' '.join(words[:450])
    outputs = QGInput.qg_intput_multirc_qapairs(context=filtered_context, question='',answers=[''])  # [{'answer':cleaned_title,'source_text':source_text}]
    return outputs



def extract_qg_data_race_negopt_qapris(line):
    context = line['source_text'].strip('Generate question and answer: [Context] ')
    if not context:
        return None
    preds = line['predictions']
    outputs = []
    #filter out question that most commonly exist
    drop = ['following is true','following is not true']
    for qa in preds:
        if '[question]' in qa and '[answer]' in qa and not any([item in qa.lower() for item in drop]):
            try:
                question, answer = qa.split('[answer]')
                question = question.strip('[question] ').strip()
                answer = answer.strip('.').strip()
                outputs.extend(QGInput.qg_intput_multirc_negoption_withq(context=context, question=question, answers=[answer], options=['']))
            except Exception as e:
                print(qa)
    # f'generate question: The context: {context}. The answer is: <hl> {cleaned_title} <hl>. Task: abstractive.'
 # [{'answer':cleaned_title,'source_text':source_text}]
    return outputs
if __name__ == '__main__':
    type = sys.argv[1]
    basic_dir = sys.argv[2]#'/home/t-wzhong/v-wanzho/promptQA'
    if type=='bool':
        file_path = os.path.join(basic_dir,'wikipedia_data/wiki_raw_data/wiki_psg_keywords_100w.jsonl')
        outf_path = os.path.join(basic_dir,'wikipedia_data/qg_inference_data/bool/qg_inference_bool.jsonl')
        data = open(file_path, 'r', encoding='utf8').readlines()
        count = 0
        fail_count = 0
        with open(outf_path, 'w', encoding='utf8') as outf:
            for line in tqdm(data):
                outputs = extract_qg_data_bool(json.loads(line))
                for output in outputs:
                    outf.write(json.dumps(output) + '\n')
                    count += 1
        print(f'Total count {count}')

    elif type=='abstractive_qapairs':
        file_path = os.path.join(basic_dir, 'wikipedia_data/wiki_raw_data/all_passages.jsonl')
        outf_path = os.path.join(basic_dir,'wikipedia_data/qg_inference_data/abstractive/qg_inference_abstractive_qapairs.jsonl')
        data = json.load(open(file_path, 'r', encoding='utf8'))
        count = 0
        with open(outf_path, 'w', encoding='utf8') as outf:
            for line in tqdm(data.items()):
                line = {'context':line[1],'title':line[0]}
                outputs = extract_qg_data_abstractive_qapris(line)
                if outputs:
                    for output in outputs:
                        outf.write(json.dumps(output) + '\n')
                        count += 1
        print(f'Total count {count}')

    elif type=='multirc_qapair_negopt':
        file_path = sys.argv[3]
        # file_path = os.path.join(basic_dir,'wikipedia_data/qg_inference_data/multirc_qapairs/result_wiki600w_qg_inference_multirc_qapairs_clean.jsonl')
        outf_path = os.path.join(basic_dir,'wikipedia_data/qg_inference_data/multirc_qapairs/qg_inference_multirc_qapairs_negopt.jsonl')
        data = open(file_path, 'r', encoding='utf8').readlines()
        count = 0
        all_outputs = []
        with open(outf_path, 'w', encoding='utf8') as outf:
            for line in tqdm(data):
                outputs = extract_qg_data_race_negopt_qapris(json.loads(line))
                if outputs:
                    all_outputs.extend(outputs)
            all_outputs = sorted(all_outputs, key=lambda k: len(k['source_text'].split(' ')), reverse=True)
            for output in all_outputs:
                if len(output['source_text'].split(' ')) > 500:
                    continue
                outf.write(json.dumps(output) + '\n')
                count += 1
        print(f'Total count {count}')
    elif type=='multirc_all_qapairs':
        # file_path = '/home/t-wzhong/v-wanzho/promptQA/wikipedia_data/keywords/wiki_psg_keywords_100w.jsonl'
        file_path = os.path.join(basic_dir, 'wikipedia_data/wiki_raw_data/all_passages.jsonl')
        outf_path = os.path.join(basic_dir,'wikipedia_data/qg_inference_data/multirc_qapairs/wiki600w_qg_inference_multirc_qapairs.jsonl')
        data = json.load(open(file_path, 'r', encoding='utf8'))
        count = 0
        running_function = extract_qg_data_all_multirc_qapris
        threads = 16
        threads = min(threads, cpu_count())
        source = list(data.items())
        with Pool(threads) as p:
            func_ = partial(running_function)
            all_results = list(tqdm(p.imap(func_, source, chunksize=32), total=len(source),
                                    desc="convert wikipsg", ))
        with open(outf_path, 'w', encoding='utf8') as outf:
            for result in all_results:
                if result:
                    for output in result:
                        outf.write(json.dumps(output) + '\n')
                        count+=1
        print(f'Total count {count}')


