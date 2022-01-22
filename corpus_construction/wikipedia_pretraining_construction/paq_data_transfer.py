import json
import pandas as pd
from tqdm import tqdm
import random
import sys,os
basic_dir = '../../../'
wiki_file = os.path.join(basic_dir,'PAQ_data/psgs_w100.tsv')
all_wiki_psg = pd.read_csv(wiki_file,sep='\t',header=0,index_col='id')
print('Done reading wikipedia corpus')
source_file = os.path.join(basic_dir,'PAQ_data/PAQ.metadata.jsonl')
output_file = os.path.join(basic_dir,'PAQ_data/preprocessed_paq_pesudo_qa_data_train_all.jsonl')
print('Reading source file')
data = open(source_file,'r',encoding='utf8').readlines()
print('Done reading source file')
random.shuffle(data)
# eval_outf = open('/home/t-wzhong/v-wanzho/promptQA/PAQ_data/preprocessed_paq_pesudo_qa_data_val.jsonl','w',encoding='utf8')
with open(output_file,'w',encoding='utf8') as outf:
    for idx,line in tqdm(enumerate(data[:3000000])):
        line = json.loads(line.strip())
        question = line['question']
        answers = line['answers']
        for aid,ans in enumerate(answers):
            pid = ans['passage_id']
            offset = ans['offset']
            text = ans['text']
            psg = all_wiki_psg.loc[int(pid),'text']
            outf.write(json.dumps({'context': psg, 'question': question, 'answers': {'text': [text], 'answer_start': [offset]}, 'id': f'{idx}-{aid}'})+'\n')




