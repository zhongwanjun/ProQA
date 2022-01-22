import json
import os
import sys
import random
from tqdm import tqdm
sys.path.append('/home/t-wzhong/v-wanzho/promptQA/code/data_process')
from QGInput import QGInput
from fuzzywuzzy import fuzz
if __name__ == '__main__':
    file_path = '/home/t-wzhong/v-wanzho/promptQA/wikipedia_data/qg_inference_data/multirc_negoption/result_wiki100w_data_multirc_negoption_t5-base.jsonl'
    out_file_path = '/home/t-wzhong/v-wanzho/promptQA/wikipedia_data/qg_inference_data/race/wiki100w_data_multirc_t5basenegopt.jsonl'
    inf = open(file_path,'r',encoding='utf8')
    count = 0
    with open(out_file_path,'w',encoding='utf8') as outf:
        for line in tqdm(inf):
            example = json.loads(line)
            source_text = example['source_text'].strip('Context: ').strip(' <hl>. Generate false option: ')
            split_text = source_text.split('. Answer is: <hl> ')
            if len(split_text)<2:
                continue
            context, answer = split_text[0].strip(),split_text[1].strip()
            cleaned_context = context.replace(answer,'')
            scored_preds = [{'text':opt,'score':fuzz.ratio(answer,opt)} for opt in example['predictions']]
            sorted_preds = sorted(scored_preds,key=lambda k:k['score'])
            if len(sorted_preds)<3:
                print(example['predictions'])
                input()
                continue
            options = sorted_preds[:3] + [answer]
            random.shuffle(options)
            options_text = f'options: A.{options[0]}; B. {options[1]}; C. {options[2]}; D. {options[3]}'
            examples = QGInput.qg_input_multirc(cleaned_context,'',[answer],options_text)
            for example in examples:
                outf.write(json.dumps(example)+'\n')
                count+=1
        print(f'Total count: {count}')
