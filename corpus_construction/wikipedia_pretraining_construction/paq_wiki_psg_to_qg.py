import json
import pandas as pd
from tqdm import tqdm
import random
import sys
sys.path.append('../../data_process')
from QGInput import QGInput
def extract_qg_data_abstractive_qapris(context,model_type='t5'):
    outputs = QGInput.qg_input_summaryqa_qapairs(context=context, question='',answers=[''])  # [{'answer':cleaned_title,'source_text':source_text}]
    return outputs
def extract_qg_data_extractive_qapris(context,model_type='t5'):
    outputs = QGInput.qg_input_extractive_qapairs(context=context, question='',answers=[''])  # [{'answer':cleaned_title,'source_text':source_text}]
    return outputs
if __name__=='__main__':
    basic_path = sys.argv[1]
    wiki_file = os.path.join(basic_path,'PAQ_data/psgs_w100.tsv')
    all_wiki_psg = pd.read_csv(wiki_file,sep='\t',header=0,index_col='id')
    psg_score = os.path.join(basic_path,'PAQ_data/PASSAGE_SCORES/passage_scores.tsv')
    psg_scores = pd.read_csv(psg_score,sep='\t',names=['id','score'],index_col='id')
    psg_scores.sort_values(by=['score'],ascending=False)
    count = 0
    outf_path = os.path.join(basic_path,'wikipedia_data/qg_inference_data/extractive/qg_inference_extractive_qapairs.jsonl')
    with open(outf_path,'w',encoding='utf8') as outf:
        for line in tqdm(psg_scores.iterrows()):
            if count==4000000:
                break
            psg_id = line[0]
            if count==0:
                print(line)
            context = all_wiki_psg.loc[int(psg_id),'text']
            outputs = extract_qg_data_extractive_qapris(context)
            if outputs:
                for output in outputs:
                    outf.write(json.dumps(output) + '\n')
                    count += 1
    print(f'Total count {count}')