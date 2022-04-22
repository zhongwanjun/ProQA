import json
import os
import sys
from tqdm import tqdm
import random
import nltk
from statistics import mean
from multiprocessing import Pool, cpu_count
from functools import partial
def merge_example_boolq(source,predq,id):
    source = json.loads(source)
    context = source['source_text'].split('. The answer is: ')[0].strip('The context: ')
    context = context.replace('</s>','')
    question = predq.replace('</s>','').replace('<pad>','').strip()+'?'
    answer = [source['answer'].capitalize()]
    # print(question)
    return {'context':context,'question':question,'answers':{'text':answer,'answer_start':[0]},'id':id}
def merge_example_squad(source,predq,id):
    source = json.loads(source)
    context = source['source_text'].strip('generate question: ')
    context = context.replace(' <hl> ','').replace('</s>','')
    question = predq.replace('</s>','').replace('<pad>','').strip()
    answer = [source['answer']]
    return {'context':context,'question':question,'answers':{'text':answer,'answer_start':[0]},'id':id}
def transfer_example_abstractive(source,preds,id):
    source_split = source['source_text'].split(', the answer is: ')
    context = source_split[0].strip('The context: ')
    answer = source_split[1].strip('This is a summary task, please generate question: ').strip().strip('.')
    question = preds[0]
    return {'context': context, 'question': question, 'answers': {'text': [answer], 'answer_start': [0]}, 'id': id}
def transfer_example_abstractive_qapairs(zip_data):
    source, id = zip_data
    context = source['source_text'].strip('Generate question and the answer: The context: ').strip('.')
    context = re.sub(r"[^a-zA-Z0-9\ ()\.,\!\?\-:;\'/]", "", context)
    qa_pairs = source['predictions']
    outputs = []
    for qid,qa in enumerate(qa_pairs):
        if '[question]' in qa and '[answer]' in qa:
            try:
                question, answer = qa.split('[answer]')[0],qa.split('[answer]')[1]
                question = question.strip('[question]: ')
                answer = answer.strip('.').strip()
                outputs.append({'context': context, 'question': question, 'answers': {'text': [answer], 'answer_start': [0]}, 'id': f'{id}-{qid}'})
            except Exception as e:
                print(qa)
    return outputs
from fuzzywuzzy import fuzz
def transfer_example_extractive_qapairs(zip_data):
    source, id = zip_data
    context = source['source_text'].strip('Generate question and the answer: [Context] ').strip('.')
    context = re.sub(r"[^a-zA-Z0-9\ ()\.,\!\?\-:;\'/]", "", context)
    qa_pairs = source['predictions']
    outputs = []
    for qid,qa in enumerate(qa_pairs):
        if '[question]' in qa and '[answer]' in qa:
            try:
                question, answer = qa.split('[answer]')[0],qa.split('[answer]')[1]
                question = question.strip('[question]').strip()
                answer = answer.strip('.').strip()
                flag = True
                for prev_qa in outputs:
                    if fuzz.partial_ratio(prev_qa['question'],question)>90:
                        flag = False
                        break
                if flag:
                    outputs.append({'context': context, 'question': question, 'answers': {'text': [answer], 'answer_start': [0]}, 'id': f'{id}-{qid}'})
            except Exception as e:
                print(qa)
    return outputs
from fuzzywuzzy import fuzz
import random
def transfer_example_multirc_qapairs_negopt(zip_data):
    try:
        source,id = zip_data
        cqa = source['source_text'].strip('Generate false option: [Context] ').replace('<hl>','').strip('.')
        cqa = cqa.split('. [Question] ')
        c,qa = cqa[0],cqa[1]
        q,a = qa.split('. [Answer]')
        preds = source['predictions']
        pred_scores = [{'text':pred,'score':fuzz.partial_ratio(pred,a)} for pred in preds]
        pred_scores = sorted(pred_scores,key=lambda k:k['score'],reverse=False)
        opt1 = pred_scores[0]['text']
        pred_scores_2 = [{'text': pred, 'score': mean([fuzz.partial_ratio(pred, ref) for ref in [a,opt1]])} for pred in preds if pred!=opt1]
        pred_scores_2 = sorted(pred_scores_2, key=lambda k: k['score'], reverse=False)
        opt2 = pred_scores_2[0]['text']
        pred_scores_3 = [{'text': pred, 'score': mean([fuzz.partial_ratio(pred, ref) for ref in [a, opt1,opt2]])} for pred in
                         preds if pred not in [opt1, opt2]]
        pred_scores_3 = sorted(pred_scores_3, key=lambda k: k['score'], reverse=False)
        opt3 = pred_scores_3[0]['text']
        opts = [opt1.lower(),opt2.lower(),opt3.lower(),a.lower()]
        random.shuffle(opts)
        outputs = [{'context': c, 'question': q, 'answer':a , 'id': f'{id}','options':opts}]
    except Exception as e:
        print(e)
        print(source)
        return []
    return outputs
import copy,re
def transfer_example_multirc_qapairs_negopt_new(zip_data):
    try:
        source,id = zip_data
        cqa = source['source_text'].strip('Generate false option: [Context] ').replace('<hl>','').strip('.')
        cqa = cqa.split('. [Question] ')
        c,qa = cqa[0],cqa[1]
        q,a = qa.split('. [Answer]')[0].strip(),qa.split('. [Answer]')[1].strip()
        c = re.sub(r"[^a-zA-Z0-9\ ()\.,\!\?\-:;\'/]","", c)
        c_words = nltk.word_tokenize(c)
        # print(c_words)
        # print(len(c_words))
        # input()
        if len(c_words)<100:
            return []
        preds = list(set(source['predictions']))
        a_psg_score = fuzz.token_set_ratio(c,a)
        psg_pred_scores = [{'text': pred, 'score': abs(fuzz.partial_ratio(pred, c)-a_psg_score)} for pred in preds]
        psg_pred_scores = sorted(psg_pred_scores, key=lambda k: k['score'], reverse=False)
        ans_pred_scores = [{'text': pred, 'score': fuzz.partial_ratio(pred, a)} for pred in preds]
        ans_pred_scores = sorted(ans_pred_scores, key=lambda k: k['score'], reverse=False)
        opts = [a]
        threshold = max(70,ans_pred_scores[-3]['score'])
        store = []
        for item in psg_pred_scores:
            if len(opts)==4:
                break
            if any([fuzz.partial_ratio(item['text'],opt)>=threshold for opt in opts]):
                store.append(item['text'].lower())
                continue
            else:
                opts.append(item['text'].lower())
        while len(opts)<4:
            opts.extend(store[:4-len(opts)])
        # print({'Context': c, 'Question': q, 'Answer': a})
        # print('---------passage match score--------')
        # print(a_psg_score)
        # print(psg_pred_scores)
        # print('---------answer match score----------')
        # print(ans_pred_scores)
        # print('----------combine match score-----------------')
        # combine_score = [{'text': pred, 'score': fuzz.partial_ratio(pred, a)+1-fuzz.partial_ratio(pred, c)} for pred in preds]
        # combine_score = sorted(combine_score, key=lambda k: k['score'], reverse=False)
        # print(combine_score)
        # print('---------options-------------')
        # print(opts)
        # print('--------------------------------')
        # input()
        # opts = copy.deepcopy([pred['text'].lower() for pred in pred_scores[:3]]+[a.lower()])
        random.shuffle(opts)
        outputs = [{'context': c, 'question': q, 'answers':{'text':[a],'answer_start':[0]} , 'id': f'{id}','options':opts}]
    except Exception as e:
        print(e)
        # print(source)
        return []
    return outputs
if __name__=='__main__':
    type = sys.argv[1]
    basic_dir = '../../../'
    def generate_data(source_file,merge_example_fn,output_file):
        source_data = [json.loads(pred.strip()) for pred in open(os.path.join(basic_dir, source_file), 'r', encoding='utf8').readlines()]
        outf = open(f'{output_file}', 'w', encoding='utf8')
        output_val_file = output_file.replace('training','val')
        val_outf = open(os.path.join(basic_dir, f'{output_val_file}'), 'w', encoding='utf8')
        exsists = set()
        val_split = list(range(len(source_data)))
        random.shuffle(val_split)
        val_split = val_split[:10000]
        count=0
        running_function = merge_example_fn
        zip_data = list(zip(source_data, list(range(len(source_data)))))
        threads = 16
        threads = min(threads, cpu_count())
        with Pool(threads) as p:
            func_ = partial(running_function)
            all_results = list(tqdm(p.imap(func_, zip_data, chunksize=32), total=len(zip_data),
                                    desc="convert prediction to trainable data", ))
        # all_results = []
        # for id,example in enumerate(source_data):
        #     all_results.extend(merge_example_fn(tuple([example,id])))
        instances_number = len(all_results)
        for id, example in tqdm(enumerate(all_results)):
            # pred = json.loads(pred.strip())
            if isinstance(example,list):
                for exp in example:
                    item = exp['context'] + exp['question']
                    if item not in exsists:
                        if id in val_split:
                            val_outf.write(json.dumps(exp) + '\n')
                        else:
                            outf.write(json.dumps(exp) + '\n')
                        exsists.add(item)
            else:
                item = example['context'] + example['question']
                if item not in exsists:
                    count+=1
                    if id in val_split:
                        val_outf.write(json.dumps(example) + '\n')
                    else:
                        outf.write(json.dumps(example) + '\n')
                    exsists.add(item)
                else:
                    continue
        print('Total instances count: {}'.format(count))
    def generate_data_withp(source_file, pred_q_file, prefix,merge_example_fn,output_file):
        source_data = open(os.path.join(basic_dir, source_file), 'r', encoding='utf8').readlines()
        predictions = open(os.path.join(basic_dir, pred_q_file), 'r', encoding='utf8').readlines()
        outf = open(os.path.join(basic_dir, f'{output_file}'), 'w', encoding='utf8')
        exsists = set()
        for id, pred in tqdm(enumerate(predictions)):
            source = source_data[id]
            example = merge_example_fn(source.strip(), pred.strip(), id)

            item = example['context']+example['question']
            if item not in exsists:
                outf.write(json.dumps(example) + '\n')
                exsists.add(item)
            else:
                continue
            # for k,v in example.items():
            #     all_example[k].append(v)
    if type=='extractive_qapairs':
        basic_dir = os.path.join(basic_dir,'wikipedia_data/qg_inference_data/extractive/output/')
        # source_file = 'result_wiki100w_data_abstractive_qaparis_t5-large.jsonl'
        source_file = 'result_paq_wiki400w_data_extractive_qapairs_t5-large.jsonl'
        merge_example_fn = transfer_example_extractive_qapairs
        # output_file = 'wiki100w_abstractive_qapairs_pesudo_training_data.jsonl'
        output_file = os.path.join(basic_dir,'wikipedia_data/pesudo_qa_data/extractive/paq_wiki400w_data_extractive_qapairs_pesudo_training_data.jsonl')
        generate_data(source_file, merge_example_fn, output_file)
    elif type=='bool':
        basic_dir = os.path.join(basic_dir,'qg_data/qg_inference_data/squad_boolq')
        # source_file = 'train.jsonl'
        # pred_q_file = 'squad_boolq_train_questions_t5-base-qg-hl.txt'
        merge_example_fn = merge_example_boolq
        basic_dir = os.path.join(basic_dir,'wikipedia_data/qg_inference_data/bool')
        source_file = 'wiki100w_qg_inference_bool.jsonl'
        pred_q_file = 'test_wiki100w_data_qg_extractive_t5-base-qg-hl_ques.txt'
        output_file = 'wiki100w_bool_pesudo_training_data.jsonl'
        generate_data(source_file, pred_q_file,  merge_example_fn, output_file)
    elif type=='abstractive':
        basic_dir = os.path.join(basic_dir,'wikipedia_data/qg_inference_data/abstractive')
        merge_example_fn = transfer_example_abstractive
        source_file = 'result_wiki100w_data_abstractive_firstsen_t5-large.jsonl'#result_wiki100w_data_{DATA_NAME}_t5-large.jsonl'
        output_file = 'wiki100w_abstractive_pesudo_data_firstsen_train.jsonl'
        generate_data(source_file,merge_example_fn,output_file)
    elif type=='abstractive_qapairs':
        basic_dir = os.path.join(basic_dir,'wikipedia_data/qg_inference_data/abstractive/output/')
        # source_file = 'result_wiki100w_data_abstractive_qaparis_t5-large.jsonl'
        source_file = 'result_paq_wiki400w_data_abstractive_qapairs_t5-large.jsonl'
        merge_example_fn = transfer_example_abstractive_qapairs
        # output_file = 'wiki100w_abstractive_qapairs_pesudo_training_data.jsonl'
        output_file = os.path.join(basic_dir,'wikipedia_data/pesudo_qa_data/abstractive/paq_wiki400w_data_abstractive_qapairs_pesudo_training_data.jsonl')
        generate_data(source_file, merge_example_fn, output_file)
    elif type=='multirc_qapairs_negopt':
        basic_dir = os.path.join(basic_dir,'wikipedia_data/qg_inference_data/multirc_qapairs/')
        # source_file = 'result_wiki100w_data_multirc_qapairs_negopt_t5-large_2.jsonl'
        # source_file = 'result_wiki100w_data_multirc_qapairs_negopt_filter.jsonl'
        source_file = 'result_wiki600w_qg_inference_multirc_qapairs_negopt_clean.jsonl'
        merge_example_fn = transfer_example_multirc_qapairs_negopt_new
        output_file =  os.path.join(basic_dir,'wikipedia_data/pesudo_qa_data/multichoice/wiki600w_multirc_qapair_negopt_pesudo_training_data_filtered_qgmodelv1.jsonl')
        generate_data(source_file,merge_example_fn,output_file)
 
