import json, argparse
from tqdm import tqdm
from flashrag.config import Config
from flashrag.utils import get_generator
from data_prompt import *

def load_data_with_passage(data_path):
    q, psgs, a = [], [], []
    with open(f'{data_path}/sample.json') as fr1, open(f'{data_path}/posp.json') as fr2:
        for line1, line2 in zip(fr1, fr2):
            line1, line2 = json.loads(line1), json.loads(line2)
            assert line1['qid'] == line2['qid']
            q.append(line1['query'])
            a.append(line1['answer'])
            psgs.append(line2['pos_psgs'])
    return q, psgs, a

def load_data_without_passage(data_path):
    q, a = [], []
    with open(f'{data_path}/sample.json') as fr:
        for line in fr:
            line = json.loads(line)
            q.append(line['query'])
            a.append(line['answer'])
    return q, a

def run_cfp(args):
    config_dict = {"save_note": args.type,
                   "gpu_id": args.gpu_id,
                }
    config = Config(args.config_file, config_dict)
    q_prompt_template = WrongAnswerPromptTemplate(config)
    psg_prompt_template = CounterFactualPassagePromptTemplate(config)
    generator = get_generator(config)
    
    q_list, psgs_list, a_list = load_data_with_passage(args.data_path)
    n, k = len(q_list), len(psgs_list[0])
    
    #Wrong Answer Generation
    input_prompts1 = [
        q_prompt_template.get_string(
            query = query,
            passages = psgs[:5],
            answer = answer
        ) for query, psgs, answer in zip(q_list, psgs_list, a_list)
    ]
    
    wrong_answer_list = generator.generate(input_prompts1)
    
    #Counterfactual Passage Generation
    input_prompts2= [] 
    for psgs, answer, wrong_answer in tqdm(zip(psgs_list, a_list, wrong_answer_list)):
        input_prompts2 += [
            psg_prompt_template.get_string(
                passage=psg,
                answer=answer,
                wrong_answer=wrong_answer
            ) for psg in psgs]
    assert len(input_prompts2) == len(q_list) * k
    
    
    preds = generator.generate(input_prompts2)
    preds = [preds[i * k : (i + 1) * k] for i in range(n)]

    with open(f'{args.data_path}/cfp.json', 'w') as fw:
        for qid, (wa, psgs) in enumerate(zip(wrong_answer_list, preds)):
            fw.write(json.dumps({'qid':qid, 'wrong_answer':[wa], 'cf_psgs':psgs}) + '\n') 



if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--data_path", type=str, default='data/e5/train/')
    parser.add_argument("--gpu_id", type=str)
    parser.add_argument("--config_file", type=str, default='configs/data_generation.yaml')
    args = parser.parse_args()
    
    run_cfp(args)