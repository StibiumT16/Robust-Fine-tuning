import json, argparse, random
import numpy as np
from tqdm import tqdm

qa_system_prompt = (
    "Answer the question based on the given document."
    "Only give me the answer and do not output any other words."
    "\nThe following are given documents.\n\n{reference}"
)
qa_user_prompt = "Question: {question}"


detect_system_prompt = (
    "Determine whether the following documents help answer the given question."
    "The assessment includes:\n"
    "### Assessment 1: The document helps answer the question."
    "### Assessment 2: The document is possibly relevant but does not help answer the question."
    "### Assessment 3: The document is irrelevant and does not help answer the question."
    "### Assessment 4: The document contains incorrect information and does not help answer the question."
    "\nOnly give me your assessment for each document and do not output any other words."
)
detect_user_prompt = "Documents:\n{reference}\n\nQuestion: {question}"


def format_reference(retrieval_result):
    output = ""
    for idx, content in enumerate(retrieval_result):
        title = content.split("\n")[0]
        text = "\n".join(content.split("\n")[1:])
        output += f"Doc {idx+1}(Title: {title}) {text}\n"

    return output

def format_detect(flags):
    output = ""
                
    for i, flg in enumerate(flags):
        output += f'Doc {i + 1}: {flg}. '
        if flg == "yes":
            output += f'Doc {i+1} helps answer the question.'
        elif flg == 'neg':
            output += f'Doc {i+1} is possibly relevant but does not help answer the question.'
        elif flg == 'nsy':
            output += f'Doc {i+1} is irrelevant and does not help answer the question.'
        elif flg == 'cf':
            output += f'Doc {i+1} contains incorrect information and does not help answer the question.'
        else: 
            raise NotImplementedError
        
        output += '\n'
    
    return output

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--input_path", type=str, default='../data/e5/train/')
    parser.add_argument("--output_file", type=str, default='data/rbft.json')
    parser.add_argument("--position", type=str, default='random', choices=['random', 'top', 'bottom'])
    parser.add_argument("--topk", type=int, default=5)
    args = parser.parse_args()
    
    random.seed(args.seed)
    np.random.seed(args.seed)
    
    q_list, a_list = [], []
    with open(f'{args.input_path}/sample.json') as fr:
        for line in fr:
            line = json.loads(line)
            query, answer= line['query'], line['answer']
            q_list.append(query)
            a_list.append(answer)
            
    pos_list, attack_list_dict = [], {'cf':[], 'neg':[], 'nsy':[]}
    
    with open(f"{args.input_path}/posp.json") as fr:
        for line in fr:
            line = json.loads(line)
            pos_list.append(line['pos_psgs'])
    
    with open(f"{args.input_path}/negp.json") as fr:
        for line in fr:
            line = json.loads(line)
            attack_list_dict['neg'].append(line['neg_psgs'])
    
    with open(f"{args.input_path}/nsyp.json") as fr:
        for line in fr:
            line = json.loads(line)
            attack_list_dict['nsy'].append(line['nsy_psgs'])
    
    with open(f"{args.input_path}/cfp.json") as fr:
        for line in fr:
            line = json.loads(line)
            attack_list_dict['cf'].append(line['cf_psgs'])
    
    train_data = []
    
    for attack in ['neg', 'nsy', 'cf', 'mix']:
        for tau in [0.2, 0.4, 0.6, 0.8, 1.0]:
            for id, (q, a, pos_psgs) in tqdm(enumerate(zip(q_list, a_list, pos_list))):
                psgs, flags = [], []
                
                if args.position == 'random':
                    sample_value = np.random.rand(min(args.topk, len(pos_psgs)))
                    for i, (pos, v) in enumerate(zip(pos_psgs, sample_value)):
                        if v <= tau: 
                            cur_attack = np.random.choice(['neg', 'nsy', 'cf']) if attack == 'mix' else attack
                            psgs.append(attack_list_dict[cur_attack][id][i])
                            flags.append(cur_attack)
                        else: 
                            psgs.append(pos)
                            flags.append("yes")
                            
                            
                elif args.position == 'top':
                    pos_psgs = pos_psgs[:args.topk]
                    attack_top_k = round(tau * len(pos_psgs))
                    for i, pos in enumerate(pos_psgs):
                        if i < attack_top_k: # attack
                            cur_attack = np.random.choice(['neg', 'nsy', 'cf']) if attack == 'mix' else attack
                            psgs.append(attack_list_dict[cur_attack][id][i])
                            flags.append(cur_attack)
                        else: # keep
                            psgs.append(pos)
                            flags.append("yes")
                
                elif args.position == 'bottom':
                    pos_psgs = pos_psgs[:args.topk]
                    keep_top_k = round((1 - tau) * len(pos_psgs))
                    for i, pos in enumerate(pos_psgs):
                        if i < keep_top_k: # keep
                            psgs.append(pos)
                            flags.append("yes")
                        else: # attack 
                            cur_attack = np.random.choice(['neg', 'nsy', 'cf']) if attack == 'mix' else attack
                            psgs.append(attack_list_dict[cur_attack][id][i])
                            flags.append(cur_attack)

                else:
                    raise NotImplementedError
                
                input_params = {"question": q, "reference": format_reference(psgs)}
                qa_data = {
                    "instruction": qa_system_prompt.format(**input_params),
                    "input": qa_user_prompt.format(**input_params),
                    "output": np.random.choice(a)
                }
                
                detect_data = {
                    "instruction": detect_system_prompt.format(**input_params),
                    "input": detect_user_prompt.format(**input_params),
                    "output": format_detect(flags)
                }
                
                train_data += [qa_data, detect_data]
    
    random.shuffle(train_data)
    
    with open(args.output_file, 'w') as fw:
        json.dump(train_data,fw,indent=2)