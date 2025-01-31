import json, argparse, datasets
import numpy as np
from prompt import *
from metrics import metric_dict
from flashrag.config import Config
from flashrag.utils import get_generator


def load_data(data_path, args):   
    def load_psgs(item):
        item['psgs'] = positive_psgs[item['qid']][:args.topk]
        
        # Attack
        if args.attack_position == 'random':
            sample_value = np.random.rand(len(item['psgs']))
            for id, v in enumerate(sample_value):
                if v <= args.tau: # random mode 
                    cur_attack = np.random.choice(['neg', 'nsy', 'cf']) \
                        if args.passage_attack == 'mix' \
                            else args.passage_attack
                    item['psgs'][id] = psgs_dict[cur_attack][item['qid']][id]
        
        elif args.attack_position == 'top':
            attack_topk = round(args.tau * len(item['psgs']))
            for id in range(len(item['psgs'])):
                if id < attack_topk: # top mode, only attack top psgs
                    cur_attack = np.random.choice(['neg', 'nsy', 'cf']) \
                        if args.passage_attack == 'mix' \
                            else args.passage_attack
                    item['psgs'][id] = psgs_dict[cur_attack][item['qid']][id]
                else: # neglect bottom psgs
                    break
        
        elif args.attack_position == 'bottom':
            keep_topk = round((1 - args.tau) * len(item['psgs']))
            for id in range(len(item['psgs'])):
                if id < keep_topk: # bottom mode, neglect top psgs
                    continue
                else: # only attack bottom psgs
                    cur_attack = np.random.choice(['neg', 'nsy', 'cf']) \
                        if args.passage_attack == 'mix' \
                            else args.passage_attack
                    item['psgs'][id] = psgs_dict[cur_attack][item['qid']][id]
        
        else:
            raise NotImplementedError
                    
                    
        return item
        
    dataset = datasets.load_dataset(
        'json',
        data_files = data_path + "/sample.json",
    )['train']
    
    
    with open(data_path + "/posp.json") as fr:
        positive_psgs = {}
        for line in fr:
            line = json.loads(line)
            positive_psgs[line['qid']] = line['pos_psgs']
    
    neg_psgs, nsy_psgs, cf_psgs = {}, {}, {}
    if args.passage_attack == 'neg' or args.passage_attack == 'mix':
        with open(data_path + "/negp.json") as fr:
            for line in fr:
                line = json.loads(line)
                neg_psgs[line['qid']] = line['neg_psgs']
                
    if args.passage_attack == 'nsy' or args.passage_attack == 'mix':
        with open(data_path + "/nsyp.json") as fr:
            for line in fr:
                line = json.loads(line)
                nsy_psgs[line['qid']] = line['nsy_psgs']
                
    if args.passage_attack == 'cf' or args.passage_attack == 'mix':
        with open(data_path + "/cfp.json") as fr:
            for line in fr:
                line = json.loads(line)
                cf_psgs[line['qid']] = line['cf_psgs']
    
    psgs_dict = {
        "neg" : neg_psgs,
        "nsy" : nsy_psgs,
        "cf" : cf_psgs
    }
    
    dataset = dataset.map(load_psgs)
    
    return dataset


def generate(generator, prompt_template, dataset, args):
    input_prompts = [
        prompt_template.get_string(
            question=query, \
            retrieval_result=psgs
        ) for query, psgs in zip(dataset['query'], dataset['psgs'])
    ] 
    preds = generator.generate(input_prompts)
    
    return preds
    

def eval(args):
    config_dict = {"save_note": "eval",
                   "gpu_id": args.gpu_id, 
                }
    config = Config(args.config_file, config_dict)
    np.random.seed(config['seed'])
    dataset = load_data(args.data_path, args)
    scorers = [metric_dict[metric](config) for metric in config['metrics']]
    
    all_pairs = []
    for query, psgs in zip(dataset['query'], dataset['psgs']):
        all_pairs.extend([[query, psg] for psg in psgs])
        
    generator = get_generator(config)
    
    if args.prompt_template == 'norag':
        prompt_template = NoRAGPromptTemplate(config)
    else:
        prompt_template = NaivePromptTemplate(config)


    preds = generate(generator, prompt_template, dataset, args)
    eval_results = [scorer.calculate_metric(preds, dataset['answer'])[0] for scorer in scorers]
    print(eval_results)
    
    with open(args.output_file, 'w') as fw:
        fw.write(json.dumps({'result':eval_results})+"\n")
        for i, (q, a, p) in enumerate(zip(dataset['query'], dataset['answer'], preds)):
            fw.write(json.dumps({i:{'query':q, 'answer':a, 'pred':p}})+'\n')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--data_path", type=str, default='data/e5/test_data')
    parser.add_argument("--gpu_id", type=str)
    parser.add_argument("--topk", type=int, default=5)
    parser.add_argument("--tau", type=float, default=0.)
    parser.add_argument("--config_file", type=str, default='configs/eval_config.yaml')
    parser.add_argument("--output_file", type=str, default='output/output.txt')
    parser.add_argument("--attack_position", type=str, default="random", choices=['random', 'top', 'bottom'])
    parser.add_argument("--passage_attack", type=str, default=None)
    parser.add_argument("--prompt_template", type=str, default=None)
    args = parser.parse_args()
    
    assert args.tau >= 0. and args.tau <= 1.
    
    eval(args)