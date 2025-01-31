import json, os, argparse
import numpy as np
from tqdm import tqdm

def sample_train(args):
    dataset_files = [f'{args.data_path}/hotpotqa_train.json',
                f'{args.data_path}/nq_train.json',
                f'{args.data_path}/triviaqa_train.json']

    os.system(f"mkdir -p {args.output_path}")
    
    datasets = []
    for file in dataset_files:
        with open(file) as fr:    
            for line in tqdm(fr):
                line = json.loads(line)
                query, pos, neg, answer = line['query'], line['psgs'][:args.k], np.random.choice(line['psgs'][50:], args.k, replace=False).tolist(), line['answer']
                #if "yes" in answer or 'no' in answer: continue
                which_dataset = file.split('/')[-1].split('_')[0]
                datasets.append((query, pos, neg, answer, which_dataset))
        
    samples = np.random.choice(len(datasets), args.sample_count, replace=False) # Train: sample args.sample_count tuples on all datasets

    with open(f'{args.output_path}/sample.json', 'w') as fw:
        for qid, sample_id in enumerate(samples):
            data = datasets[sample_id]
            fw.write(json.dumps({'qid' : qid, 'query' : data[0], 'answer':data[3], 'from':data[4]}) + '\n')

    corpus = {}
    with open(args.corpus_file) as fr:
        for line in tqdm(fr):
            line = json.loads(line)
            corpus[line['id']] = line['contents']

    with open(f'{args.output_path}/posp.json', 'w') as fw_pos, \
        open(f'{args.output_path}/negp.json', 'w') as fw_neg, \
        open(f'{args.output_path}/nsyp.json', 'w') as fw_nsy:
        
        for qid, sample_id in tqdm(enumerate(samples)):
            data = datasets[sample_id]
            
            pos_psgs = [corpus[psg_id] for psg_id in data[1]]
            neg_psgs = [corpus[psg_id] for psg_id in data[2]]
            nsy_psgs = [corpus[str(psg_id)] for psg_id in np.random.choice(args.corpus_count, args.k, replace=False)]
            
            fw_pos.write(json.dumps({'qid' : qid, 'pos_psgs' : pos_psgs}) + '\n')
            fw_neg.write(json.dumps({'qid' : qid, 'neg_psgs' : neg_psgs}) + '\n')
            fw_nsy.write(json.dumps({'qid' : qid, 'nsy_psgs' : nsy_psgs}) + '\n')

def sample_test(args):
    dataset_files = [f'{args.data_path}/hotpotqa_eval.json',
                f'{args.data_path}/nq_eval.json',
                f'{args.data_path}/triviaqa_eval.json']

    os.system(f"mkdir -p {args.output_path}")

    samples = []
    for file in dataset_files:
        dataset, dataset_count = [], 0
        with open(file) as fr:    
            for line in tqdm(fr):
                dataset_count += 1
                line = json.loads(line)
                query, pos, neg, answer = line['query'], line['psgs'][:args.k], np.random.choice(line['psgs'][50:], args.k, replace=False).tolist(), line['answer']
                if "yes" in answer or 'no' in answer: continue
                which_dataset = file.split('/')[-1].split('_')[0]
                dataset.append((query, pos, neg, answer, which_dataset))
                
        
        sample_ids = np.random.choice(len(dataset), args.sample_count, replace=False) # Test: sample args.sample_count tuples on each dataset
        samples.extend(dataset[id] for id in sample_ids)
        
    with open(f'{args.output_path}/sample.json', 'w') as fw:
        for qid, data in enumerate(samples):
            fw.write(json.dumps({'qid' : qid, 'query' : data[0], 'answer':data[3], 'from':data[4]}) + '\n')

    corpus = {}
    with open(args.corpus_file) as fr:
        for line in tqdm(fr):
            line = json.loads(line)
            corpus[line['id']] = line['contents']

    with open(f'{args.output_path}/posp.json', 'w') as fw_pos, \
        open(f'{args.output_path}/negp.json', 'w') as fw_neg, \
        open(f'{args.output_path}/nsyp.json', 'w') as fw_nsy:
        
        for qid, data in tqdm(enumerate(samples)):
            
            pos_psgs = [corpus[psg_id] for psg_id in data[1]]
            neg_psgs = [corpus[psg_id] for psg_id in data[2]]
            nsy_psgs = [corpus[str(psg_id)] for psg_id in np.random.choice(args.corpus_count, args.k, replace=False)]
            
            fw_pos.write(json.dumps({'qid' : qid, 'pos_psgs' : pos_psgs}) + '\n')
            fw_neg.write(json.dumps({'qid' : qid, 'neg_psgs' : neg_psgs}) + '\n')
            fw_nsy.write(json.dumps({'qid' : qid, 'nsy_psgs' : nsy_psgs}) + '\n')
        
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--mode", type=str, default='train', choices=['train', 'test'])
    parser.add_argument("--data_path", type=str, default='data/e5/retrieve_results/')
    parser.add_argument("--output_path", type=str, default='data/e5/train')
    parser.add_argument("--corpus_file", type=str, default='FlashRAG_datasets/retrieval-corpus/wiki18_100w.jsonl')
    parser.add_argument("--k", type=int, default=10)
    parser.add_argument("--corpus_count", type=int, default=21015324)
    parser.add_argument("--sample_count", type=int, default=20000)
    parser.add_argument("--seed", type=int, default=0)
    args = parser.parse_args()
    
    np.random.seed(args.seed)
    
    if args.mode == 'train':
        sample_train(args)
    else:
        sample_test(args)