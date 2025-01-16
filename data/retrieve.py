import json, argparse, os
from flashrag.config import Config
from flashrag.utils import get_dataset
from flashrag.retriever import DenseRetriever, BM25Retriever

def run(args):
    config_dict = {
                   "retrieval_method" : args.retrieval_method, 
                   "gpu_id": args.gpu_id, 
                   "dataset_name": args.dataset_name,
                   "retrieval_topk": args.retrieval_topk,
                   "split":[args.split],
                }
    config = Config(args.config_file,config_dict)
    
    os.system(f"mkdir -p {args.data_save_path}")
    
    dataset = get_dataset(config)[args.split]
    input_query = dataset.question
    answer = dataset.golden_answers
    if config['retrieval_method'] == 'bm25':
        retriever = BM25Retriever(config)
    else:
        retriever = DenseRetriever(config)
    retrieval_results = retriever.batch_search(input_query)
    print("Finish searching!")
    
    
    if args.split == 'train':        
        data = [(q, [dic['id'] for dic in result], a) for q, result, a in zip(input_query, retrieval_results, answer)]
        with open(args.data_save_path+f"{args.dataset_name}_train.json", 'w') as fw:
            for id, (q, psg, a) in enumerate(data):
                fw.write(json.dumps({"qid" : id, "query":q, "answer":a, "psgs":psg})+"\n")
                
    else: 
        data = [(q, [dic['id'] for dic in result], a) for q, result, a in zip(input_query, retrieval_results, answer)]
        with open(args.data_save_path+f"{args.dataset_name}_eval.json", 'w') as fw:
            for id, (q, psg, a) in enumerate(data):
                fw.write(json.dumps({"qid" : id, "query":q, "answer":a, "psgs":psg})+"\n")
    


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Running exp")
    parser.add_argument("--dataset_name", type=str, default='nq')
    parser.add_argument("--gpu_id", type=str)
    parser.add_argument("--retrieval_method", type=str, default='e5')
    parser.add_argument('--retrieval_topk',type=int,default=100)
    parser.add_argument("--data_save_path", type=str, default='data/e5/retrieve_results/')
    parser.add_argument("--split", type=str, default='train')
    parser.add_argument("--config_file", type=str, default='configs/retrieve.yaml')
    args = parser.parse_args()
    
    run(args)