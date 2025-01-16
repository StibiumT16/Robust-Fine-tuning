model="e5"
gpu_id="0,1,2,3"

python data/retrieve.py \
    --config_file configs/retrieve.yaml \
    --retrieval_method $model \
    --dataset_name nq \
    --data_save_path data/${model}/retrieve_results/ \
    --gpu_id $gpu_id \
    --retrieval_topk 100 \
    --split train

python data/retrieve.py \
    --config_file configs/retrieve.yaml \
    --retrieval_method $model \
    --dataset_name nq \
    --data_save_path data/${model}/retrieve_results/ \
    --gpu_id $gpu_id \
    --retrieval_topk 100 \
    --split test

python data/retrieve.py \
    --config_file configs/retrieve.yaml \
    --retrieval_method $model \
    --dataset_name hotpotqa \
    --data_save_path data/${model}/retrieve_results/ \
    --gpu_id $gpu_id \
    --retrieval_topk 100 \
    --split train

python data/retrieve.py \
    --config_file configs/retrieve.yaml \
    --retrieval_method $model \
    --dataset_name hotpotqa \
    --data_save_path data/${model}/retrieve_results/ \
    --gpu_id $gpu_id \
    --retrieval_topk 100 \
    --split dev

python data/retrieve.py \
    --config_file configs/retrieve.yaml \
    --retrieval_method $model \
    --dataset_name triviaqa \
    --data_save_path data/${model}/retrieve_results/ \
    --gpu_id $gpu_id \
    --retrieval_topk 100 \
    --split train

python data/retrieve.py \
    --config_file configs/retrieve.yaml \
    --retrieval_method $model \
    --dataset_name triviaqa \
    --data_save_path data/${model}/retrieve_results/ \
    --gpu_id $gpu_id \
    --retrieval_topk 100 \
    --split test