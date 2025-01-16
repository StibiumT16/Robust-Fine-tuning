# Robust Fine-tuning (RbFT)
The code of our paper: *Robust Fine-tuning for Retrieval-Augmented Generation against Retrieval Defects*

### 0. Installation
We implement the training and RAG pipeline based on [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) and [FlashRAG](https://github.com/RUC-NLPIR/FlashRAG) respectively. 

Please install them according to their requirements.

### 1. Data Process
##### 1.1. Download Dataset 
We directly adopt the dataset pro-processed by FlashRAG. 

Download FlashRAG Datasets from huggingface:
```bash
huggingface-cli download --repo-type dataset --local-dir-use-symlinks False --resume-download RUC-NLPIR/FlashRAG_datasets --local-dir FlashRAG_datasets
```

##### 1.2. Build Index
```bash
CUDA_VISIBLE_DEVICES=0,1,2,3 python -m flashrag.retriever.index_builder \
    --retrieval_method e5 \
    --model_path intfloat/e5-base-v2 \
    --corpus_path FlashRAG_datasets/retrieval-corpus/wiki18_100w.jsonl \
    --save_dir data/indexes/ \
    --use_fp16 \
    --max_length 512 \
    --batch_size 256 \
    --pooling_method mean \
    --faiss_type Flat 
```

##### 1.3. Retrieve Documents
First set `data_dir` and `corpus_path` based on your *FlashRAG_datasets* path in `configs/retrieve.yaml`

Then run:
```bash
bash retrieve.sh
```
All retrieved results (train and eval data of all 3 datasets) will be saved at `data/e5/retrieve_results`

##### 1.4. Data Sampling
Sample train/test data and generate noisy/irrelevant retrieval inputs:
```bash
python data/sample.py \
    --mode test \
    --data_path data/e5/retrieve_results/ \
    --output_path data/e5/test/ \
    --corpus_file FlashRAG_datasets/retrieval-corpus/wiki18_100w.jsonl

python data/sample.py \
    --mode train \
    --data_path data/e5/retrieve_results/ \
    --output_path data/e5/train/ \
    --corpus_file FlashRAG_datasets/retrieval-corpus/wiki18_100w.jsonl \
    --sample_count 20000 
```

`sample.json`,`posp.json`,`nsyp.json`,`irrp.json` will be saved at `data/e5/train` and `data/e5/test`

##### 1.5. Generate Counterfactual Retrieval Inputs
```bash
python data/data_gen.py \
    --data_path data/e5/train \
    --config_file configs/data_generation.yaml \
    --gpu_id 0

python data/data_gen.py \
    --data_path data/e5/test \
    --config_file configs/data_generation.yaml \
    --gpu_id 0
```

`cfp.json` will be saved at `data/e5/train` and `data/e5/test`

##### 1.6. Directlty Download Our Pre-processed Data
We also provide our pre-processed data at this [Google Drive Link](https://drive.google.com/drive/folders/1ByebjYy3jRyK2PmGYMpQpxRM5LUXjJ4W?usp=sharing).



### 2. Run Vanilla RAG
First select the generation model in the config file `configs/eval.yaml` by setting `generator_model`to`llama`or`qwen`

Then you can run the following command to generate and evaluate the outputs:
```bash
python eval/generate.py \
    --data_path data/e5/test \
    --config_file configs/eval.yaml \
    --output_file output/vanilla_clean.json \
    --gpu_id 0
```

To simulate retrieval defects, you can set the args:
* passage_attack: `nsy`, `irr`, `cf`, `mix`
* $0 \leq tau \leq 1$


```bash
python eval/generate.py \
    --data_path data/e5/test \
    --config_file configs/eval.yaml \
    --output_file output/vanilla_cf_0.4.json \
    --passage_attack cf \
    --tau 0.4 \ 
    --gpu_id 0
```

### 3. Robust Fine-tuning
First generate the training data:
```bash
cd rbft
python data.py
```

Then you can train RbFT models and merge the LoRA weights through the LLaMA-Factory toolkit:
Llama:
```bash
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2 llamafactory-cli train train_llama.yaml
llamafactory-cli export merge_llama.yaml
```
Qwen:
```bash
FORCE_TORCHRUN=1 CUDA_VISIBLE_DEVICES=0,1,2 llamafactory-cli train train_qwen.yaml
llamafactory-cli export merge_qwen.yaml
```

The merged model will be saved at `rbft/model`

We also provide our LoRA weights at this [Google Drive Link](https://drive.google.com/drive/folders/1ByebjYy3jRyK2PmGYMpQpxRM5LUXjJ4W?usp=sharing). 

You can directly merge them with the original `Llama-3.2-3B-Instruct` and `Qwen2.5-3B-Instruct` model without training.



### 4. Evaluate
Similar to Vanilla RAG, don't forget to select the generator model by setting the parameter `generator_model` to `rbft_llama` or `rbft_qwen` in the config file `configs/eval.yaml`

```bash
python eval/generate.py \
    --data_path data/e5/test \
    --config_file configs/eval.yaml \
    --output_file output/rbft_cf_0.4.json \
    --passage_attack cf \
    --tau 0.4 \ 
    --gpu_id 0
```