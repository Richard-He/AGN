CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=SAGE --dataset=PubMed --reset=False 
CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=GAT --dataset=PubMed --reset=False
CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=GEN --dataset=PubMed --reset=False 
CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=GCN --dataset=PubMed --reset=False 
