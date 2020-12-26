CUDA_VISIBLE_DEVICES=2 python3 old.py --gnn=SAGE --dataset=PubMed --reset=True --runs=20 --layers=1
CUDA_VISIBLE_DEVICES=2 python3 old.py --gnn=GAT --dataset=PubMed --reset=True --runs=20 --layers=1
CUDA_VISIBLE_DEVICES=2 python3 old.py --gnn=GEN --dataset=PubMed --reset=True --runs=20 --layers=1
CUDA_VISIBLE_DEVICES=2 python3 old.py --gnn=GCN --dataset=PubMed --reset=True --runs=20 --layers=1 
