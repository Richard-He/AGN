CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=SAGE --dataset=dblp --reset=True --runs=20 --layers=1
CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=GAT --dataset=dblp --reset=True --runs=20 --layers=1
CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=GEN --dataset=dblp --reset=True --runs=20 --layers=1
CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=GCN --dataset=dblp --reset=True --runs=20 --layers=1