CUDA_VISIBLE_DEVICES=3 python3 old.py --gnn=SAGE --dataset=CiteSeer --reset=True --runs=20 --layers=1
CUDA_VISIBLE_DEVICES=3 python3 old.py --gnn=GAT --dataset=CiteSeer --reset=True --runs=20 --layers=1
CUDA_VISIBLE_DEVICES=3 python3 old.py --gnn=GEN --dataset=CiteSeer --reset=True --runs=20 --layers=1
CUDA_VISIBLE_DEVICES=3 python3 old.py --gnn=GCN --dataset=CiteSeer --reset=False --runs=20 --layers=1
