CUDA_VISIBLE_DEVICES=3 python3 old.py --gnn=MLP --dataset=Cora --reset=True --runs=20 --layers=1
CUDA_VISIBLE_DEVICES=3 python3 old.py --gnn=MLP --dataset=PubMed --reset=True --runs=20 --layers=1
CUDA_VISIBLE_DEVICES=3 python3 old.py --gnn=MLP --dataset=CiteSeer --reset=True --runs=20 --layers=1
CUDA_VISIBLE_DEVICES=3 python3 old.py --gnn=MLP --dataset=dblp --reset=True --runs=20 --layers=1
