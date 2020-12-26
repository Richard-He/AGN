# CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=SAGE --dataset=Cora --reset=True --runs=20 --layers=1
# CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GAT --dataset=Cora --reset=True --runs=20 --layers=1
#CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GEN --dataset=Cora --reset=True --runs=20 --layers=1
# CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GEN --dataset=CiteSeer --reset=True --runs=20 --layers=1
# #CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GEN --dataset=CiteSeer --reset=True 
# CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GEN --dataset=CiteSeer --reset=False 
# # CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GEN --dataset=Cora --reset=False 
CUDA_VISIBLE_DEVICES=3 python3 greedySRM_pt.py