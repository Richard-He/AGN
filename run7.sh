CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=SAGE --dataset=CiteSeer --reset=False 
CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=GAT --dataset=CiteSeer --reset=False
CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=GEN --dataset=CiteSeer --reset=False 
CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=GCN --dataset=CiteSeer --reset=False 
