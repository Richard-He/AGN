CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=SAGE --dataset=Cora --reset=False 
CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GAT --dataset=Cora --reset=False
CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GEN --dataset=Cora --reset=False 
CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GCN --dataset=Cora --reset=False 
