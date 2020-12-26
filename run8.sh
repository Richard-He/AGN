CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=SAGE --dataset=dblp --reset=False 
CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=GAT --dataset=dblp --reset=False
CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=GEN --dataset=dblp --reset=False 
CUDA_VISIBLE_DEVICES=1 python3 old.py --gnn=GCN --dataset=dblp --reset=False 
