# CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=SAGE --dataset=Cora --reset=True --runs=20 --layers=1
# CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GAT --dataset=Cora --reset=True --runs=20 --layers=1
#CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GEN --dataset=Cora --reset=True --runs=20 --layers=1
# CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GEN --dataset=CiteSeer --reset=True --runs=20 --layers=1
# #CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GEN --dataset=CiteSeer --reset=True 
# CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GEN --dataset=CiteSeer --reset=False 
# # CUDA_VISIBLE_DEVICES=0 python3 old.py --gnn=GEN --dataset=Cora --reset=False 
# CUDA_VISIBLE_DEVICES=0 python3 GreedySRM.py --gnn=GCN --dataset=product --layers=20 --reset=True --early=20 --epochs=50 --runs=10 
# CUDA_VISIBLE_DEVICES=1 python3  AdaGNN_h.py --gnn=GCN --dataset=protein --layers=1 --early=20
CUDA_VISIBLE_DEVICES=3 python3 AdaGNN_h.py --gnn=GCN --dataset=protein --early=20 --num_train_parts=20 --num_test_parts=5 --epochs=30
