import argparse

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--lr", default=2e-5, type=list)
    parser.add_argument("--root", default='./MoE_Retriever/', type=str)
    parser.add_argument("--replication", default=5, type=int)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--emb_size", default=768, type=int)
    parser.add_argument("--time_series_size", default=25, type=int) # feature dimension
    parser.add_argument("--act", default='relu', type=str) # activation
    parser.add_argument("--output_dim", default=2, type=int) # 2 for binary
    parser.add_argument("--num_modalities", default=2, type=int)
    parser.add_argument("--hidden_dim", default=768, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--num_layers_pred", default=2, type=int)
    parser.add_argument('--num_experts', type=int, default=32)
    parser.add_argument('--top_k', type=int, default=4) # top_k Routers
    parser.add_argument('--num_routers', type=int, default=1) # Number of Routers
    parser.add_argument('--num_heads', type=int, default=4) # Number of heads
    parser.add_argument('--dropout', type=float, default=0.1)
    
    
    
    
    return parser.parse_args()
