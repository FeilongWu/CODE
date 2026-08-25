import argparse

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--lr", default=2e-5, type=float)
    parser.add_argument("--root", default='./CODE/', type=str)
    parser.add_argument("--replication", default=5, type=int)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--emb_size", default=1024, type=int)
    parser.add_argument("--time_series_size", default=25, type=int) # feature dimension
    parser.add_argument("--act", default='relu', type=str) # activation
    parser.add_argument("--output_dim", default=2, type=int) # 2 for binary
    parser.add_argument("--num_modalities", default=2, type=int)
    parser.add_argument("--hidden_dim", default=1024, type=int)
    parser.add_argument("--num_layers", default=2, type=int)
    parser.add_argument("--num_layers_pred", default=2, type=int)
    parser.add_argument('--num_experts', type=int, default=32)
    parser.add_argument('--top_k', type=int, default=4) # top_k Routers
    parser.add_argument('--num_routers', type=int, default=1) # Number of Routers
    parser.add_argument('--num_heads', type=int, default=4) # Number of heads
    parser.add_argument('--dropout', type=float, default=0.1)

    parser.add_argument("--diff_learning_rate", default=2e-5, type=float)
    parser.add_argument("--diff_embed_dim", default=20, type=int)
    parser.add_argument("--diff_layer", default=2, type=int)
    parser.add_argument("--lambda0", default=0.1, type=float)
    parser.add_argument("--lambda1", default=0.1, type=float)
    parser.add_argument("--lambda2", default=0.1, type=float)
    parser.add_argument("--highest_percent", default=0.1, type=float)
    parser.add_argument("--embed_dim", default=60, type=int)
    parser.add_argument("--n_times", default=40, type=int)
    parser.add_argument("--ts_cond", action='store_false') # diffusion uses ts as condition, default: True
    parser.add_argument("--embed_time", default=64, type=int, help="emdedding for time.")
    parser.add_argument("--seed", default=66, type=int)
    parser.add_argument("--use_diffusion", action='store_false')
    parser.add_argument("--multimodal", action='store_false')
    parser.add_argument("--accumulate_step",  default=4, type=int)
    
    
    
    
    return parser.parse_args()
