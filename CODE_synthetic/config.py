import argparse

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--ts_learning_rate", default=0.0004, type=list)
    parser.add_argument("--txt_learning_rate", default=0.00002, type=list)
    parser.add_argument("--pretrain_learning_rate", default=0.00002, type=list)
    parser.add_argument("--root", default='./CODE_synthetic/', type=str)
    parser.add_argument("--replication", default=5, type=int)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--emb_size", default=768, type=int)
    parser.add_argument("--tp_embed_dim", default=128, type=int)
    parser.add_argument("--tp_layer", default=2, type=int)
    parser.add_argument("--dx", default=25, type=int) # feature dimension
    parser.add_argument("--act", default='relu', type=str) # activation
    parser.add_argument("--lambda0", default=1.0, type=float)
    parser.add_argument("--lambda1", default=1.0, type=float)
    parser.add_argument("--lambda2", default=1.0, type=float)
    parser.add_argument("--highest_percent", default=0.5, type=float)


    parser.add_argument("--kernel_size", default=1, type=int)
    parser.add_argument("--d_cls", default=0.8, type=float)
    parser.add_argument("--sentence_del", default=1., type=float)
    parser.add_argument("--L", default=4.0, type=int)
    parser.add_argument("--seed", default=66, type=int)
    parser.add_argument("--reg_ts", action='store_false')
    parser.add_argument("--hidden_auto", default=128, type=int)
    parser.add_argument("--latent_dim_auto", default=128, type=int)
    parser.add_argument("--TS_mixup", action='store_false', help='mix up reg and irg data')
    parser.add_argument("--mixup_level", default='batch', type=str)
    parser.add_argument("--task", default='cls', type=str)
    parser.add_argument("--tt_max", default=48, type=int)
    parser.add_argument("--cross_method", default='self_cross', type=str)
    parser.add_argument("--layers", type=int, default=3, help="Number of transformer encoder layer.")
    parser.add_argument("--cross_layers", type=int, default=3, help="Number of transformer cross encoder layer.")
    parser.add_argument("--num_labels", default=2, type=int)
    parser.add_argument("--num_heads", default=8, type=int)
    parser.add_argument("--dropout", default=0.1, type=float)
    parser.add_argument("--embed_dim", default=128, type=int)
    parser.add_argument("--modeltype", default='Text', type=str)
    parser.add_argument("--text_seq_num", default=10, type=str, help='number of notes including padded notes')
    parser.add_argument("--irregular_learn_emb_ts", action='store_false')
    parser.add_argument("--irregular_learn_emb_text", action='store_false')
    parser.add_argument("--checkpoint", action='store_false')
    parser.add_argument("--embed_time", default=64, type=int, help="emdedding for time.")
    parser.add_argument("--ts_cond", action='store_true') # diffusion uses ts as condition, default: True
    parser.add_argument("--n_times", default=40, type=int) # diffusion schedule
    
    return parser.parse_args()
