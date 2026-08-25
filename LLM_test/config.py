import argparse

def init_arg():
    parser = argparse.ArgumentParser()
    parser.add_argument("--batch_size", default=128, type=int)
    parser.add_argument("--epochs", default=2000, type=int)
    parser.add_argument("--ts_learning_rate", default=0.0004, type=list)
    parser.add_argument("--txt_learning_rate", default=0.00002, type=list)
    parser.add_argument("--root", default='./LLM_test/', type=str)
    parser.add_argument("--replication", default=5, type=int)
    parser.add_argument("--device", default='cuda:0', type=str)
    parser.add_argument("--emb_size", default=768, type=int)
    parser.add_argument("--dx", default=25, type=int) # feature dimension
    parser.add_argument("--act", default='relu', type=str) # activation

    return parser.parse_args()
