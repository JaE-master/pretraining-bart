import argparse
from utils import constant as config

def parse_args():
    parser = argparse.ArgumentParser()
    # I/O config
    parser.add_argument('--save_path', default='/data/mj/saved/')
    parser.add_argument('--cache_dir',default='/data/mj/cache_pretrain',type=str)
    parser.add_argument('--train_file', default='/data/mj/pretrain_cleaned_10k.txt',type=str)
    parser.add_argument('--tokenizer_path', default='./tokenizer/transformers-tok/tokenizer.json',type=str)
    
    # multi-processing (FSDP) config
    parser.add_argument('--num_worker', default=4,type=int)
    parser.add_argument('--start_worker_idx', default=0,type=int)
    
    # training config
    parser.add_argument('--epoch',default=config.epoch, type=int)
    parser.add_argument('--batch_size', default=config.batch_size, type=int)
    parser.add_argument('--num_step',default=10000,type=int)
    parser.add_argument('--logging_step',default=1000,type=int)
    parser.add_argument('--saving_step',default=50000,type=int)
    parser.add_argument('--seed',default=0,type=int)

    # model config
    parser.add_argument('--learning_rate', default=config.learning_rate,type=float)
    parser.add_argument('--max_grad_norm', default=5.0, type=float)
    
    args = parser.parse_args()
    # args.device = config.device
    args.gradient_accumulation_steps = config.gradient_accumulation_steps
    return args