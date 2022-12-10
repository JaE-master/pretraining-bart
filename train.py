import torch
from torch.utils.data import DataLoader
import logging
import datetime

# BART (pre)training module
from transformers import BartForConditionalGeneration
from utils.dataset import BARTDataset
from utils.trainer import Trainer
from utils import constant as config
from utils.bart_util import set_seed, bart_config
from utils.argmuents import parse_args
from utils.tokenizer import BartTokenizer

#fsdp and subprocess module
from torch.utils.data.distributed import DistributedSampler
import os, sys, subprocess
from time import sleep
import numpy as np

def run(args):
    rank = int(os.environ.get("LOCAL_RANK",args.start_worker_idx)) # default 0
    if rank==args.start_worker_idx:
        call_child_script(args.start_worker_idx,args.num_worker) # subprocess 생성
        logger.info('args - '+str(args))

    args.world_size = int(os.environ.get("WORLD_SIZE"))
    logger.info('{} rank logger initiated'.format(rank))

    args.bartconf = bart_config(vocab_size=len(args.tok.vocab))
    model = BartForConditionalGeneration(args.bartconf)

    # load data
    train_dataset = BARTDataset('tr',args)
    sampler = DistributedSampler(train_dataset, rank=rank, num_replicas=args.world_size, shuffle=True, seed=args.seed)
    mp_context = torch.multiprocessing.get_context('fork') # DDP process spawn, dataloade fork not spawn. https://m.facebook.com/groups/PyTorchKR/permalink/1719272184879122/?_se_imp=0VMBba5MyMHXi9p1V
    train_dataloader = DataLoader(dataset=train_dataset,
                                batch_size = args.batch_size,
                                shuffle=False, # sampler option is mutually exclusive with shuffle
                                drop_last=True,
                                num_workers=16,
                                collate_fn=train_dataset.pad_collate,
                                pin_memory=True,
                                sampler=sampler,
                                multiprocessing_context=mp_context
                                ) ## to-do : auto_wrap_policy -> optional if wrap in one FSDP unit

    # trainer
    trainer = Trainer(args, model, train_dataloader, None, logger, rank) # rank 추가
    trainer.train()
    logger.info('{} rank CLEANED'.format(rank)) # after cleanup()

def call_child_script(offset, world_size):
    """References

    https://github.com/Lightning-AI/lightning
    """
    path_lib = os.path.abspath
    command = sys.argv
    full_path = path_lib(command[0])
    command[0] = full_path
    command = [sys.executable] + command # 완성
    logging_info = ''
    for item in command:
        logging_info += item
    logger.info('command - '+logging_info)

    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12389"
    os.environ["NODE_RANK"] = str(0)
    os.environ["LOCAL_RANK"] = str(offset)
    os.environ['CUDA_LAUNCH_BLOCKING'] = '1'
    
    num_processes = world_size # num_processes = torch.cuda.device_count()
    num_nodes = config.num_nodes
    os.environ["WORLD_SIZE"] = f"{num_processes*num_nodes}" # num_processes * num_nodes

    for local_rank in range(offset+1,offset+num_processes):
        env_copy = os.environ.copy()
        env_copy["LOCAL_RANK"] = f"{local_rank}"
        
        if os.environ.get("PL_GLOBAL_SEED") is None and "PL_GLOBAL_SEED" in env_copy:
            del env_copy["PL_GLOBAL_SEED"]
        subprocess.Popen(command, env=env_copy, cwd=None)
        # starting all processes at once can cause issues with dataloaders delay between 1-10 seconds
        delay = np.random.uniform(1, 10, 1)[0]
        sleep(delay)

if __name__ == "__main__":
    # set arguments
    args = parse_args()
    args.tok = BartTokenizer(args.tokenizer_path)
    args.num_worker = min(args.num_worker, torch.cuda.device_count())

    # set logger
    log_file_name = '/'+datetime.datetime.now().strftime("%Y-%m-%d-%H-%M")+'.log'
    logging.basicConfig(filename= args.save_path+log_file_name,
                        filemode = 'a',
                        format = "%(levelname)s\t%(asctime)s\t%(message)s",
                        level = logging.INFO,
                        )
    logger = logging.getLogger("BartLogger")
    if os.environ.get("LOCAL_RANK",args.start_worker_idx)==0:
        print('save_path: {}, log_file_name: {}'.format(args.save_path, log_file_name))
        logger.info('log_file_name :{}\ttokenizer path :{}'.format(log_file_name,args.tok.__dict__['name_or_path']))
    
    set_seed(args.seed)
    # run subprocess
    run(args)