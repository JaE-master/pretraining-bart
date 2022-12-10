import os
import torch
from transformers.optimization import get_linear_schedule_with_warmup
from torch.optim import AdamW
from tqdm import tqdm
import time

# DDP module
import torch.distributed as dist
from torch.distributed.fsdp import FullyShardedDataParallel as FSDP
# from fairscale.nn.data_parallel import FullyShardedDataParallel as FSDP # if using torch version below 1.11
# from torch.nn.parallel import DistributedDataParallel as DDP # if using DDP instead of FSDP

# DDP functions
def setup(rank, world_size):
    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size) # 각 gpu별 process 등록 및 broadcast, all-reduce등의 communication
    torch.cuda.set_device(rank) # model.to(torch.device('cuda')) -> model.to(rank_number)
def cleanup():
    dist.destroy_process_group()

class Trainer:
    def __init__(self, args, model, train_dataloader, valid_dataloader, logger, rank):
        self.args = args
        self.model = model
        self.train_dataloader = train_dataloader
        self.logger = logger
        self.save_path = args.save_path

        # ddp
        setup(rank,args.world_size)
        self.rank = rank
        self.sampler = self.train_dataloader.sampler
        self.t_total = self.args.epoch * len(self.train_dataloader)
        self.global_step = 0
        self.global_loss = 0.0 # rank 0 only
        self.states = None

    def get_model_parameters(self):
        param_optimizer = list(self.model.named_parameters())
        no_decay = ['bias', 'LayerNorm.weight', 'LayerNorm.bias']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 
             'weight_decay': 0.01},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 
             'weight_decay': 0.0}]
        return optimizer_grouped_parameters

    def train(self):
        # wrap DDP model
        self.model.to(self.rank)
        self.model = FSDP(self.model) # DDP 변경 가능
        if self.rank==0:
            self.logger.info('TRAIN START.. using FSDP')

        # (re)set after wrapping model
        self.optimizer = AdamW(self.model.parameters(), lr = self.args.learning_rate)
        self.scheduler = get_linear_schedule_with_warmup(self.optimizer, 0.1*self.t_total, self.t_total)

        self.model.train()
        self.model.zero_grad()
        
        ddp_loss = torch.zeros(2).to(self.rank) # accumulated to global_loss
        for epoch in range(1,self.args.epoch+1):
            time1 = time.time()
            train_loader = tqdm(self.train_dataloader)
            self.sampler.set_epoch(epoch) # shuffle =True일때 epoch 바탕으로 data ordering
            self.logger.info('EPOCH START: {}\trank: {}\tglobal_step: {}'.format(epoch,self.rank,self.global_step)) # asynchronous logging
            for step, batch in enumerate(train_loader):
                loss = self.training_step(self.model, batch)
                ddp_loss[0] += loss
                ddp_loss[1] += self.args.gradient_accumulation_steps
                if (step + 1) % self.args.gradient_accumulation_steps ==0 or (
                    len(train_loader) == (step + 1)
                ):
                    if hasattr(self.optimizer, "clip_grad_norm"):
                        # Some optimizers (like the sharded optimizer) have a specific way to do gradient clipping
                        self.optimizer.clip_grad_norm(self.args.max_grad_norm)
                    else:
                        # Revert to normal clipping otherwise, handling Apex or full precision
                        torch.nn.utils.clip_grad_norm_(
                            self.model.parameters(),
                            self.args.max_grad_norm,
                        )

                    # Optimizer step
                    self.optimizer.step()
                    self.scheduler.step()
                    self.model.zero_grad()
                    
                self.global_step+=1
                if self.global_step%self.args.logging_step==1:
                    dist.reduce(ddp_loss, 0, op=dist.ReduceOp.SUM) # 0번 rank로 reduce
                    if self.rank==0:
                        self.global_loss+=ddp_loss[0]
                        self.logger.info('STEP: {}\tLoss: {:.6f}\tGlobal Loss: {:.6f}\tElapsed Time: {}'.format(
                            self.global_step, ddp_loss[0] / ddp_loss[1],self.global_loss/self.global_step,time.time()-time1))
                        time1=time.time()
                    ddp_loss = torch.zeros(2).to(self.rank) # reset after logging
                if self.global_step%self.args.saving_step==1:
                    dist.barrier() # sync training done
                    self.states = self.model.state_dict() # to save the FSDP model, we need to call the state_dict on each rank then on Rank 0 save the overall states.
                    if self.rank==0:
                        self._save_model(self.global_loss/self.global_step, self.global_step, self.states)
                if self.global_step==self.args.num_step:
                    if self.rank==0:
                        self._save_model(self.global_loss/self.global_step, self.global_step, self.states) # save last step
                    break # break batch loop
            if self.global_step==self.args.num_step:
                self.logger.info('TRAIN FINISHED EPOCH: {}\trank: {}\tglobal_step: {}'.format(epoch,self.rank,self.global_step))
                break # break epoch loop
        dist.barrier()
        if self.rank==0:
            self.logger.info('TRAIN FINISHED')
        cleanup()
        
    def _save_model(self, loss, step, states):
        run_name = 'step_{}_loss_{:.4f}'.format(step,loss)
        output_dir = os.path.join(self.args.save_path,run_name)
        checkpoint_path = output_dir+'_model.pt'
        torch.save(states,checkpoint_path)
        self.logger.info('Model saved to {}'.format(checkpoint_path))

    def _prepare_inputs(self, inputs):
        input_ids, noised_ids = inputs['input_ids'].to(self.rank), inputs['noised_ids'].to(self.rank)
        
        decoder_input_ids = input_ids.new_zeros(input_ids.shape)
        decoder_input_ids[:,1:] = input_ids[:,:-1].clone()
        decoder_input_ids[:,0] = self.args.bartconf.decoder_start_token_id # updated 221204
        
        outs = {
            'input_ids' : input_ids,
            'noised_ids': noised_ids,
            'decoder_input_ids' : decoder_input_ids
        }
        return outs

    def training_step(self, model, inputs):
        model.train()
        inputs = self._prepare_inputs(inputs)
        
        input_ids = inputs['noised_ids']
        decoder_input_ids = inputs['decoder_input_ids']
        attention_mask = input_ids.ne(self.args.bartconf.pad_token_id).float()
        decoder_attention_mask = decoder_input_ids.ne(self.args.bartconf.pad_token_id).float()
        output = model(
            input_ids = input_ids,
            decoder_input_ids = decoder_input_ids,
            attention_mask = attention_mask,
            decoder_attention_mask = decoder_attention_mask,
            labels = inputs['input_ids']
        )
        loss = output['loss']
        # if self.args.n_gpu > 1: # 필요없음
        #     loss = loss.mean()  # mean() to average on multi-gpu parallel training
        # https://engineering.fb.com/2021/07/15/open-source/fsdp/
        if self.args.gradient_accumulation_steps > 1:
            loss = loss / self.args.gradient_accumulation_steps
        loss.backward()
        return loss.detach()


""" Legacies

import subprocess
import gc

def getCurrentMemoryUsage(): ## deprecated
    ''' Memory usage in kB '''

    with open('/proc/self/status') as f:
        memusage = f.read().split('VmRSS:')[1].split('\n')[0][:-3]

    return int(memusage.strip())

def write_ram_info(file_name,step):
    with open(file_name, 'a') as fin:
        try:
            a = subprocess.run("nvidia-smi",shell=True,capture_output=True)
            b = subprocess.run("free -h",shell=True,capture_output=True)
            fin.write('*********current step: {} **************'.format(step))
            fin.write(a.stdout.decode('ascii'))
            fin.write(b.stdout.decode('ascii'))
        except:
            print('ram info error at step',step)
            exit()
    # os.environ["TOKENIZERS_PARALLELISM"] = "false"
"""