from torch.utils.data import Dataset
from torch.nn.utils.rnn import pad_sequence
import torch
from datasets import load_dataset
from utils.bart_util import text_infilling

class BARTDataset(Dataset):
    def __init__(self, dtype, args):
        self.cnt=0
        self.args = args
        self.max_len = args.bartconf.max_position_embeddings
        self.pad_token_id = args.bartconf.pad_token_id
        if dtype=='tr':
            self.tok = self.args.tok
            self.docs = load_dataset("text",data_files=args.train_file,cache_dir=args.cache_dir)['train']
            self.length = len(self.docs)
        self.mask_token_id = args.tok.vocab['<mask>']

    def __getitem__(self, idx):
        line = self.docs[idx]['text']
        input_ids = torch.tensor(self.tok.encode(line)) 
        noised_ids = text_infilling(input_ids,0.3, self.mask_token_id)
        return input_ids, noised_ids
                
    def __len__(self):
        return self.length

    def pad_collate(self,batch):
        input_ids, noised_ids = zip(*batch)
        input_ids = list(input_ids) # tuple to list for modify
        noised_ids = list(noised_ids)

        for i in range(self.args.batch_size):
            if len(input_ids[i]) > self.max_len:
                input_ids[i] = input_ids[i][:self.max_len]
            if len(noised_ids[i]) > self.max_len:
                noised_ids[i] = noised_ids[i][:self.max_len]
            if len(input_ids) < self.args.batch_size: # `drop_last = False` exception
                if i == len(input_ids)-1:
                    break
        input_ids = pad_sequence(input_ids,True,self.pad_token_id)
        noised_ids = pad_sequence(noised_ids,True,self.pad_token_id)

        return {'input_ids': input_ids, 'noised_ids': noised_ids}
