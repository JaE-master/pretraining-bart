import torch
import numpy as np
import random
import math
from transformers import BartConfig

def set_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def bart_config(vocab_size):
    conf = BartConfig() # default BART LARGE >> vocab: 50265 bos_token_id: 0 eos_token_id: 2 decoder_start_token_id: 2
    
    # modify to BART-base configuration
    conf.vocab_size = vocab_size
    conf.d_model = 768 # 1024 -> 768
    conf.max_position_embeddings = 512 # max seq length
    conf.encoder_layers = 6 # 12 -> 6
    conf.decoder_layers = 6
    conf.encoder_ffn_dim = 4*768 # 4096 ->3072
    conf.decoder_ffn_dim = 4*768
    conf.classifier_dropout = 0.1 # default dropout도 0.1
    # num_attention_heads = 12 , pad_token_id =1, decoder_start_token_id =2 (eos)
    return conf


def text_infilling(src, p, mask_token_id):
    """ BART TEXT INFILLING ALGORITHM

    src : source tensor tokenized(encoded).
    p : proportion of masking (e.g., BERT for 15%)
    mask_token_id : args.tok.vocab['<mask>'], usually 4 in huggingface modules

    References - https://github.com/facebookresearch/fairseq
    """

    is_word_start = torch.ones(src.size()) # word_starts(src) -> deprecated (to Legacies): 토크나이저 post_processor 사용 안함
    num_to_mask = int(math.ceil(is_word_start.float().sum()) * p) # 30% 마스킹
    if num_to_mask ==0:
        return src
    lambda_ = torch.ones(num_to_mask) +2
    lengths = torch.poisson(lambda_) # num_to_mask개의 poisson(3) 배열

    # Make sure we have enough to mask -> deprecated (to Legacies): zero mask일 경우 너무 늘어날 수 있음. e.g., num_to_mask 1인데 [0,0,0,0,1]
    
    starting_words = is_word_start.nonzero(as_tuple=False) # torch.tensor.nonzero : returns indices of non-zero elements
    indices = starting_words[torch.randperm(starting_words.size(0))[:num_to_mask]].squeeze(1) # indices : 랜덤하게 num_to_mask개 뽑음
    
    # indices토큰부터 lengths개만큼 masking
    # 1. zero-length masking
    zero_len_indices = [i for i in range(len(lengths)) if lengths[i]==0]
    zero_mask_indices = []
    for i in zero_len_indices:
        zero_mask_indices.append(indices[i])
    zero_mask_indices.sort()

    src,lengths,indices,is_word_start=src.tolist(),lengths.tolist(),indices.tolist(), is_word_start.tolist()
    zero_len_loc = []
    for i,j in enumerate(zero_mask_indices):
        src.insert(i+j,mask_token_id)
        is_word_start.insert(i+j,255) # big integer for no more span masking
        zero_len_loc.append(j)
    
    # trim after zero span handling
    for i,j in enumerate(zero_len_indices):
        del lengths[j-i]
        del indices[j-i]
    zero_len_loc.sort()
    for i in range(len(indices)):
        for j in zero_len_loc:
            if indices[i]>j:
                indices[i]+=1
            else:
                break
    src,lengths,indices, is_word_start=torch.tensor(src),torch.tensor(lengths),torch.tensor(indices), torch.tensor(is_word_start)
    num_to_mask = lengths.size(0)
    if num_to_mask == 0:
        return src
    assert (lengths>0).all()
    
    ## 2. initial masking (start indices)
    is_word_start[-1] = 255 # acts as a long length, so spans don't go over the end of doc
    src[indices] = mask_token_id
    lengths-=is_word_start[indices]
    uncompleted = lengths>0
    indices = indices[uncompleted]
    lengths = lengths[uncompleted]
    
    # 3. single masking
    while indices.size(0) > 0: # mask 처리할 indices 남아있음
        assert lengths.size() == indices.size()
        lengths -= is_word_start[indices+1].long()
        uncompleted = lengths >=0
        indices = indices[uncompleted]+1 # span 내에서 next token
        lengths = lengths[uncompleted]
        if indices.size(0)>0: # zero masking 처리된 애들 고려
            src[indices] = -100 # 임시
    idx=0
    while idx<len(src):
        if src[idx]==-100:
            assert src[idx-1]==mask_token_id
            src = torch.cat([src[:idx],src[idx+1:]])
            idx-=1
        idx+=1
    return src

""" Legacies

    def word_starts(src): # If tokenizer use post processing of tagging <bos> and <eos>
        is_word_start = torch.ones(src.size())
        is_word_start[0]=0
        is_word_start[-1] = 0
        return is_word_start


    # Make sure we have enough to mask
    cum_length = torch.cumsum(lengths,0)
    # while cum_length[-1] < num_to_mask:
    if cum_length[-1] < num_to_mask:
        lengths = torch.cat([lengths, torch.poisson(lambda_)],dim=0)
        cum_length = torch.cumsum(lengths,0)
        
    #Trim to masking budget
    idx=0
    while cum_length[idx] < num_to_mask: # lengths와 num_to_mask 개수 조
        idx+=1
    lengths[idx] = num_to_mask - (0 if idx==0 else cum_length[idx-1])
    num_to_mask = idx+1
    lengths = lengths[:num_to_mask]
"""