from transformers import PreTrainedTokenizerFast

def BartTokenizer(path):
    tokenizer = PreTrainedTokenizerFast.from_pretrained(
        path,
        eos_token='</s>',
        bos_token='<s>',
        unk_token='[UNK]',
        sep_token='</s>',
        pad_token='<pad>',
        cls_token='<s>',
        mask_token='<mask>',
    )
    return tokenizer