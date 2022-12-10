"""
- Sample Command
python pretraining-tokenizer.py --corpus_file=./tokenizer/naver_review.txt --min_frequency=5 --save_dir=./tokenizer
python pretraining-tokenizer.py --corpus_file=/data/minjae_kim/pretraining_bart/pretrain_cleaned.txt --min_frequency=5 --save_dir=./tokenizer

- References
https://huggingface.co/
https://blog.naver.com/PostView.naver?blogId=sooftware&logNo=222494375953&parentCategoryNo=&categoryNo=13&viewDate=&isShowPopularPosts=false&from=postView
"""
import os
import argparse
from tokenizers import BertWordPieceTokenizer
from transformers import PreTrainedTokenizerFast
parser = argparse.ArgumentParser()
parser.add_argument('--corpus_file', type=str, required=True)
parser.add_argument('--limit_alphabet', type=int, default=1000) # 유지
parser.add_argument('--min_frequency', type=int, default=5) # hugging default 2 -> 5로 변경
parser.add_argument('--vocab_size', type=int, default=30000) # hugging default 유지
parser.add_argument('--save_dir', type=str, default='./tok')
args = parser.parse_args()

# save path
tokenizers_path = os.path.join(args.save_dir,'tokenizers-tok')
transformers_path = os.path.join(args.save_dir,'transformers-tok')
if not os.path.exists(tokenizers_path):
    os.mkdir(tokenizers_path)
if not os.path.exists(transformers_path):
    os.mkdir(transformers_path)
print('transformers_path confirmed')

# train tokenizers
special_tokens = [# 허깅페이스 config
    "<s>",
    "<pad>",
    "</s>",
    "[UNK]", # default for unk_token
    "<mask>",
]

# <unused0>~<unused99>
unused_str = '<unused'
for i in range(100):
    special_tokens.append(unused_str+str(i)+'>')

tokenizer = BertWordPieceTokenizer(
    strip_accents=True,
    lowercase=False, # 대소문자 구분
)

tokenizer.train(
    files=args.corpus_file,
    limit_alphabet=args.limit_alphabet,
    min_frequency=args.min_frequency, # 병합 최소 기준
    special_tokens=special_tokens,
    vocab_size=args.vocab_size)
tokenizer.save(os.path.join(tokenizers_path,'tokenizer.json'))
# tokenizer.save_model(args.save_dir+'/tokenizers-tok')
print('tokenizers trained finished')

# convert to transformers tokenizer
tok = PreTrainedTokenizerFast(tokenizer_object=tokenizer,
                                eos_token='</s>',
                                bos_token='<s>',
                                unk_token='[UNK]',
                                sep_token='</s>',
                                pad_token='<pad>',
                                cls_token='<s>',
                                mask_token='<mask>',)
tok.save_pretrained(transformers_path)
print('transformers trained finished')

# test
tok_test = PreTrainedTokenizerFast.from_pretrained(transformers_path+'/tokenizer.json',
                                                        eos_token='</s>',
                                                        bos_token='<s>',
                                                        unk_token='[UNK]',
                                                        sep_token='</s>',
                                                        pad_token='<pad>',
                                                        cls_token='<s>',
                                                        mask_token='<mask>',)
encoded_tok = tok_test.encode('아 배고픈데 치킨 먹고싶다.')
decoded_str = tok_test.decode(encoded_tok)
print(encoded_tok)
print(decoded_str)
print(tok_test.__dict__)
vocab_dict = {v:k for k,v in tok_test.vocab.items()}
print('\n\ntest finished!')