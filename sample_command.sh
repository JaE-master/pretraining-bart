python train.py --batch_size=128 --num_step=1000 --epoch=100 --seed=0 --num_worker=4 \
--train_file=./sample/data/pretrain_sample_10k.txt \
--save_path=./sample/cache_dir \
--cache_dir=./sample/cache_dir/ \
--tokenizer_path=./sample/tokenizer/tokenizer.json

"""
python train.py --batch_size=128 --num_step=2000000 --epoch=100 --seed=0 --num_worker=4 \
--train_file=/data/minjae_kim/pretraining_bart/pretrain_cleaned.txt \
--save_path=/data/minjae_kim/pretraining_bart/saved/ \
--cache_dir=/data/minjae_kim/pretraining_bart/cache_dir/ \
--tokenizer_path=./tokenizer/transformers-tok/tokenizer.json
"""
