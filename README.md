# Pretraining-BART

This project is to pre-train **BART**[[1]](#1) model for Korean language using **FSDP (Fully Sharded Data Parallel)**[[2]](#2) method.<br/>
FSDP is a type of data-parallel training which shards parameters and caches them to fully utilize GPU workers' memory.<br/>
This project only implements *text-infilling* for pretraining objective out of BART's 5 proposed noising algorithms.

<br/>

## Details
`files and path structure`
```
/data
└--pretraining_bart
    └--PRETRAIN_DATA.txt
    └--cache_dir
    └--saved
.
└--train.py
└--pretraining-tokenizer.py
└--exec_pretrain.sh
└--exec_pretrain_tokenizer.sh
└--requirements.txt
└--README.md
└--utils
    └--arguments.py
    └--bart_util.py
    └--constant.py
    └--dataset.py
    └--tokenizer.py
    └--trainer.py
└--tokenizer
    └--transformers-tok
        └--tokenizer.json
```
<br/>

`description for principal source files`
| source          |  description                                                    | 
| ----------------- |  -------------------------------------------------------------- | 
| `train.py`       |  Spawn multi-processes for FSDP training.<br/> Initiate training modules and launch. | 
| `bart-util.py`       |  Declare BART-base config and provide text-infilling algorithm. | 
| `trainer.py`       |  Train BART using FSDP method. | 
| `pretraining-tokenizer.py`       |  Train BART tokenizer with WordPiece algorithm.<br/>*Sample trained tokenizer is provided in `sample` directory.*| 
<br/>


`Data`<p>
Korean sentences from `모두의 말뭉치, 위키, 나무위키, 뉴스`. Total size about 40GB.<br/>
*Sample PRETRAIN_DATA is provided in `sample` directory.*
</p>
<br/><br/>

## How to run

1. Install all required packages.<br/>

    First, install appropriate torch version depending on your compute platform. You can find installation command from [here](https://pytorch.org/get-started/locally/). For instance, 
    ```
    pip install torch==1.12.1+cu113 torchvision==0.13.1+cu113 torchaudio==0.12.1 --extra-index-url https://download.pytorch.org/whl/cu113
    ```
    *Torch's FSDP functionality is provided **version above 1.10**. You can use fairscale library if torch version unmet.*<br/>

    Install the rest of the required packages.

    ```
    pip install -r requirements.txt
    ```
    <br/>

2. Pre-train Tokenizer.
    ```shell
    sh exec_pretrain_tokenizer.sh
    ```
    <br/>
3. Run the train command.
    ```shell
    sh exec_pretrain.sh
    ```
<br/>

---

### References

<a id="1">[1]</a> *Lewis, Mike et al. “BART: Denoising Sequence-to-Sequence Pre-training for Natural Language Generation, Translation, and Comprehension.” Annual Meeting of the Association for Computational Linguistics (2019).* ([https://arxiv.org/abs/1910.13461](https://arxiv.org/abs/1910.13461))

<a id="2">[2]</a> *Ott, Myle et al. (2021). Fully Sharded Data Parallel: faster AI training with fewer GPUs.* ([https://engineering.fb.com/2021/07/15/open-source/fsdp/](https://engineering.fb.com/2021/07/15/open-source/fsdp/))