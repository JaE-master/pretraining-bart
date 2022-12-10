from __future__ import absolute_import

# Model
epoch = 100
batch_size = 128
# dropout = 0.1 # default 0.1 설정, classifier_dropout 수동 0.1 설정
learning_rate = 2e-5
# GPU device
num_nodes = 1
# device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
seed=0

# Steps
gradient_accumulation_steps = 1 ## save_step의 약수로 ㄱ