import os
import random

import numpy as np
import torch
from smart.modeling_smart import SMARTModel
from dataset.dataset import get_dataloader
from smart.trainer import Trainer

# set random seed
seed = 42
random.seed(seed)
np.random.seed(seed)
torch.manual_seed(seed)
torch.cuda.manual_seed(seed)
os.environ['PYTHONASHSEED'] = str(seed)
os.environ['TOKENIZERS_PARALLELISM']='false'

# only pretrain on ADNI train data and NACC train data
train_datalist = [
    'ADNI-train',
    # 'AIBL-train',
    # 'PPMI-train',
]

val_datalist = [
    'ADNI-test',
    # 'AIBL-test',
    # 'PPMI-test',
]

trainloader = get_dataloader(train_datalist, batch_size=6,shuffle=True,num_workers=2, drop_last=True)
valloader = get_dataloader(val_datalist, batch_size=6,shuffle=False,num_workers=2, drop_last=False)

model = SMARTModel()
load_model_path = ''
# load_model_path = './checkpoints/epoch0.pth'
if load_model_path != '':
    model_dict = model.state_dict()
    pretrained_dict = torch.load(load_model_path, map_location='cpu')
    pretrained_dict = {k : v for k, v in pretrained_dict.items() if (k in model_dict.keys() and np.shape(model_dict[k]) == np.shape(v))}
    model_dict.update(pretrained_dict)
    model.load_state_dict(model_dict)
model.cuda()
trainer = Trainer()

NAME = 'lintra_linter_lcls'
trainer.train(
    model,
    trainloader,
    valloader,
    warmup_ratio=0.1,
    epochs=300,
    optimizer_params={'lr':3e-4},
    output_path='./checkpoints/{}/'.format(NAME),
    metric_path='./checkpoints/{}_metrics.txt'.format(NAME),
    weight_decay=1e-4,
    )
    

