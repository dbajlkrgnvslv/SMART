import os
from typing import List, Dict, Type
import math

import torch
from torch.optim import Optimizer
import transformers
from sklearn.metrics import accuracy_score,roc_auc_score
from torch.nn import CrossEntropyLoss
from sklearn import preprocessing
from sklearn.manifold import TSNE
import numpy as np

WEIGHTS_NAME = "pytorch_model.bin"

class Trainer:
    '''trainer for single-gpu training.
    '''
    def __init__(self, args=None):
        pass

    def train(self,
        model,
        dataloader,
        valdataloader,
        epochs: int = 1,
        scheduler: str = 'WarmupCosine',
        warmup_steps: int = 10000,
        warmup_ratio: float = 0.01,
        output_path: str = './checkpoints/',
        metric_path: str = '/data-pool/data/data2/qiuhui/code/Alifuse_bibm/checkpoints/ablation_metrics.txt',
        optimizer_class: Type[Optimizer] = torch.optim.AdamW,
        optimizer_params : Dict[str, object]= {'lr': 2e-5},
        weight_decay: float = 0.01,
        max_grad_norm: float = 1,
        accumulation_steps: int = 1,
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''
        self.accumulation_steps = accumulation_steps
        steps_per_epoch = len(dataloader)
        num_train_steps = int((steps_per_epoch) * epochs)
        warmup_steps = math.ceil(num_train_steps * warmup_ratio) #10% of train data for warm-up

        # Prepare optimizers
        param_optimizer = list(model.named_parameters())

        no_decay = ['bias', 'LayerNorm.bias', 'LayerNorm.weight']
        optimizer_grouped_parameters = [
            {'params': [p for n, p in param_optimizer if not any(nd in n for nd in no_decay)], 'weight_decay': weight_decay},
            {'params': [p for n, p in param_optimizer if any(nd in n for nd in no_decay)], 'weight_decay': 0.0}
        ]

        optimizer = optimizer_class(optimizer_grouped_parameters, **optimizer_params)
        scheduler = self._get_scheduler(optimizer, scheduler=scheduler, warmup_steps=warmup_steps, t_total=num_train_steps)

        model = model.cuda()
        skip_scheduler = False
        for epoch in range(epochs):

            # self.test(model, valdataloader, metric_path,epoch)
            # import pdb;pdb.set_trace()

            data_iterator = iter(dataloader)
            for train_iter in range(steps_per_epoch):
                model.zero_grad()
                model.train()              
                data = next(data_iterator)

                loss = model(data)
                loss_value = loss['loss'] / self.accumulation_steps
                loss_value.backward()
                torch.nn.utils.clip_grad_norm_(model.parameters(), max_grad_norm)
                optimizer.step()

                print('Epoch[{}/{}]/Iter[{}/{}]: loss: {:.4f}'.format(epoch,epochs,train_iter,steps_per_epoch,loss_value))
                
                optimizer.zero_grad()

                if not skip_scheduler:
                    scheduler.step()
            self._save_ckpt(model,epoch,output_path)
            self.test(model, valdataloader, metric_path,epoch)
            

    def test(
        self,
        model,
        eval_dataloader,
        metric_path,
        epoch,
        num_classes=3,
        ):
        '''
        output_path: model save path
        checkpoint_path: model load and continue to learn path
        '''

        steps_per_epoch = len(eval_dataloader)
        model = model.cuda()
        data_iterator = iter(eval_dataloader)
        gts = []
        preds = []
        for eval_iter in range(steps_per_epoch): # steps_per_epoch
            print('EVAL: ', eval_iter, '/', steps_per_epoch)
            model.eval()
            data = next(data_iterator)
            with torch.no_grad():
                pred = model.predict(data)
            preds.append(pred)
            gts.append(data[2])
        
        preds = torch.cat(preds,dim=0)
        gts = torch.cat(gts,dim=0)
        nonlabel_indices = torch.nonzero(gts==-100).squeeze()
        gts = torch.index_select(gts, 0, torch.tensor([i for i in range(gts.shape[0]) if i not in nonlabel_indices]))
        preds = torch.index_select(preds.cpu(), 0, torch.tensor([i for i in range(preds.shape[0]) if i not in nonlabel_indices]))
        gts_one_hot = torch.nn.functional.one_hot(gts, num_classes=num_classes)

        aa = preds.argmax(-1)
        preds_ont_hot = torch.nn.functional.one_hot(aa, num_classes=num_classes)
        auc = roc_auc_score(gts_one_hot.ravel(), preds_ont_hot.ravel())
        acc = accuracy_score(aa.cpu(),gts.cpu())
        print('AUC: ',auc, ' ACC: ',acc)
        with open(metric_path,'a+') as f:
            line = 'epoch '+str(epoch)+', AUC: '+str(auc)+ ' ACC: '+str(acc) +'\n'
            f.write(line)
        return
        
    @staticmethod
    def _get_scheduler(optimizer, scheduler: str, warmup_steps: int, t_total: int):
        """
        Returns the correct learning rate scheduler. Available scheduler: constantlr, warmupconstant, warmuplinear, warmupcosine, warmupcosinewithhardrestarts
        """
        scheduler = scheduler.lower()
        if scheduler == 'constantlr':
            return transformers.get_constant_schedule(optimizer)
        elif scheduler == 'warmupconstant':
            return transformers.get_constant_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps)
        elif scheduler == 'warmuplinear':
            return transformers.get_linear_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosine':
            return transformers.get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        elif scheduler == 'warmupcosinewithhardrestarts':
            return transformers.get_cosine_with_hard_restarts_schedule_with_warmup(optimizer, num_warmup_steps=warmup_steps, num_training_steps=t_total)
        else:
            raise ValueError("Unknown scheduler {}".format(scheduler))

    def _save_ckpt(self, model, epoch, save_dir):
        if not os.path.exists(save_dir): 
            os.makedirs(save_dir)
        state_dict = model.state_dict()
        torch.save(state_dict, os.path.join(save_dir, 'epoch{}.pth'.format(epoch)))
