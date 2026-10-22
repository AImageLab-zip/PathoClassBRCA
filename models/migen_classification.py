import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F

from modules.encoder_classificator_migen import EncoderClassificator


class MiGenModelClassification(nn.Module):
    def __init__(self, args, cfg):
        super(MiGenModelClassification, self).__init__()
        self.args = args
        self.cfg = cfg
        self.encoder_classificator = EncoderClassificator(args, cfg)
            
        if args.dataset_name:
            self.forward = self.forward_brca
        else:
            raise ValueError('no forward function')


    def cal_parameters(self):
        Total_params = 0
        Trainable_params = 0
        NonTrainable_params = 0

        for param in self.parameters():

            mulValue = np.prod(param.size())  
            Total_params += mulValue  
            if param.requires_grad:
                Trainable_params += mulValue  
            else:
                NonTrainable_params += mulValue  

        print(f'Total params: {Total_params}')
        print(f'Trainable params: {Trainable_params}')
        print(f'Non-trainable params: {NonTrainable_params}')

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_brca(self, images, targets=None, mode='train'):
        att_feats = images  
        output = self.encoder_classificator(att_feats)

        return output
    

