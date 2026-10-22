import numpy as np
import torch
import torch.nn as nn
from PathoClass_GitHub.modules.encoder_classificator_histgen import BaseHistGenClassification

class HistGenModelClassification(nn.Module):
    def __init__(self, args, cfg):
        super(HistGenModelClassification, self).__init__()
        self.args = args
        self.encoder_classification = BaseHistGenClassification(args, cfg)
        self.forward = self.forward_pathology

    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_pathology(self, images, targets=None, mode='train', update_opts={}):
        att_feats = images                          # (B, N_patches, 1024)
        # fc_feats = torch.mean(att_feats, dim=1)     # (B, N_patches)
        output = self.encoder_classification(att_feats)          
        return output
        
