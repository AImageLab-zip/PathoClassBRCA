import torch
import torch.nn as nn
import numpy as np
import torch.nn.functional as F
from modules.encoder_decoder_bigen import EncoderDecoder


class BiGenModel(nn.Module):
    def __init__(self, args, tokenizer, encoder_decoder=None):
        super(BiGenModel, self).__init__()
        self.args = args
        self.tokenizer = tokenizer
        self.prompt = nn.Parameter(torch.randn(1, 1, args.d_vf))
        self.prompt_plip = nn.Parameter(torch.randn(1, 1, 512))
        self.fc = nn.Sequential(nn.LayerNorm(args.d_model),nn.Linear(args.d_model,args.d_model),nn.Linear(args.d_model,args.n_classes))
        if not encoder_decoder:
            print('use encoder_decoder: default')
            self.encoder_decoder = EncoderDecoder(args, tokenizer)
        if args.dataset_name:
            self.forward = self.forward_brca
        else:
            raise ValueError('no forward function')


    def __str__(self):
        model_parameters = filter(lambda p: p.requires_grad, self.parameters())
        params = sum([np.prod(p.size()) for p in model_parameters])
        return super().__str__() + '\nTrainable parameters: {}'.format(params)

    def forward_brca(self, images, targets=None, mode='train'):

        att_feats = images  # shape 1*N*1536
    
        # self.prompt       : (B, 1, D1)
        # self.prompt_plip  : (B, 1, D2)
        # image             : (B, N, D1)
        # image_plip        : (B, N, D2)
        #--------------------------------
        # prompt            : (B, 1, D1 + D2)
        # att_feats         : (B, N, D1 + D2)


        att_feats = torch.cat([torch.cat((self.prompt, self.prompt_plip),dim=2), att_feats], dim=1)
        fc_feats = torch.sum(att_feats[:,:,:-512], dim=1) #shape 1*1024

        if mode == 'train':
            output = self.encoder_decoder(fc_feats, att_feats, targets, mode='forward')
        elif mode == 'sample':
            output, _ = self.encoder_decoder(fc_feats, att_feats, mode='sample')
        elif mode == 'encode':
            output = self.encoder_decoder(fc_feats, att_feats, mode='encode')
            logits = self.fc(output[0,0,:]).unsqueeze(0)
            Y_hat = torch.argmax(logits, dim=1)
            Y_prob = F.softmax(logits, dim=1)
            return Y_hat, Y_prob
        else:
            raise ValueError
        return output

