from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import repeat
from .att_model import pack_wrapper
from .classification_utils import MultiTaskClassifier


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def pad_tokens(att_feats):
    # ---->pad
    H = att_feats.shape[1]
    _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
    add_length = _H * _W - H
    att_feats = torch.cat([att_feats, att_feats[:, :add_length, :]], dim=1) 
    return att_feats

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

def process_features(src, short_text, m, k):
    # src: (1, N, 512)
    n = src.size(1) 
    w = (n + m - 1) // m # number of groups of m patches
    padding = m* w - n   # padding to arrive to a multiple of m -> N=w*m
    src_padded = F.pad(src.squeeze(0), (0, 0, 0, padding), value=0)
    src_groups = src_padded.view(w, m, 512)
    src_avg = src_groups.mean(dim=1) # (w, 512)

    cos_sim = F.cosine_similarity(src_avg.unsqueeze(1), short_text.unsqueeze(0), dim=2)  # (w, 1, 512)*(1, T, 512)->(w, 1, 512)*(w, T, 512)->[w, t]
    # cos_sim = (w, t) = row: a group, column: phrase of the bank
    topk_sim, topk_idx = cos_sim.topk(k, dim=1, largest=True, sorted=True)  # [w, k]-> keep for each group the k phrases most similar
    topk_features = short_text[topk_idx]  # (w, k, 512)

    topk_avg = topk_features.mean(dim=1)  # (w, 512)

    final_feature = topk_avg 
    
    return final_feature

class Transformer(nn.Module):
    def __init__(self, encoder, d_model):
        super(Transformer, self).__init__()
        self.encoder = encoder
        self.d_model = d_model
        # self.decoder = decoder
        # self.src_embed = src_embed
        # self.tgt_embed = tgt_embed
        # self.down_proj = nn.Linear(512, 512)

    def forward(self, src, src_mask):
        return self.encode(src, src_mask)

    def encode(self, src_all, src_mask):
        src_uni = src_all[:, :, :self.d_model]
        x_plip = src_all[:, :, self.d_model:]

        return self.encoder(src_uni, src_mask, x_plip)

class Encoder(nn.Module):
    def __init__(self, layer, N, PAM, bank_path,v, m, k, num_tasks):
        super(Encoder, self).__init__()
        self.bank_path = bank_path
        self.layers = layer
        self.norm = LayerNorm(layer.d_model)
        self.PAM = clones(PAM, N)
        self.N = N
        self.num_tasks = num_tasks
        self.latents = nn.Parameter(torch.randn(self.num_tasks, layer.d_model))
        self.latents_t = nn.Parameter(torch.randn(self.num_tasks, layer.d_model))
        self.down_proj = nn.Linear(512, layer.d_model)
      
        self.short_text = torch.load(self.bank_path) #(7986, 512)
        self.v = v # percentage of patch to keep
        self.k = k # number of top patches to select
        self.m = m # group size for averaging
        self.num_tasks = num_tasks

    def forward(self, x, mask, x_plip):
        # expand for multi-batch
        fu = repeat(self.latents, 'n d -> b n d', b = x.shape[0])     #(B, num_tasks, 512)
        fu_t = repeat(self.latents_t, 'n d -> b n d', b = x.shape[0]) #(B, num_tasks, 512)
        # list for intermediate features of fu and fu_t
        s=[]
        st = []
      
        for i in range(self.N): #3 layers
            if i == 0: # first layer
                # update fu only based on image features
                # sorted_indices are attention scores of patches: (B, N_patches)
                fu, fu_t, sorted_indices = self.layers(x, mask, fu, fu_t, None) # no text retrieval at first layer
                s.append(fu)

            elif i == 1: # second layer
                # reorder x_plip based on attention scores
                sorted_plip = x_plip.squeeze(0)[sorted_indices.squeeze(0)] 
                # keep top v% patches
                x_plip = sorted_plip[:int(sorted_plip.shape[0]*self.v), :].unsqueeze(0) # (1, N*v, 512)
                
                m = self.m
                k = self.k
                short_text = self.short_text.to(x.device)
                retrieval_text = process_features(x_plip, short_text, m, k) # (w, 512)
                
                retrieval_text = self.down_proj(retrieval_text)  # (w, d_model)
                retrieval_text = retrieval_text.unsqueeze(0)     # (1, w, d_model)
                
                #update fu with image features and fu_t with retrieved text features
                fu, fu_t, _  = self.layers(x, mask, fu, fu_t, retrieval_text)
                st.append(fu_t)
                s.append(fu)
            
            else: # third and last layers
                # update fu and fu_t with image features and retrieved text features
                fu, fu_t, _ = self.layers(x, mask, fu, fu_t, retrieval_text)
                st.append(fu_t)
                s.append(fu)

        o = s[0]+s[1]+s[2]
        ot = st[0]+st[1]
        o = torch.cat([o, ot], dim=2)        # (B, num_tasks, 512)
 
        return o 
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, src_attn, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.src_attn = src_attn
        self.self_attn = self_attn
        self.feed_forward = clones(feed_forward,2)
        self.sublayer = clones(SublayerConnection(d_model, dropout), 4)
        self.d_model = d_model

    def forward(self, x, mask, fu, fu_t, retrieval_text):
        # cross-attention: query->fu, key/value->x
        # attention_weights: (B, heads, 1, N_keys)
        fu, attention_weights = self.sublayer[0](fu, lambda fu: self.src_attn(fu, x, x, mask=None)) 
        attention_weights = attention_weights.mean(dim=1) #mean along the heads->(B, 1, N_keys)
        attention_weights = attention_weights.mean(dim=1) #mean along the task prompts
        attention_values = attention_weights.squeeze(1) #(B, N_keys)

        # sort indices based on attention values
        sorted_indices = torch.argsort(attention_values, dim=-1, descending=True)  
        
        fu = self.sublayer[1](fu, self.feed_forward[0])
        fu,_ = self.sublayer[2](fu, lambda fu: self.self_attn(fu, fu, fu, mask=None))
        fu = self.sublayer[3](fu, self.feed_forward[1])
        if retrieval_text is not None: #textual part if retrieved_text exists
            # cross-attention: query->fu_t, key/value->retrieval_text
            # attention_weights: (B, heads, 1, N_keys)
            fu_t,_= self.sublayer[0](fu_t, lambda fu_t: self.src_attn(fu_t, retrieval_text, retrieval_text, mask=None))
            fu_t = self.sublayer[1](fu_t, self.feed_forward[0])
            fu_t,_ = self.sublayer[2](fu_t, lambda fu_t: self.self_attn(fu_t, fu_t, fu_t, mask=None))
            fu_t = self.sublayer[3](fu_t, self.feed_forward[1])
      
        return fu, fu_t, sorted_indices

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        norm_x = self.norm(x)
        output = sublayer(norm_x)
       
        if isinstance(output, tuple):
            out, attn = output
            return x + self.dropout(out), attn
        else:
            return x + self.dropout(output)

class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.gamma = nn.Parameter(torch.ones(features))
        self.beta = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.gamma * (x - mean) / (std + self.eps) + self.beta

class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)

    def forward(self, x, hidden_states, src_mask, tgt_mask):
        
        for layer in self.layers:
            x = layer(x, hidden_states, src_mask, tgt_mask)
            # x: (B, seq_lenth-1, 512), hidden_states: (B, 2, 512)
        return self.norm(x)

class DecoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.d_model = d_model
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 3)

    def forward(self, x, hidden_states, src_mask, tgt_mask):
        m = hidden_states
        x ,_= self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x,_ = self.sublayer[1](x, lambda x: self.src_attn(x, m, m))
       
        return self.sublayer[2](x, self.feed_forward)

class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        query, key, value = \
            [l(x).view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for l, x in zip(self.linears, (query, key, value))]

        x, self.attn = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), self.attn

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class Embeddings(nn.Module):
    def __init__(self, d_model, vocab):
        super(Embeddings, self).__init__()
        self.lut = nn.Embedding(vocab, d_model)
        self.d_model = d_model

    def forward(self, x):
        return self.lut(x) * math.sqrt(self.d_model)

class PositionalEncoding(nn.Module):
    def __init__(self, d_model, dropout, max_len=5000):
        super(PositionalEncoding, self).__init__()
        self.dropout = nn.Dropout(p=dropout)

        pe = torch.zeros(max_len, d_model)
        position = torch.arange(0, max_len).unsqueeze(1).float()
        div_term = torch.exp(torch.arange(0, d_model, 2).float() *
                             -(math.log(10000.0) / d_model))
        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)
        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)

    def forward(self, x):
        x = x + self.pe[:, :x.size(1)]
        return self.dropout(x)

class PAM(nn.Module):
    def __init__(self, dim=512):
        super(PAM, self).__init__()
        self.proj = nn.Conv2d(dim, dim, 13, 1, 13//2, groups=dim)
        self.proj1 = nn.Conv2d(dim, dim, 7, 1, 7//2, groups=dim)
        self.proj2 = nn.Conv2d(dim, dim, 3, 1, 3//2, groups=dim)

    def forward(self, x):
        B, H, C = x.shape
        assert int(math.sqrt(H))**2==H, f'{x.shape}'
        cnn_feat = x.transpose(1, 2).view(B, C, int(math.sqrt(H)), int(math.sqrt(H))).contiguous()
        x = self.proj(cnn_feat)+cnn_feat+self.proj1(cnn_feat)+self.proj2(cnn_feat)
        x = x.flatten(2).transpose(1, 2)

        return x


class EncoderClassificator(nn.Module):
    def make_model(self, cfg):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        position = PositionalEncoding(self.d_model, self.dropout)
        pp = PAM(self.d_model)
        backbone = Transformer(
            Encoder(
                EncoderLayer(
                    self.d_model, 
                    c(attn), 
                    c(attn), 
                    c(ff), 
                    self.dropout), 
                    self.num_layers, 
                    pp, 
                    self.bank_path, 
                    self.v, 
                    self.m, 
                    self.k,
                    len(cfg["tasks"])
                    ),
                    self.d_model
            )

        model = MultiTaskClassifier(
            backbone=backbone,
            cfg=cfg,
            d_model=self.d_model*2,
            d_ff=self.d_ff,
            dropout=self.dropout,
            num_prompts_per_task=self.num_prompts_per_task,
            use_BERTPooler= True
        )

        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, cfg):
        super(EncoderClassificator, self).__init__()
        self.args = args
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.bank_path = args.bank_path
        self.v = args.v
        self.m = args.m
        self.k = args.k
        self.cfg = cfg
        self.num_prompts_per_task = args.num_prompts_per_task

        # projector 
        self.input_encoding_size = args.d_model
        self.drop_prob_lm = args.drop_prob_lm
        self.att_feat_size = args.d_vf
        self.use_bn = args.use_bn
        self.att_embed = nn.Sequential(*(
            ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) + # batch norm deactivated
            (nn.Linear(self.att_feat_size, self.input_encoding_size),
                nn.ReLU(),
                nn.Dropout(self.drop_prob_lm)) +
            ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ()))) # batch norm deactivated

        self.model = self.make_model(cfg)
        
    def init_hidden(self, bsz):
        return []

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature_mesh(self, att_feats, att_masks=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks) #(B, N+1, d_model)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        return att_feats, att_masks


    def forward(self, att_feats, att_masks=None):
        att_feats = att_feats["features"]
        att_feats_uni = att_feats[:,:,:-512]  #(B, N+1, 1024)
        att_feats_plip = att_feats[:,:,-512:] #(B, N+1, 512)

        att_feats_uni, att_masks = self._prepare_feature_mesh(att_feats_uni, att_masks)
        # att_feats_uni: (B, N+1, d_model), att_masks: (B, 1, N+1)

        att_feats = torch.cat((att_feats_uni, att_feats_plip), dim=2)  #(B, N+1, d_model+512)
        out = self.model(att_feats, att_masks) #(B, N+1, d_model)

        return {"output": out}