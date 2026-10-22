from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import copy
import math
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from .att_model import pack_wrapper, AttModel
from .classification_utils import MultiTaskClassifier, TaskHead

def clip_att(att_feats, att_masks):
    # Clip the length of att_masks and att_feats to the maximum length
    if att_masks is not None:
        max_len = att_masks.data.long().sum(1).max()
        att_feats = att_feats[:, :max_len].contiguous()
        att_masks = att_masks[:, :max_len].contiguous()
    return att_feats, att_masks

def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])

def pad_tokens(att_feats):
    # ---->pad
    H = att_feats.shape[1]
    _H, _W = int(np.ceil(np.sqrt(H))), int(np.ceil(np.sqrt(H)))
    add_length = _H * _W - H
    att_feats = torch.cat([att_feats, att_feats[:, :add_length, :]], dim=1)  # [B, N, L]
    return att_feats

def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, -1e9)
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn.detach().cpu()

def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0

class Transformer(nn.Module):
    def __init__(self, encoder):
        super(Transformer, self).__init__()
        self.encoder = encoder


    def forward(self, src, src_mask):
        return self.encode(src, src_mask)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)
   
class Encoder(nn.Module):
    def __init__(self, layer, N, PAM):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.d_model)
        self.PAM = clones(PAM, N)
        self.N = N
        
    def forward(self, x, mask):
        s=[]
        attn_scores = []
        for i, layer in enumerate(self.layers): #3 layers
            x, attn_score = layer(x, mask)       # x: (B, N_patches, 512)
            s.append(self.PAM[i](x)) # x: (B, N_patches, 512)
            attn_scores.append(attn_score)
        
        o = s[0]                     # o: (B, N_patches, 512)
        for i in range(1, len(s)):
            o += s[i]
        return {"att_feats": o, "attn_scores": attn_scores}
        
class EncoderLayer(nn.Module):
    def __init__(self, d_model, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(d_model, dropout), 2)
        self.d_model = d_model

    def forward(self, x, mask):
        x, attn_score = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward), attn_score

class SublayerConnection(nn.Module):
    def __init__(self, d_model, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        # return x + self.dropout(sublayer(self.norm(x)))
        out = sublayer(self.norm(x))
        if isinstance(out, tuple):
            att_feats, att_scores = out
            return x + self.dropout(att_feats), att_scores
        else:
            att_feats = out
            return x + self.dropout(att_feats)

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
            # x: (B, seq_length-1, 512), hidden_states: (B, N_patches, 512)
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
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
        x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
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

        x, att_score = attention(query, key, value, mask=mask, dropout=self.dropout)

        x = x.transpose(1, 2).contiguous().view(nbatches, -1, self.h * self.d_k)
        return self.linears[-1](x), att_score

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
    

class MultiTaskClassifier(MultiTaskClassifier):
    def adapter_forward(self, backbone_output):
        return backbone_output
    
    def forward(self, att_feats, att_masks=None):
        out = self.backbone(att_feats, att_masks)   # (B, N+1, d_model)
        out = self.adapter_forward(out)  # (B, N+1, d_model)
        output = {}
        for i, task_name in enumerate(self.cfg["tasks"].keys()):
            head = self.heads[task_name]
            start_idx = i * self.num_prompts_per_task
            end_idx = start_idx + self.num_prompts_per_task
            pooled = out["att_feats"][:, start_idx:end_idx, :] # (B, num_prompts_per_task, d_model)
            if self.BERTPooler is not None:
                pooled = self.BERTPooler(pooled)  # (B, d_model)
            output[task_name] = head(pooled)  # logits (B, num_classes)
        return {"output": output, "attn_scores": out["attn_scores"]}

class EncoderClassificator(nn.Module):
    def make_model(self, cfg):
        c = copy.deepcopy
        attn = MultiHeadedAttention(self.num_heads, self.d_model)
        ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        pp = PAM(self.d_model)

        backbone = Transformer(
            Encoder(
                EncoderLayer(self.d_model, c(attn), c(ff), self.dropout),
                self.num_layers,
                pp
            ),
        )
        
        model = MultiTaskClassifier(
            backbone=backbone,
            cfg=cfg,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            num_prompts_per_task=self.num_prompts_per_task,
            use_BERTPooler=None
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
            ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ()))) 
    
        self.prompts = nn.ParameterDict({
            task_name: nn.Parameter(
                torch.randn(1, self.num_prompts_per_task, self.input_encoding_size)
            )
            for task_name in self.cfg["tasks"]
        })

        self.model = self.make_model(cfg)
        
    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, att_feats, att_masks=None, prompts=None):
        
        att_feats, att_masks = clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)

        if prompts is not None:
            att_feats = torch.cat([prompts, att_feats], dim=1)

        att_feats = pad_tokens(att_feats)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)
        att_masks = att_masks.unsqueeze(-2)

        return att_feats, att_masks

    def forward(self, att_feats, att_masks=None):
        att_feats = att_feats["features"]

        # prepend task-specific prompts
        batch_size = att_feats.size(0)
        prompts = torch.cat([self.prompts[task_name] for task_name in self.cfg["tasks"].keys()], dim=1)
        prompts = prompts.expand(batch_size, -1, -1)

        att_feats, att_masks = self._prepare_feature(att_feats, att_masks, prompts)
        # att_feats = (B, N_patches+1, 512), att_masks=(B, 1, N_patches+1)

        out = self.model(att_feats, att_masks)

        return out