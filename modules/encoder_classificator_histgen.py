from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import math

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


from .att_model import pack_wrapper
from .classification_utils import MultiTaskClassifier

def transform_tokens2regions(hidden_states, num_regions, region_size):

    # transform sequence into regions
    patch_hidden_states = torch.reshape(hidden_states, (hidden_states.size(0), num_regions, region_size, hidden_states.size(-1)))
    # squash regions into sequence into a single axis (samples * regions, region_size, hidden_size)
    hidden_states_reshape = patch_hidden_states.contiguous().view(hidden_states.size(0) * num_regions,
                                                                region_size, patch_hidden_states.size(-1))

    return hidden_states_reshape

def transform_masks2regions(mask, num_regions, region_size):
    
    # transform sequence mask into regions
    patch_mask = torch.reshape(mask, (mask.size(0), num_regions, region_size))
    # squash regions into sequence into a single axis (samples * regions, region_size)
    mask_reshape = patch_mask.contiguous().view(mask.size(0) * num_regions, 1, region_size)

    return mask_reshape

def transform_sentences2tokens(seg_hidden_states, num_sentences, max_sentence_length):
    # transform squashed sequence into segments
    hidden_states = seg_hidden_states.contiguous().view(seg_hidden_states.size(0) // num_sentences, num_sentences,
                                                        max_sentence_length, seg_hidden_states.size(-1))
    # transform segments into sequence
    hidden_states = hidden_states.contiguous().view(hidden_states.size(0), num_sentences * max_sentence_length,
                                                    hidden_states.size(-1))
    return hidden_states

class TransformerLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(TransformerLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)
    
class HATLayer(nn.Module):
    def __init__(self, heads = 8, 
                        d_model = 512, 
                        d_ff = 512, 
                        region_size = 256, 
                        use_region_encoder = True, 
                        use_WSI_encoder = False, 
                        dropout = 0.1, 
                        max_patch = 100000, 
                        first_layer = False,
                        num_tasks=1,
                        ):
        super().__init__()
        self.region_size = region_size
        self.max_patch = max_patch
        self.max_region = int(np.ceil(self.max_patch / self.region_size))
        self.first_layer = first_layer # if is the first HAT layer, interpolate global token and add region level positional encodings
        
        self.d_model = d_model
        self.heads = heads
        self.dropout = dropout
        self.d_ff = d_ff
        self.num_tasks = num_tasks
        
        self.global_tokens = nn.Parameter(torch.randn(1, self.num_tasks, self.d_model))
        self.region_position_embeddings = PositionalEncoding(self.d_model, self.dropout, max_len = 100000)
        
        self.use_region_encoder = use_region_encoder
        self.use_WSI_encoder = use_WSI_encoder
        
        self.attn = MultiHeadedAttention(self.heads, self.d_model, self.dropout)
        self.ff = PositionwiseFeedForward(self.d_model, self.d_ff, self.dropout)
        c = copy.deepcopy

        if self.use_region_encoder:
            self.region_encoder = TransformerLayer(self.d_model, c(self.attn), c(self.ff), self.dropout)

        if self.use_WSI_encoder:
            self.WSI_encoder = TransformerLayer(self.d_model, c(self.attn), c(self.ff), self.dropout)
            self.position_embeddings = nn.Embedding(self.max_region + 1, self.d_model, padding_idx = 0) #! Usage?
    
    def forward(self, x, mask, num_regions):
        
        assert self.use_region_encoder == True or self.use_WSI_encoder == True, "One of the encoders needs to be used"
        
        if self.first_layer: # add global token
            x, mask = self.interpolate_global_token(x, mask) # pad and add global token
            
        if self.use_region_encoder: # True
            region_inputs = transform_tokens2regions(x, num_regions, self.region_size)      #(N_regions, region_size, 512)
            region_masks = transform_masks2regions(mask, num_regions, self.region_size)     #(N_regions, 1, 512)
            region_inputs = self.region_position_embeddings(region_inputs)                  #(N_regions, region_size, 512)
            
            outputs = self.region_encoder(region_inputs, region_masks)                      #(N_regions, region_size, 512)
        else:
            outputs = x
            
        if self.use_WSI_encoder:  # True
            assert self.use_region_encoder == True, "Region encoder needs to be used before WSI encoder"                                                   #(N_regions, 1, 512)
            B = x.size(0)
            R = self.region_size
            T = self.num_tasks

            outputs_r = outputs.view(B, num_regions, R, self.d_model)

            # take task tokens (last T positions)
            task_tokens = outputs_r[:, :, -T:, :]        # (B, num_regions, T, D)

            # flatten for WSI encoder
            task_tokens = task_tokens.reshape(B, num_regions * T, self.d_model)

            task_mask = mask.view(B, 1, num_regions, R)[:, :, :, -T:]
            task_mask = task_mask.contiguous().view(B, 1, num_regions * T)

            #* Relative Positional Encoding
            task_tokens = self.region_position_embeddings(task_tokens)                                    #(1, N_regions, 512)

            #*
            WSI_outputs = self.WSI_encoder(task_tokens, task_mask)                                        #(1, N_regions, 512)
            WSI_outputs = WSI_outputs.view(WSI_outputs.size(1), 1, WSI_outputs.size(2))                   #(N_regions, 1, 512)


            # put back
            task_tokens = task_tokens.view(B, num_regions, T, self.d_model)         # WSI_outputs.view(B, num_regions, T, self.d_model)
            outputs_r[:, :, -T:, :] = task_tokens
            outputs = outputs_r.view(B, num_regions * R, self.d_model)

        else:
            outputs = outputs.view(x.size(0), num_regions * self.region_size, outputs.size(-1))
        
        outputs_mask = mask
        
        return outputs, outputs_mask
    
    def interpolate_global_token(self, hidden_states, mask):
        """
        This function takes a sequence of tokens (B, seq_len, D) and splits it into regions.
        Each region contains:
            -> region_size - 1 original tokens
            -> num_tasks global tokens added at the end of the region.
        If the sequence length doesn't fit perfectly, it adds padding.
        The same process is applied to the mask, marking the global tokens as valid.
        The output is a longer, structured sequence where tokens are grouped into regions, each with its own summary (global) token.
        """
        batch_size, seq_len, hidden_dim = hidden_states.size()
        T = self.num_tasks
        R = self.region_size

        patch_per_region = R - T
        num_regions = int(np.ceil(seq_len / patch_per_region))

        #* Calculate the total size after division and padding
        total_patch_tokens = num_regions * patch_per_region

        #* Calculate the padding size
        padding_size = total_patch_tokens - seq_len

        #* Pad the sequence and mask if needed
        if padding_size > 0:
            hidden_padding = torch.zeros(batch_size, padding_size, hidden_dim, device=hidden_states.device)
            hidden_states = torch.cat([hidden_states, hidden_padding], dim=1)

            mask_padding = torch.zeros(batch_size, 1, padding_size, device=mask.device)
            mask = torch.cat([mask, mask_padding], dim=2)

        # reshape patches into regions
        patch_tokens = hidden_states.view(batch_size, num_regions, patch_per_region, hidden_dim)

        # task tokens: (B, num_regions, T, D)
        task_tokens = self.global_tokens.repeat(batch_size, num_regions, 1)
        task_tokens = task_tokens.view(batch_size, num_regions, T, hidden_dim)
        
        # concatenate -> (B, num_regions, R, D)
        hidden_states = torch.cat([patch_tokens, task_tokens], dim=2)
        hidden_states = hidden_states.view(batch_size, num_regions * self.region_size, hidden_dim)

        # mask
        patch_mask = mask.view(batch_size, 1, num_regions, patch_per_region)
        task_mask = torch.ones(batch_size, 1, num_regions, T, device=mask.device)

        mask = torch.cat([patch_mask, task_mask], dim=3)
        mask = mask.view(batch_size, 1, num_regions * R)

        # return hidden_states_with_global, mask_with_global
        return hidden_states, mask

    
class HATEncoder(nn.Module):
    def __init__(self, encoder_layout = None):
        super().__init__()
        self.encoder_layout = encoder_layout
        self.layer = nn.ModuleList([HATLayer(heads = encoder_layout['num_heads'], d_model = encoder_layout['d_model'], d_ff = encoder_layout['d_ff'], 
                                             region_size = encoder_layout['region_size'], use_region_encoder = encoder_layout[str(idx)]['region_encoder'], 
                                             use_WSI_encoder = encoder_layout[str(idx)]['WSI_encoder'], dropout = encoder_layout['dropout'], 
                                             first_layer = encoder_layout[str(idx)]['first_layer'], num_tasks = encoder_layout['num_tasks']) for idx in range(int(encoder_layout['num_layers']))])
        self.norm = LayerNorm(encoder_layout['d_model'])
        self.pooler = HATPooler(encoder_layout, 
                                pooling = encoder_layout['pooling']
                                )
        self.region_size = encoder_layout['region_size']
        self.num_tasks = encoder_layout['num_tasks']
        
    def forward(self, x, mask):
        
        num_regions = int(np.ceil(x.shape[1] / (self.region_size - self.num_tasks)))         
        
        for idx, layer in enumerate(self.layer):
            x, mask = layer(x, mask, num_regions)
        x = self.norm(x)
        
        #* Take out the global token of each region and send to the decoder
        x = x.view(num_regions, self.region_size, x.size(-1))
        task_tokens = x[:, -self.num_tasks:, :]   # (num_regions, T, D)
        task_tokens = task_tokens.transpose(0, 1)  # (T, R, D)
        pooled_tasks = self.pooler(task_tokens)     # (T, D)
        return pooled_tasks.unsqueeze(0)             # (1, T, D)

    
class AttentivePooling(nn.Module):
    def __init__(self, encoder_layout):
        super().__init__()
        self.attn_dropout = encoder_layout['dropout']
        self.lin_proj = nn.Linear(encoder_layout['d_model'], encoder_layout['d_model'])
        self.v = nn.Linear(encoder_layout['d_model'], 1, bias=False)

    def forward(self, inputs):
        lin_out = self.lin_proj(inputs)
        attention_weights = torch.tanh(self.v(lin_out)).squeeze(-1)
        attention_weights_normalized = torch.softmax(attention_weights, -1)
        return torch.sum(attention_weights_normalized.unsqueeze(-1) * inputs, 1)


class HATPooler(nn.Module):
    def __init__(self, encoder_layout, pooling = 'max'):
        super().__init__()
        self.dense = nn.Linear(encoder_layout['d_model'], encoder_layout['d_model'])
        self.pooling = pooling
        if self.pooling == 'attentive':
            self.attentive_pooling = AttentivePooling(encoder_layout)        
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        if self.pooling == 'attentive':
            pooled_output = self.attentive_pooling(hidden_states)
        elif self.pooling == 'max':
            pooled_output = torch.max(hidden_states, dim=1)[0]

        pooled_output = self.dense(pooled_output)
        pooled_output = self.activation(pooled_output)
        return pooled_output


def clones(module, N):
    return nn.ModuleList([copy.deepcopy(module) for _ in range(N)])


def subsequent_mask(size):
    attn_shape = (1, size, size)
    subsequent_mask = np.triu(np.ones(attn_shape), k=1).astype('uint8')
    return torch.from_numpy(subsequent_mask) == 0


def attention(query, key, value, mask=None, dropout=None):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    p_attn = F.softmax(scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn, value), p_attn


def memory_querying_responding(query, key, value, mask=None, dropout=None, topk=32):
    d_k = query.size(-1)
    scores = torch.matmul(query, key.transpose(-2, -1)) / math.sqrt(d_k)
    if mask is not None:
        scores = scores.masked_fill(mask == 0, float('-inf'))
    selected_scores, idx = scores.topk(topk)
    dummy_value = value.unsqueeze(2).expand(idx.size(0), idx.size(1), idx.size(2), value.size(-2), value.size(-1))
    dummy_idx = idx.unsqueeze(-1).expand(idx.size(0), idx.size(1), idx.size(2), idx.size(3), value.size(-1))
    selected_value = torch.gather(dummy_value, 3, dummy_idx)
    p_attn = F.softmax(selected_scores, dim=-1)
    if dropout is not None:
        p_attn = dropout(p_attn)
    return torch.matmul(p_attn.unsqueeze(3), selected_value).squeeze(3), p_attn


class Transformer(nn.Module):
    def __init__(self, encoder):
        super(Transformer, self).__init__()
        self.encoder = encoder

    def forward(self, src, src_mask):
        return self.encode(src, src_mask)

    def encode(self, src, src_mask):
        return self.encoder(src, src_mask)


class Encoder(nn.Module):
    def __init__(self, layer, N):
        super(Encoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, mask):
        for layer in self.layers:
            x = layer(x, mask)
        return self.norm(x)


class LayerNorm(nn.Module):
    def __init__(self, features, eps=1e-6):
        super(LayerNorm, self).__init__()
        self.a_2 = nn.Parameter(torch.ones(features))
        self.b_2 = nn.Parameter(torch.zeros(features))
        self.eps = eps

    def forward(self, x):
        mean = x.mean(-1, keepdim=True)
        std = x.std(-1, keepdim=True)
        return self.a_2 * (x - mean) / (std + self.eps) + self.b_2


class SublayerConnection(nn.Module):
    def __init__(self, size, dropout):
        super(SublayerConnection, self).__init__()
        self.norm = LayerNorm(size)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x, sublayer):
        _x = sublayer(self.norm(x))
        if type(_x) is tuple:
            return x + self.dropout(_x[0]), _x[1]
        return x + self.dropout(_x)


class EncoderLayer(nn.Module):
    def __init__(self, size, self_attn, feed_forward, dropout):
        super(EncoderLayer, self).__init__()
        self.self_attn = self_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 2)
        self.size = size

    def forward(self, x, mask):
        x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, mask))
        return self.sublayer[1](x, self.feed_forward)


class Decoder(nn.Module):
    def __init__(self, layer, N):
        super(Decoder, self).__init__()
        self.layers = clones(layer, N)
        self.norm = LayerNorm(layer.size)

    def forward(self, x, memory, src_mask, tgt_mask, past=None):
        if past is not None:
            present = [[], []]
            x = x[:, -1:]
            tgt_mask = tgt_mask[:, -1:] if tgt_mask is not None else None
            past = list(zip(past[0].split(2, dim=0), past[1].split(2, dim=0)))
        else:
            past = [None] * len(self.layers)
        for i, (layer, layer_past) in enumerate(zip(self.layers, past)):
            x = layer(x, memory, src_mask, tgt_mask,
                      layer_past)
            if layer_past is not None:
                present[0].append(x[1][0])
                present[1].append(x[1][1])
                x = x[0]
        if past[0] is None:
            return self.norm(x)
        else:
            return self.norm(x), [torch.cat(present[0], 0), torch.cat(present[1], 0)]


class DecoderLayer(nn.Module):
    def __init__(self, size, self_attn, src_attn, feed_forward, dropout):
        super(DecoderLayer, self).__init__()
        self.size = size
        self.self_attn = self_attn
        self.src_attn = src_attn
        self.feed_forward = feed_forward
        self.sublayer = clones(SublayerConnection(size, dropout), 3)

    def forward(self, x, memory, src_mask, tgt_mask, layer_past=None):
        m = memory
        src_mask = m.new_ones(m.shape[:2], dtype=torch.long)
        src_mask = src_mask.unsqueeze(-2)
        if layer_past is None:
            x = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask))
            x = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask))
            return self.sublayer[2](x, self.feed_forward)
        else:
            present = [None, None]
            x, present[0] = self.sublayer[0](x, lambda x: self.self_attn(x, x, x, tgt_mask, layer_past[0]))
            x, present[1] = self.sublayer[1](x, lambda x: self.src_attn(x, m, m, src_mask, layer_past[1]))
            return self.sublayer[2](x, self.feed_forward), present


class MultiThreadMemory(nn.Module):
    def __init__(self, h, d_model, dropout=0.1, topk=32):
        super(MultiThreadMemory, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)
        self.topk = topk

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)

        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]
        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = memory_querying_responding(query, key, value, mask=mask, dropout=self.dropout, topk=self.topk)

        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


class MultiHeadedAttention(nn.Module):
    def __init__(self, h, d_model, dropout=0.1):
        super(MultiHeadedAttention, self).__init__()
        assert d_model % h == 0
        self.d_k = d_model // h
        self.h = h
        self.linears = clones(nn.Linear(d_model, d_model), 4)
        self.attn = None
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, query, key, value, mask=None, layer_past=None):
        if mask is not None:
            mask = mask.unsqueeze(1)
        nbatches = query.size(0)
        if layer_past is not None and layer_past.shape[2] == key.shape[1] > 1:
            query = self.linears[0](query)
            key, value = layer_past[0], layer_past[1]
            present = torch.stack([key, value])
        else:
            query, key, value = \
                [l(x) for l, x in zip(self.linears, (query, key, value))]

        if layer_past is not None and not (layer_past.shape[2] == key.shape[1] > 1):
            past_key, past_value = layer_past[0], layer_past[1]
            key = torch.cat((past_key, key), dim=1)
            value = torch.cat((past_value, value), dim=1)
            present = torch.stack([key, value])

        query, key, value = \
            [x.view(nbatches, -1, self.h, self.d_k).transpose(1, 2)
             for x in [query, key, value]]

        x, self.attn = attention(query, key, value, mask=mask,
                                 dropout=self.dropout)
        x = x.transpose(1, 2).contiguous() \
            .view(nbatches, -1, self.h * self.d_k)
        if layer_past is not None:
            return self.linears[-1](x), present
        else:
            return self.linears[-1](x)


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
    def __init__(self, d_model, dropout, max_len=70000):
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


class BaseHistGenClassification(nn.Module):
    def make_model(self, args, cfg, encoder_layout = None):
        encoder_layout["num_tasks"] = len(cfg["tasks"])

        backbone = Transformer(
            HATEncoder(encoder_layout),
        )
        model = MultiTaskClassifier(
            backbone=backbone,
            cfg=cfg,
            d_model=self.d_model,
            d_ff=self.d_ff,
            dropout=self.dropout,
            num_prompts_per_task=args.num_prompts_per_task,
        )
        
        for p in model.parameters():
            if p.dim() > 1:
                nn.init.xavier_uniform_(p)
        return model

    def __init__(self, args, cfg):
        super(BaseHistGenClassification, self).__init__()
        self.args = args
        self.cfg = cfg
        self.num_layers = args.num_layers
        self.d_model = args.d_model
        self.d_ff = args.d_ff
        self.num_heads = args.num_heads
        self.dropout = args.dropout
        self.topk = args.topk
        self.K = args.prototype_num

        self.cmn = MultiThreadMemory(args.num_heads, args.d_model, topk=args.topk)
        
        self.region_size = args.region_size
        self.encoder_layout = {
                'num_heads': self.num_heads,
                'd_model': self.d_model,
                'd_ff': self.d_ff,
                'region_size': self.region_size,
                'dropout': self.dropout,
                'pooling': 'attentive',
                'num_layers': 2,
                '0': {
                    'region_encoder': True,
                    'WSI_encoder': True,
                    'first_layer': True
                },
                '1': {
                    'region_encoder': True,
                    'WSI_encoder': False,
                    'first_layer': False
                },
            }
            
        self.input_encoding_size = args.d_model
        self.att_feat_size = args.d_vf
        self.use_bn = args.use_bn
        self.drop_prob_lm = args.drop_prob_lm
        self.att_embed = nn.Sequential(*(
            ((nn.BatchNorm1d(self.att_feat_size),) if self.use_bn else ()) + # batch norm deactivated
            (nn.Linear(self.att_feat_size, self.input_encoding_size),
                nn.ReLU(),
                nn.Dropout(self.drop_prob_lm)) +
            ((nn.BatchNorm1d(self.input_encoding_size),) if self.use_bn == 2 else ()))) # batch norm deactivated

        self.model = self.make_model(self.args, self.cfg, self.encoder_layout)

        self.memory_matrix = nn.Parameter(torch.FloatTensor(args.cmm_size, args.cmm_dim))
        nn.init.normal_(self.memory_matrix, 0, 1 / args.cmm_dim)
        
        self.attn_mem = MultiHeadedAttention(self.num_heads, self.d_model)

    def init_hidden(self, bsz):
        return []

    def _prepare_feature(self, fc_feats, att_feats, att_masks):
        att_feats, seq, att_masks, seq_mask = self._prepare_feature_forward(att_feats, att_masks)
        memory = self.model.encode(att_feats, att_masks)

        return fc_feats[..., :1], att_feats[..., :1], memory, att_masks

    def clip_att(self, att_feats, att_masks):
        # Clip the length of att_masks and att_feats to the maximum length
        if att_masks is not None:
            max_len = att_masks.data.long().sum(1).max()
            att_feats = att_feats[:, :max_len].contiguous()
            att_masks = att_masks[:, :max_len].contiguous()
        return att_feats, att_masks

    def _prepare_feature_forward(self, att_feats, att_masks=None):
        att_feats, att_masks = self.clip_att(att_feats, att_masks)
        att_feats = pack_wrapper(self.att_embed, att_feats, att_masks)              #(B, N_patches, 512)

        if att_masks is None:
            att_masks = att_feats.new_ones(att_feats.shape[:2], dtype=torch.long)

        #* Memory querying and responding for visual features
        dummy_memory_matrix = self.memory_matrix.unsqueeze(0).expand(att_feats.size(0), self.memory_matrix.size(0), self.memory_matrix.size(1))
        indices = torch.linspace(0, att_feats.shape[1] - 1, steps=self.K).long()    #(512)
        M = att_feats[:, indices, :]                                                #(B, 512, 512)
        responses = self.cmn(M, dummy_memory_matrix, dummy_memory_matrix)           #(B, 512, 512)
        response_mask = responses.new_ones(responses.shape[:2], dtype=torch.long)   
        response_mask = response_mask.unsqueeze(-2)                                 #(B, 1, 512)
        att_feats = att_feats + self.attn_mem(att_feats, responses, responses, response_mask)   #(B, N_patches, 512)
        #*

        att_masks = att_masks.unsqueeze(-2)                                         #(B, 1, 512)

        return att_feats, att_masks

    def forward(self, att_feats, att_masks=None):
        att_feats = att_feats["features"]
        att_feats, att_masks = self._prepare_feature_forward(att_feats, att_masks)

        out = self.model(att_feats, att_masks)

        return {"output": out}

    def _save_attns(self, start=False):
        if start:
            self.attention_weights = []
        self.attention_weights.append([layer.src_attn.attn.cpu().numpy() for layer in self.model.decoder.layers])

