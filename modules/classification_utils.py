import torch
import torch.nn as nn
import torch.nn.functional as F
# from modules.encoder_classificator_migen import PositionwiseFeedForward

class PositionwiseFeedForward(nn.Module):
    def __init__(self, d_model, d_ff, dropout=0.1):
        super(PositionwiseFeedForward, self).__init__()
        self.w_1 = nn.Linear(d_model, d_ff)
        self.w_2 = nn.Linear(d_ff, d_model)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        return self.w_2(self.dropout(F.relu(self.w_1(x))))

class BERTPooler(nn.Module):
    def __init__(self, d_model):
        super(BERTPooler, self).__init__()
        self.dense = nn.Linear(d_model, d_model)
        self.activation = nn.Tanh()

    def forward(self, hidden_states):
        # We "pool" the model by simply taking the hidden state corresponding
        # assume the hidden state include only CLS tokens
        pooled_output = self.dense(hidden_states)
        pooled_output = self.activation(pooled_output)
        return pooled_output

class TaskHead(nn.Module):
    def __init__(self, 
                d_model, 
                d_ff, 
                dropout, 
                num_classes, 
                num_prompts_per_task):
        super().__init__()
        self.ffn = PositionwiseFeedForward(d_model, d_ff, dropout)  # (B, d_model) -> (B, d_model)
        self.classifier = nn.Linear(d_model, num_classes)

    def forward(self, x):
        x = x.mean(dim=1)  # (B, d_model)
        x = self.ffn(x)
        return self.classifier(x)  # logits (B, num_classes)

class MultiTaskClassifier(nn.Module):
    def __init__(self, 
                    backbone, 
                    cfg, 
                    d_model, 
                    d_ff, 
                    dropout, 
                    num_prompts_per_task,
                    use_BERTPooler=None):
        super().__init__()
        self.backbone = backbone
        self.cfg = cfg
        self.num_prompts_per_task = num_prompts_per_task
        self.num_tasks = len(cfg["tasks"])
        self.BERTPooler = BERTPooler(d_model) if use_BERTPooler is not None else None
        self.heads = nn.ModuleDict({
            task_name: TaskHead(
                d_model=d_model,
                d_ff=d_ff,
                dropout=dropout,
                num_classes=task_cfg["num_classes"],
                num_prompts_per_task=num_prompts_per_task
            )
            for task_name, task_cfg in cfg["tasks"].items()
        })

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
            pooled = out[:, start_idx:end_idx, :] # (B, num_prompts_per_task, d_model)
            if self.BERTPooler is not None:
                pooled = self.BERTPooler(pooled)  # (B, d_model)
            output[task_name] = head(pooled)  # logits (B, num_classes)
        return output