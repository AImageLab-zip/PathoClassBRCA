import torch
from torch.optim.lr_scheduler import LambdaLR, StepLR
import math

def build_optimizer(args, model):

    ed_params = filter(lambda x: id(x), model.parameters())
    optimizer = getattr(torch.optim, args.optim)(
        [ {'params': ed_params, 'lr': args.lr_ed}],
        weight_decay=args.weight_decay,
        amsgrad=args.amsgrad
    )
    return optimizer

def build_lr_scheduler(args, optimizer, num_training_steps: int):
    if args.lr_scheduler == "StepLR":
        if args.last_epoch != -1:
            # Ensure initial_lr exists when resuming
            for g in optimizer.param_groups:
                g.setdefault("initial_lr", g["lr"])

            print(f"Resuming: last_epoch={args.last_epoch} for StepLR")
            return StepLR(optimizer, step_size=args.step_size, gamma=args.gamma, last_epoch=args.last_epoch)
        else:
            return StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)

    if args.lr_scheduler == "warmup_cosine":
        
        if args.warmup_steps > 0:
            num_warmup_steps = args.warmup_steps
        else:
            num_warmup_steps = int(args.warmup_ratio * num_training_steps)

        def lr_lambda(current_step: int):
            # warmup lineare
            if current_step < num_warmup_steps:
                return float(current_step) / float(max(1, num_warmup_steps))

            # progress per il cosine
            progress = float(current_step - num_warmup_steps) / \
                       float(max(1, num_training_steps - num_warmup_steps))
            progress = min(max(progress, 0.0), 1.0)

            return  0.5 * (1.0 + math.cos(math.pi * progress))

        return LambdaLR(optimizer, lr_lambda)
    
    if args.lr_scheduler == "OneCycleLR":
        return torch.optim.lr_scheduler.OneCycleLR(
            optimizer,
            max_lr=args.lr_ed,
            total_steps=num_training_steps,
            pct_start=args.warmup_ratio,
            anneal_strategy='cos',
            div_factor=25.0,
            final_div_factor=1e4,
        )

    raise ValueError(f"Unknown lr_scheduler: {args.lr_scheduler}")
