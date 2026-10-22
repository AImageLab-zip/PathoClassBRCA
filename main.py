import torch
import torch.distributed as dist
from torch.nn.parallel import DistributedDataParallel as DDP
import torch.multiprocessing as mp
import os
import argparse
from modules.tokenizers import Tokenizer
from modules.dataloaders import R2DataLoader, ClassificationDataLoader
from modules.metrics import compute_scores, compute_scores_classification
from modules.optimizers import build_optimizer, build_lr_scheduler
from utils.utils import init_seeds
from modules.trainer import Trainer
from modules.loss import compute_loss, compute_class_weights_multitask
import warnings
warnings.filterwarnings("ignore")
import random
import wandb
import sys
import math
from datetime import date, datetime
import time
from hashlib import shake_256
import yaml

# Baselines Report Generation
from models.bigen import BiGenModel
from models.histgen_model import HistGenModel
from models.migen import MiGenModel

# Baselines Classification
from models.migen_classification import MiGenModelClassification
from models.histgen_classification import HistGenModelClassification
from models.bigen_classification import BiGenModelClassification

# torch.autograd.set_detect_anomaly(True)
sys.path.append('./')

def timehash():
    t = time.time()
    t = str(t).encode()
    h = shake_256(t)
    h = h.hexdigest(5)  # output len: 2*5=10
    return h.upper()

def parse_args():
    parser = argparse.ArgumentParser()

    # Data input settings
    parser.add_argument('--image_dir', type=str, default='.../pt_files', help='the path to the directory containing the encoded wsi patches using UNI.')
    parser.add_argument('--image_dir_plip', type=str, default=None, help='the path to the directory containing the encoded wsi patches using PLIP.')
    parser.add_argument('--ann_path', type=str, default='.../TCGA_BRCA', help='the path to the directory containing the data.')
    parser.add_argument('--split_path', type=str, default='../ocr/dataset_csv/splits_3.csv', help='the path to the directory containing the train/val/test splits.')
    parser.add_argument('--bank_path', type=str, default='../memory_short.pt', help='the path to the directory containing the knowledge bank')
    parser.add_argument('--model_name', type=str, default='bigen', choices=['bigen', 
                                                                            'migen', 
                                                                            'histgen', 
                                                                            'wsi_llava', 
                                                                            ], help='model used for experiment')
    parser.add_argument('--train_json', type=str, default=None, help='the path to the training json file (for JSON-based dataset).')
    parser.add_argument('--test_json', type=str, default=None, help='the path to the test json file (for JSON-based dataset).')
    parser.add_argument('--val_json', type=str, default=None, help='the path to the test json file (for JSON-based dataset).')
    parser.add_argument("--device", type=str, default="cuda")
    
    # Data loader settings
    parser.add_argument('--dataset_name', type=str, default='TCGA', choices=['TCGA',], help='the dataset to be used.')
    parser.add_argument('--max_fea_length', type=int, default=10000, help='the maximum sequence length of the patch embeddings.')
    parser.add_argument('--max_seq_length', type=int, default=600, help='the maximum sequence length of the reports.')
    parser.add_argument('--threshold', type=int, default=3, help='the cut off frequency for the words.')
    parser.add_argument('--num_workers', type=int, default=2, help='the number of workers for dataloader.')
    parser.add_argument('--batch_size', type=int, default=1, help='the number of samples for a batch.')
    parser.add_argument('--use_json_dataset', action="store_true", help='whether to use the JSON-based dataset class')
    parser.add_argument('--classification', action="store_true")
    parser.add_argument("--tasks", nargs="+", type=str, default=['diagnosis'], help="List of tasks (e.g. diagnosis grade)")

    # Model settings (for Transformer)
    parser.add_argument('--d_model', type=int, default=512, help='the dimension of Transformer.')
    parser.add_argument('--d_ff', type=int, default=512, help='the dimension of FFN.')
    parser.add_argument('--d_vf', type=int, default=1024, help='the dimension of the patch features.')
    parser.add_argument('--num_heads', type=int, default=4, help='the number of heads in Transformer.')
    parser.add_argument('--num_layers', type=int, default=3, help='the number of layers of Transformer.')
    parser.add_argument('--dropout', type=float, default=0.1, help='the dropout rate of Transformer.')
    parser.add_argument('--logit_layers', type=int, default=1, help='the number of the logit layer.')
    parser.add_argument('--bos_idx', type=int, default=0, help='the index of <bos>.')
    parser.add_argument('--eos_idx', type=int, default=0, help='the index of <eos>.')
    parser.add_argument('--pad_idx', type=int, default=0, help='the index of <pad>.')
    parser.add_argument('--use_bn', type=int, default=0, help='whether to use batch normalization.')
    parser.add_argument('--drop_prob_lm', type=float, default=0.5, help='the dropout rate of the output layer.')
    parser.add_argument('--n_classes', type=int, default=2, help='how many classes to predict')
    parser.add_argument('--num_prompts_per_task', type=int, default=1, help='how many prompts per classification task')
    parser.add_argument('--save_attn_scores', action='store_true', help='save attention scores to disk during evaluation')
    
        
    # Baseline settings        
    # Memory settings (HistGen)
    parser.add_argument('--topk', type=int, default=32, help='the number of k.')
    parser.add_argument('--cmm_size', type=int, default=2048, help='the numebr of cmm size.')
    parser.add_argument('--cmm_dim', type=int, default=512, help='the dimension of cmm dimension.')
    
    # Local-global hierachical visual encoder settings (HistGen)
    parser.add_argument("--region_size", type=int, default=256, help="the size of the region for region transformer.")
    parser.add_argument("--prototype_num", type=int, default=512, help="the number of visual prototypes for cross-modal interaction")

    # BiGen
    parser.add_argument('--v', type=float, default=0.4, help='the ratio of selected top-v knowledge features for each region based on the similarity in the knowledge retrieval module')
    parser.add_argument('--m', type=int, default=10, help='the region size in the knowledge retrieval module')
    parser.add_argument('--k', type=int, default=3, help='the number of selected top-k features with the highest attention scores in the knowledge retrieval module')


    # Sample related
    parser.add_argument('--sample_method', type=str, default='beam_search', help='the sample methods to sample a report.')
    parser.add_argument('--beam_size', type=int, default=3, help='the beam size when beam searching.')
    parser.add_argument('--temperature', type=float, default=1.0, help='the temperature when sampling.')
    parser.add_argument('--sample_n', type=int, default=1, help='the sample number per image.')
    parser.add_argument('--group_size', type=int, default=1, help='the group size.')
    parser.add_argument('--output_logsoftmax', type=int, default=1, help='whether to output the probabilities.')
    parser.add_argument('--decoding_constraint', type=int, default=1, help='whether decoding constraint.')
    parser.add_argument('--suppress_UNK', type=int, default=1, help='suppress UNK tokens in the decoding.')
    parser.add_argument('--block_trigrams', type=int, default=1, help='whether to use block trigrams.')

    # Trainer settings
    parser.add_argument('--n_gpu', type=str, default='1', help='the  gpus to be used.')
    parser.add_argument('--epochs', type=int, default=60, help='the number of training epochs.')
    parser.add_argument('--epochs_val', type=int, default=2, help='interval between eval epochs')
    parser.add_argument('--start_val', type=int, default=0, help='start eval epochs')
    parser.add_argument('--save_dir', type=str, default='/results/', help='the patch to save the models.')
    parser.add_argument('--record_dir', type=str, default='records/', help='the patch to save the results of experiments')
    parser.add_argument('--save_period', type=int, default=1, help='the saving period.')
    parser.add_argument('--monitor_mode', type=str, default='max', choices=['min', 'max'], help='whether to max or min the metric.')
    parser.add_argument('--monitor_metric', type=str, default='BLEU_4', help='the metric to be monitored.')
    parser.add_argument('--early_stop', type=int, default=20, help='the patience of training.')
    parser.add_argument('--accum_steps', type=int, default=1, help='number of gradient accumulation steps.')

    # Optimization
    parser.add_argument('--optim', type=str, default='Adam', help='the type of the optimizer.')
    parser.add_argument('--lr_ed', type=float, default=1e-4, help='the learning rate for the remaining parameters.')
    parser.add_argument('--weight_decay', type=float, default=5e-5, help='the weight decay.')
    parser.add_argument("--amsgrad", action="store_false")

    # Learning Rate Scheduler
    parser.add_argument('--lr_scheduler', type=str, default='StepLR', help='the type of the learning rate scheduler.')
    parser.add_argument('--step_size', type=int, default=50, help='the step size of the learning rate scheduler.')
    parser.add_argument('--gamma', type=float, default=0.1, help='the gamma of the learning rate scheduler.')
    parser.add_argument('--warmup_ratio', type=float, default=0.1, help='fraction of total training steps used for warmup')
    parser.add_argument('--warmup_steps', type=int, default=1000, help='number of warmup steps')
    parser.add_argument("--last_epoch", type=int, default=-1, help='epoch to start training (useful for resuming LR scheduler)')
    
    # debug
    parser.add_argument("--checkpoint_dir", type=str, default='')
    parser.add_argument("--mode", type=str, default='Train')
    parser.add_argument("--local_rank", type=int, default=-1)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--masking_fixed", action="store_true", help="whether to exlude padding from sequence mask for padded reports")
    parser.add_argument("--use_padding", action="store_true")
    
    # Others
    parser.add_argument('--config_path', type=str, default='config/config_BRCA.yaml', help='the path to the configuration file.')
    parser.add_argument('--seed', type=int, default=9233, help='.')
    parser.add_argument('--resume', type=str, help='whether to resume the training from existing checkpoints.')
    parser.add_argument('--resume_last', action="store_true", help='whether to resume from the last checkpoint instead of the best checkpoint.')
    parser.add_argument('--comment', type=str, default='N.A.', help='comment about the experiment.')
    parser.add_argument("--force_wandb", action="store_true")

    args = parser.parse_args()
    return args

def setup(rank, world_size):
    os.environ['MASTER_ADDR'] = 'localhost'
    os.environ['MASTER_PORT'] = str(random.randint(30000, 40000))
    os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"

    # initialize the process group
    dist.init_process_group("nccl", rank=rank, world_size=world_size)


def main(local_rank, world_size):
    args = parse_args()
    ##########print args
    for k, v in args._get_kwargs():
        pad = ' '.join(['' for _ in range(25-len(k))])
        print(f"{k}:{pad} {v}", flush=True)

    if args.classification and args.monitor_metric == 'BLEU_4':
        args.monitor_metric = 'f1_macro'   
        
    # scaling learning rate
    args.lr_ed *= world_size
    
    setup(local_rank, world_size)

    if not args.debug:
        torch.cuda.set_device(local_rank)

    # fix random seeds
    init_seeds(args.seed + local_rank)

    if args.classification:
        with open(args.config_path, "r") as f:
            cfg = yaml.safe_load(f)
            # filter cfg to only keep the tasks specified in args.tasks
            cfg["tasks"] = {task_name: task_cfg for task_name, task_cfg in cfg["tasks"].items() if task_name in args.tasks}
    else:
        tokenizer = Tokenizer(args)

    # store run starting time in args
    if not args.debug or args.force_wandb:
        todays_date = date.today()
        now = datetime.now()
        setattr(args, "run_year", str(todays_date.year))
        setattr(args, "run_month", str(todays_date.month).zfill(2))
        setattr(args, "run_day", str(todays_date.day).zfill(2))
        setattr(args, "run_hour", f"{now.hour:02}")
        setattr(args, "run_minute", f"{now.minute:02}")
        setattr(args, "run_timehash", timehash())
        setattr(args, "save_dir", args.save_dir+f"_YY{args.run_year}-MM{args.run_month}-DD{args.run_day}-HH{args.run_hour}-MM{args.run_minute}_{args.run_timehash}")

    # setup wandb
    if (not args.debug or args.force_wandb) and local_rank==0 and args.mode == 'Train':
        wandb.init(
            entity="computational-pathology",
            project="Report_Generation",
            name=args.model_name,
            config=vars(args)
        )
        
    # create data loader
    if args.classification:
        train_dataloader = ClassificationDataLoader(args, split='train', shuffle=False)
        val_dataloader = ClassificationDataLoader(args, split='val', shuffle=False)
        test_dataloader = ClassificationDataLoader(args, split='test', shuffle=False)
    else:
        train_dataloader = R2DataLoader(args, tokenizer, split='train', shuffle=False)
        val_dataloader = R2DataLoader(args, tokenizer, split='val', shuffle=False)
        test_dataloader = R2DataLoader(args, tokenizer, split='test', shuffle=False)
    
    print("train_dataloader:",train_dataloader)
    print("val_dataloader:",val_dataloader)
    print("test_dataloader:",test_dataloader)

    # build model architecture
    if args.classification:
        if args.model_name == 'migen':
            model = MiGenModelClassification(args, cfg)

        elif args.model_name == 'histgen':
            model = HistGenModelClassification(args, cfg)
            if args.num_prompts_per_task > 1:
                raise ValueError('The current implementation of HistGenClassification only supports num_prompts_per_task=1')
        
        elif args.model_name == 'bigen':
            model = BiGenModelClassification(args, cfg)
            if args.num_prompts_per_task > 1:
                raise ValueError('The current implementation of BiGenClassification only supports num_prompts_per_task=1')
        else:
            raise ValueError('Invalid model name')
    else:
        if args.model_name == 'histgen':
            model = HistGenModel(args, tokenizer)

        elif args.model_name == 'migen':
            model = MiGenModel(args, tokenizer)

        elif args.model_name == 'bigen':
            model = BiGenModel(args, tokenizer)

        else:
            raise ValueError('Invalid model name')
    
    model = model.to(local_rank)

    if args.mode == 'Test':
        ckpt_name = 'current_checkpoint.pth' if args.resume_last else 'model_best.pth'
        resume_path = os.path.join(args.checkpoint_dir, ckpt_name)
        print(f"Loading checkpoint: {resume_path} ...")
        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)
        model.load_state_dict(checkpoint['state_dict'])

    model = DDP(model, device_ids=[local_rank], output_device=local_rank, find_unused_parameters=True)
    
    # build optimizer, learning rate scheduler. set after DDP.
    optimizer = build_optimizer(args, model)
    steps_per_epoch = math.ceil(len(train_dataloader) / args.accum_steps)
    num_training_steps = steps_per_epoch * args.epochs
    lr_scheduler = build_lr_scheduler(args, optimizer, num_training_steps)

    # get function handles of loss and metrics
    if args.classification:
        metrics = compute_scores_classification
        class_weights_dict = compute_class_weights_multitask(task_labels=train_dataloader.class_labels, cfg=cfg)
        criterion = {}
        for task in args.tasks:   
            criterion[task] = torch.nn.CrossEntropyLoss(weight=class_weights_dict[task].to(local_rank))
    else:
        criterion = compute_loss
        metrics = compute_scores

    # build trainer and start to train
    trainer = Trainer(
                    model, 
                    criterion, 
                    metrics, 
                    optimizer, 
                    args, 
                    lr_scheduler, 
                    train_dataloader, 
                    val_dataloader, 
                    test_dataloader, 
                    args.debug, 
                    cfg if args.classification else None)
    
    checkpoint_dir = args.save_dir
    if not os.path.exists(checkpoint_dir):
        if local_rank == 0:
            os.makedirs(checkpoint_dir)
            
    if args.mode == 'Train':
        if args.classification:
            trainer.train_classification(local_rank)
            if not args.debug:
                trainer.test_classification(local_rank, checkpoint_dir, best=False)
                trainer.test_classification(local_rank, checkpoint_dir, best=True)
        else:
            trainer.train(local_rank)
    else:
        if args.classification:
            trainer.test_classification(local_rank, args.checkpoint_dir, best=not args.resume_last)
        else:
            trainer.test(local_rank)


if __name__ == '__main__':
    args = parse_args()
    print("MAIN CUDA_VISIBLE_DEVICES:", os.environ.get("CUDA_VISIBLE_DEVICES"))
    print("MAIN torch.cuda.device_count():", torch.cuda.device_count())

    n_gpus = torch.cuda.device_count()
    world_size = 1
    
    if args.debug or world_size == 1:
        assert n_gpus==1
        main(0, 1)
    else:
        # main(1,1)
        mp.spawn(main,
                 args=(world_size,),
                 nprocs=world_size,
                 join=True)
