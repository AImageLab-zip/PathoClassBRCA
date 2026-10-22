import os
import json
import time
from abc import abstractmethod
from collections import OrderedDict

import numpy as np
import pandas as pd
import torch
import torch.distributed as dist
import matplotlib.pyplot as plt
import wandb
from numpy import inf
from sklearn.metrics import confusion_matrix
from tqdm import tqdm

DEBUG_BATCHES = 2

def make_cm_image(y_true, y_pred, class_names, key, normalize=None, step=None):
    """
    normalize: None | "true" | "pred" | "all" (sklearn normalization)
    """
    plt.rcParams.update({
    "font.size": 16,
    "axes.titlesize": 12,
    "axes.labelsize": 12,
    "xtick.labelsize": 12,
    "ytick.labelsize": 12,
    "legend.fontsize": 12,
    "figure.titlesize": 12,          # suptitle if used
    "figure.labelsize": 12, 
    "font.family": "serif",
    "text.usetex": False,
    })

    cm = confusion_matrix(y_true, y_pred, labels=range(len(class_names)), normalize=normalize)

    fig, ax = plt.subplots(figsize=(7, 6))
    im = ax.imshow(cm, cmap="Blues")  # don't set colors explicitly if you don't need to
    ax.set_title(key)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_xticks(range(len(class_names)))
    ax.set_yticks(range(len(class_names)))
    ax.set_xticklabels(class_names, rotation=45, ha="right")
    ax.set_yticklabels(class_names)

    # annotate cells
    for i in range(cm.shape[0]):
        for j in range(cm.shape[1]):
            ax.text(j, i, f"{cm[i, j]:.2f}" if normalize else f"{cm[i, j]:.0f}",
                    ha="center", va="center")

    fig.tight_layout()
    # buffer = io.BytesIO()
    # fig.savefig(buffer, format='png', bbox_inches='tight')
    # buffer.seek(0)
    # # Close the figure after saving it to avoid memory issues
    plt.close(fig)
    # image = Image.open(buffer)
    #image.save("MultimodalDecider/cm.png")
    
    # Return the PIL Image object to be logged later
    return fig


class BaseTrainer(object):
    def __init__(
            self, 
            model, 
            criterion, 
            metric_ftns, 
            optimizer, 
            args, 
            lr_scheduler, 
            train_dataloader, 
            val_dataloader, 
            test_dataloader, 
            debug, 
            cfg=None
        ):

        self.args = args

        self.lr_scheduler = lr_scheduler
        self.train_dataloader = train_dataloader
        self.val_dataloader = val_dataloader
        self.test_dataloader = test_dataloader

        self.model = model
        self.debug = debug
        self.tasks = self.args.tasks
        self.cfg = cfg

        self.criterion = criterion
        self.metric_ftns = metric_ftns
        self.optimizer = optimizer

        self.epochs = self.args.epochs
        self.epochs_val = self.args.epochs_val
        self.start_val = self.args.start_val
        self.save_period = self.args.save_period
        self.accum_steps = self.args.accum_steps
        self.scheduler_type = self.args.lr_scheduler

        self.mnt_mode = args.monitor_mode
        self.mnt_metric = 'val_' + args.monitor_metric
        #self.mnt_metric_test = 'test_' + args.monitor_metric
        assert self.mnt_mode in ['min', 'max', 'off']

        self.mnt_best = inf if self.mnt_mode == 'min' else -inf
        self.early_stop = getattr(self.args, 'early_stop', inf)

        self.start_epoch = 1

        self.checkpoint_dir = args.save_dir
        if args.resume is not None:
            self._resume_checkpoint(args.resume)

        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best}}

    @abstractmethod
    def _train_epoch(self, rank, epoch):
        raise NotImplementedError

    def train(self, rank):
        not_improved_count = 0
        for epoch in range(self.start_epoch, self.epochs + 1):
            print("epoch: ",epoch)
            self.train_dataloader.sampler.set_epoch(epoch)
            if (self.debug) and (epoch > self.epochs_val):
                print("Debug mode: break training loop after val epoch")
                break

            result = self._train_epoch(rank, epoch, self.debug)
            best = False

            log = {
                'epoch': epoch,
                "train/loss": result['train_loss'],
                "lr": self.optimizer.param_groups[0]["lr"],
            }

            # validation and test
            if (epoch % self.epochs_val == 0) and (epoch > self.start_val):
                val_result = self._val_epoch(rank, result, self.debug) if self.val_dataloader is not None else {}
                test_result = self._test_epoch(rank, result, self.debug) if self.test_dataloader is not None else {}
                log.update(val_result)
                log.update(test_result)

                # print logged informations to the screen
                for key, value in log.items():
                    print('\t{:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                if self.mnt_mode != 'off' and self.val_dataloader is not None:
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = log[self.mnt_metric]
                        not_improved_count = 0
                        best = True

                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                            self.early_stop))
                        break
                
                if self.val_dataloader is not None and self.mnt_mode != 'off':
                    self._record_best(log)  

            # wandb logging
            if (not self.debug or self.args.force_wandb) and rank == 0:
                wandb.log({
                    (f"{k[:k.index('_')]}/{k[k.index('_') + 1:]}" if k.startswith(("val_", "test_")) else k): v
                    for k, v in log.items() if k != "train_loss"
                })

            if (epoch % self.save_period == 0 or best) and (rank == 0) and (not self.debug):
                self._save_checkpoint(epoch, save_best=best)

        if (not self.debug or self.args.force_wandb) and rank == 0:
            if self.val_dataloader is not None:
                self._print_best()
                self._print_best_to_file()
                best_val_metrics = {
                    k.replace("val_", "best_val_"): float(v)
                    for k, v in self.best_recorder["val"].items()
                    if k.startswith("val_")
                }

            best_test_metrics = {
                k.replace("test_", "best_test_"): float(v)
                for k, v in self.best_recorder["val"].items()
                if k.startswith("test_")
            }
            if self.val_dataloader is not None:
                wandb.log({
                    **best_val_metrics,
                    **best_test_metrics,
                    "best_val_epoch": self.best_recorder["val"]["epoch"] if "epoch" in self.best_recorder["val"] else "",
                })
            else:
                wandb.log({
                    **best_test_metrics,
                })

    def train_classification(self, rank):
        not_improved_count = 0
        self.mnt_metric = f"val/aggregated/{self.args.monitor_metric}"
        self.best_recorder = {'val': {self.mnt_metric: self.mnt_best}}

        for epoch in range(self.start_epoch, self.epochs + 1):
            print("epoch: ",epoch)
            self.train_dataloader.sampler.set_epoch(epoch)

            if (self.debug or self.args.force_wandb) and (epoch > self.epochs_val):
                print("Debug mode: break training loop after val epoch")
                break

            result = self._train_epoch_classification(rank, epoch, self.debug)
            to_log = {
                    "epoch": epoch,
                    "lr": self.optimizer.param_groups[0]["lr"]
                }
            to_log.update(result)  

            best = False

            # validation and test
            if (epoch % self.epochs_val == 0) and (epoch > self.start_val):
                if self.val_dataloader is not None:
                    to_log.update(self._val_epoch_classification(rank, result, self.debug))
                if self.test_dataloader is not None:
                    to_log.update(self._test_epoch_classification(rank, result, self.debug))

                # print logged informations to the screen
                for key, value in to_log.items():
                    print('\t{:15s}: {}'.format(str(key), value))

                # evaluate model performance according to configured metric, save best checkpoint as model_best
                if self.mnt_mode != 'off' and self.val_dataloader is not None:
                    try:
                        # check whether model performance improved or not, according to specified metric(mnt_metric)
                        improved = (self.mnt_mode == 'min' and to_log[self.mnt_metric] <= self.mnt_best) or \
                                   (self.mnt_mode == 'max' and to_log[self.mnt_metric] >= self.mnt_best)
                    except KeyError:
                        print("Warning: Metric '{}' is not found. " "Model performance monitoring is disabled.".format(
                            self.mnt_metric))
                        
                        self.mnt_mode = 'off'
                        improved = False

                    if improved:
                        self.mnt_best = to_log[self.mnt_metric]
                        not_improved_count = 0
                        best = True
                    else:
                        not_improved_count += 1

                    if not_improved_count > self.early_stop:
                        print("Validation performance didn\'t improve for {} epochs. " "Training stops.".format(
                            self.early_stop))
                        break
                
                if self.val_dataloader is not None and self.mnt_mode != 'off':
                    self._record_best(to_log)

            # wandb logging
            if (not self.debug or self.args.force_wandb) and rank == 0:
                wandb.log(to_log)

            if (epoch % self.save_period == 0 or best) and (rank == 0) and (not self.debug):
                self._save_checkpoint(epoch, save_best=best)

        if rank == 0 and (not self.debug  or self.args.force_wandb):
            if self.val_dataloader is not None:
                self._print_best()
                self._print_best_to_file()
                
                # log best metric on val
                best_val_metrics = {
                    f"best_{k}": float(v)
                    for k, v in self.best_recorder["val"].items()
                    if k.startswith("val/") 
                }

            best_test_metrics = {
                f"best_{k}": v if isinstance(v, wandb.Image) else float(v)
                for k, v in self.best_recorder["val"].items()
                if k.startswith("test/")
            }

            if self.val_dataloader is not None:
                wandb.log({
                    **best_val_metrics,
                    **best_test_metrics,
                    "best_val_epoch": self.best_recorder["val"]["epoch"] if "epoch" in self.best_recorder["val"] else "",
                })
            else:
                wandb.log({
                    **best_test_metrics,
                })

    def test_classification(self, rank, checkpoint_dir, best=False, debug=False):
        log = {}
        dist.barrier()
        net = self.model

        ckpt_name = 'model_best.pth' if best else 'current_checkpoint.pth'
        checkpoint_path = os.path.join(checkpoint_dir, ckpt_name)

        net.module.load_state_dict(torch.load(checkpoint_path, map_location="cpu")['state_dict'])
        net.eval()

        test_loss = {}
        gts_local, preds_local = {}, {}
        gts_all, preds_all = {}, {}
        test_met = {}
        for task in self.tasks:
            gts_local[task] = []
            preds_local[task] = []
            gts_all[task] = []
            preds_all[task] = []
            test_met[task] = []
            test_loss[task] = 0.0

        n_valid = {task: 0 for task in self.tasks}
        
        attn_dir = os.path.join(self.checkpoint_dir, 'attn_scores')

        if self.args.save_attn_scores:
            os.makedirs(attn_dir, exist_ok=True)

        patients_dict = {}

        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(self.test_dataloader, desc='Testing')):
                if debug and batch_idx >= DEBUG_BATCHES:
                    print(f"Debug mode: break test loop after {DEBUG_BATCHES} batches")
                    break

                # images = sample["input"]["features"].cuda(non_blocking=True)
                model_input = sample["input"]
                model_input = {k: v.cuda(non_blocking=True) for k, v in model_input.items() if isinstance(v, torch.Tensor)}

                output = net(model_input)

                attn_scores = output["attn_scores"] if "attn_scores" in output else None

                if attn_scores is not None and self.args.save_attn_scores:
                    attn_scores = torch.cat([attn_scores[i].unsqueeze(0) for i in range(len(attn_scores))], dim=0)  # (num_layers, B, heads, N_patches, N_patches)
                    attn_scores = attn_scores.mean(dim=0)  # (B, heads, N_patches, N_patches)
                    attn_scores = attn_scores.mean(dim=1)  # (B, N_patches, N_patches)                    

                    if best:
                        torch.save(attn_scores, os.path.join(attn_dir, f'model_best_attn_scores_sampleID_{sample["ids"][0]}.pt'))
                    else:
                        torch.save(attn_scores, os.path.join(attn_dir, f'current_checkpoint_attn_scores_sampleID_{sample["ids"][0]}.pt'))

                output = output["output"]
                loss_item = 0.0
                patient = {"tasks": {}, "stats": {}}
                patient_correct_count = 0
                patient_valid_count = 0

                for k, v, in output.items():
                    # labels = sample['tasks'][k]["labels"].cuda(non_blocking=True)
                    labels_cpu = sample["tasks"][k]["labels"]

                    if labels_cpu.item() == -1: # unspecified
                        print(f"{sample['ids'][0]} is skipped for task {k}")
                        continue
                    
                    patient_valid_count += 1
                    
                    labels = labels_cpu.cuda(non_blocking=True)

                    # batch loss
                    task_loss = self.criterion[k](v, labels)
                    loss_item += task_loss

                    # batch metrics
                    preds = v.argmax(dim=1)
                    gts_local[k].extend(labels.detach().cpu().tolist())
                    preds_local[k].extend(preds.detach().cpu().tolist())
                    patient["tasks"][k] = {"labels": labels.detach().cpu().tolist(), "preds": preds.detach().cpu().tolist()}

                    if labels.item() == preds.item():
                        patient_correct_count += 1

                    test_loss[k] += task_loss.item()
                    n_valid[k] += 1

                patient["stats"]["accuracy"] = patient_correct_count / patient_valid_count if patient_valid_count > 0 else float("nan")
                patient["stats"]["correct"] = patient_correct_count
                patient["stats"]["total"] = patient_valid_count

                patients_dict[sample["ids"][0]] = patient


        # compute metrics and loss per task
        # gather (o fallback non-DDP)
        for task in self.tasks:
            # compute metrics per task
            if dist.is_available() and dist.is_initialized():
                preds_all[task] = gather_strings(preds_local[task])
                gts_all[task] = gather_strings(gts_local[task])
            else:
                preds_all[task], gts_all[task] = preds_local[task], gts_local[task]

            test_met[task] = self.metric_ftns(gts_all[task], preds_all[task])
            print(f"test_met for task {task}:", test_met[task])
            print(f'test/loss/{task}: ', test_loss[task] / n_valid[task] if n_valid[task] > 0 else np.nan)

            cm_plot = make_cm_image(gts_all[task], preds_all[task], self.cfg["tasks"][task]['class_names'], task)

            log.update({f'test/{task}/confusion_matrix': wandb.Image(cm_plot)})
            
            # log update
            log.update({f'test/{task}/{k}': v for k, v in test_met[task].items()})
            log.update({f'test/{task}/loss': test_loss[task] / n_valid[task] if n_valid[task] > 0 else np.nan})

        print("Results in test set")
        for key, value in log.items():
            print(f"\t{str(key):15s}: {value}")

        dist.barrier()

        # best_test is already logged in train_classification
        if not best and (not self.debug or self.args.force_wandb) and wandb.run is not None and rank == 0:
            wandb.log({
                f"last_{k}": v if isinstance(v, wandb.Image) else float(v)
                for k, v in log.items() if k.startswith("test/")
            })

        # dump patients_dict with predictions and stats to json
        if rank == 0:
            with open(os.path.join(self.checkpoint_dir, f'{"model_best" if best else "current_checkpoint"}_predictions.json'), 'w') as f:
                json.dump(patients_dict, f, indent=2)
        return log

    def test(self, rank):
        log = {}
        dist.barrier()
        self.model.eval()

        if not os.path.exists(os.path.join(self.checkpoint_dir, 'reports')) and rank==0:
            os.mkdir(os.path.join(self.checkpoint_dir, 'reports'))

        with torch.no_grad():
            test_gts, test_res = [], []
            for batch_idx, batch in enumerate(tqdm(self.test_dataloader)):
                if batch is None:
                    continue

                images_id, images, reports_ids, reports_masks = batch
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()

                output = self.model(images, mode='sample')

                reports = self.model.module.tokenizer.decode_batch(output.cpu().numpy())
                ground_truths = self.model.module.tokenizer.decode_batch(reports_ids[:, 1:].cpu().numpy())

                test_res.extend(reports)
                test_gts.extend(ground_truths)

                for i in range(len(reports)):
                    with open(os.path.join(self.checkpoint_dir, 'reports', images_id[i]) + '.txt','w') as f:
                        content = {}
                        content['predict'] = reports[i]
                        content['target'] = ground_truths[i]
                        f.write(str(content))

            if dist.is_available() and dist.is_initialized():
                test_res = gather_strings(test_res)
                test_gts = gather_strings(test_gts)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts)},
                                        {i: [re] for i, re in enumerate(test_res)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})

        print('Results in test set')
        for key, value in log.items():
            print('\t{:15s}: {}'.format(str(key), value))

    def _print_best_to_file(self):
        crt_time = time.asctime(time.localtime(time.time()))
        self.best_recorder['val']['time'] = crt_time
        #self.best_recorder['test']['time'] = crt_time
        self.best_recorder['val']['seed'] = self.args.seed
        #self.best_recorder['test']['seed'] = self.args.seed
        self.best_recorder['val']['best_model_from'] = 'val'
       # self.best_recorder['test']['best_model_from'] = 'test'

        if not os.path.exists(self.args.record_dir):
            os.makedirs(self.args.record_dir)

        masking_fixed = "masking_fixed_" if self.args.masking_fixed else ""
        padding = "with_padding_" if self.args.use_padding else ""
        record_path = os.path.join(
            self.args.record_dir,
            f"{self.args.model_name}_{self.args.seed}_{masking_fixed}{padding}{self.args.lr_scheduler}_YY{self.args.run_year}-MM{self.args.run_month}-DD{self.args.run_day}-HH{self.args.run_hour}-MM{self.args.run_minute}_{self.args.run_timehash}.csv"
        )

        if not os.path.exists(record_path):
            record_table = pd.DataFrame()
        else:
            record_table = pd.read_csv(record_path)

        new_rows = pd.DataFrame([
            {k: v for k, v in self.best_recorder['val'].items() if not isinstance(v, wandb.Image)}
        ])

        record_table = pd.concat([record_table, new_rows], ignore_index=True)
        record_table.to_csv(record_path, index=False)

    def _prepare_device(self, n_gpu_use):
        n_gpu = torch.cuda.device_count()
        if n_gpu_use > 0 and n_gpu == 0:
            print("Warning: There\'s no GPU available on this machine," "training will be performed on CPU.")
            n_gpu_use = 0
        if n_gpu_use > n_gpu:
            print(
                "Warning: The number of GPU\'s configured to use is {}, but only {} are available " "on this machine.".format(
                    n_gpu_use, n_gpu))
            n_gpu_use = n_gpu
        device = torch.device('cuda:0' if n_gpu_use > 0 else 'cpu')
        list_ids = list(range(n_gpu_use))
        return device, list_ids

    def _save_checkpoint(self, epoch, save_best=False):
        state = {
            'epoch': epoch,
            'state_dict': self.model.module.state_dict(),
            'optimizer': self.optimizer.state_dict(),
            'scheduler': self.lr_scheduler.state_dict() if self.lr_scheduler is not None else None,
            'monitor_best': self.mnt_best,
        }
        filename = os.path.join(self.checkpoint_dir, 'current_checkpoint.pth')
        torch.save(state, filename)
        print("Saving checkpoint: {} ...".format(filename))
        if save_best:
            best_path = os.path.join(self.checkpoint_dir, 'model_best.pth')
            torch.save(state, best_path)
            print("Saving current best: model_best.pth ...")

    def _resume_checkpoint(self, resume_path):
        resume_path = str(resume_path)
        print("Loading checkpoint: {} ...".format(resume_path))

        checkpoint = torch.load(resume_path, map_location="cpu", weights_only=False)

        self.start_epoch = checkpoint['epoch'] + 1
        self.mnt_best = checkpoint['monitor_best']
        state_dict = checkpoint["state_dict"]
        
        is_model_ddp = hasattr(self.model, "module")
        is_ckpt_ddp = next(iter(state_dict)).startswith("module.")

        if is_model_ddp and not is_ckpt_ddp:
            self.model.module.load_state_dict(state_dict, strict=True)
        elif (not is_model_ddp) and is_ckpt_ddp:
            new_state = OrderedDict((k.replace("module.", "", 1), v) for k, v in state_dict.items())
            self.model.load_state_dict(new_state, strict=True)
        else:
            self.model.load_state_dict(state_dict, strict=True)
            #self.model.load_state_dict(checkpoint['state_dict'])
            
        self.optimizer.load_state_dict(checkpoint['optimizer'])
        if 'scheduler' in checkpoint and checkpoint['scheduler'] is not None and self.lr_scheduler is not None:
            self.lr_scheduler.load_state_dict(checkpoint['scheduler'])
            
        print("Checkpoint loaded. Resume training from epoch {}".format(self.start_epoch))

    def _record_best(self, log):
        improved_val = (self.mnt_mode == 'min' and log[self.mnt_metric] <= self.best_recorder['val'][
            self.mnt_metric]) or \
                       (self.mnt_mode == 'max' and log[self.mnt_metric] >= self.best_recorder['val'][self.mnt_metric])
        if improved_val:
            self.best_recorder['val'].update(log)

    def _print_best(self):
        print('Best results (w.r.t {}) in validation set:'.format(self.args.monitor_metric))
        for key, value in self.best_recorder['val'].items():
            print('\t{:15s}: {}'.format(str(key), value))


class Trainer(BaseTrainer):
    def __init__(self, model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader, debug, cfg=None):
        super(Trainer, self).__init__(model, criterion, metric_ftns, optimizer, args, lr_scheduler, train_dataloader, val_dataloader,
                 test_dataloader, debug, cfg=cfg)
        self.cfg = cfg

    def _train_epoch_classification(self, rank, epoch, debug=False):
        dist.barrier()
        self.model.train()

        # initialize training variables
        avg_train_loss = {}
        train_loss = {}
        gts_local, preds_local = {}, {}
        gts_all, preds_all = {}, {}
        train_met = {}
        for task in self.tasks:
            gts_local[task] = []
            preds_local[task] = []
            gts_all[task] = []
            preds_all[task] = []
            train_met[task] = []
            train_loss[task] = 0.0

        n_valid = {task: 0 for task in self.tasks}

        for batch_idx, sample in enumerate(tqdm(self.train_dataloader, desc='Training')):
            if debug and batch_idx >= DEBUG_BATCHES:
                print("Debug mode: break training loop after {} batches".format(DEBUG_BATCHES))
                break

            model_input = sample["input"]
            model_input = {k: v.cuda(non_blocking=True) for k, v in model_input.items() if isinstance(v, torch.Tensor)}
            # images = (1, N_patches, 1536=1024(UNI)+512(PLIP))
            output = self.model(model_input)    # (B, num_classes)
            output = output["output"]

            loss_item = 0.0
            n_valid_tasks_in_batch = 0.0

            for k, v, in output.items():
                # labels = sample['tasks'][k]["labels"].cuda(non_blocking=True)
                labels_cpu = sample["tasks"][k]["labels"]                

                if model_input["features"].shape[0] > 1:
                    raise NotImplementedError("Batch size > 1 is not supported for classification tasks yet")
                
                if labels_cpu.item() == -1: # unspecified
                    print(f"{sample['ids'][0]} is skipped for task {k}")
                    continue

                labels = labels_cpu.cuda(non_blocking=True)
                # batch loss
                task_loss = self.criterion[k](v, labels)

                loss_item += task_loss
                n_valid_tasks_in_batch += 1

                # update train loss epoch per task
                train_loss[k] += task_loss.item()
                n_valid[k] += 1

                # batch metrics
                preds = v.argmax(dim=1)
                gts_local[k].extend(labels.detach().cpu().tolist())
                preds_local[k].extend(preds.detach().cpu().tolist())

            # backpropagation: loss_item = sum of the losses of all tasks
            if n_valid_tasks_in_batch == 0:
                print(f"Warning: all tasks in the batch are unspecified. This batch is skipped for backpropagation.")
                continue

            loss = loss_item / n_valid_tasks_in_batch
            loss = loss / self.accum_steps
            loss.backward()

            # optimizer/scheduler step
            if (batch_idx + 1) % self.accum_steps == 0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler_type=='warmup_cosine' or self.scheduler_type=='OneCycleLR':
                    self.lr_scheduler.step()
        
        # if the number of batches is not multiple of accum_steps
        if (batch_idx + 1) % self.accum_steps != 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler_type=='warmup_cosine' or self.scheduler_type=='OneCycleLR':
                self.lr_scheduler.step()

        # gather (o fallback non-DDP)
        log = {}
        for task in self.tasks:
            # compute metrics per task
            if dist.is_available() and dist.is_initialized():
                preds_all[task] = gather_strings(preds_local[task])
                gts_all[task] = gather_strings(gts_local[task])
            else:
                preds_all[task], gts_all[task] = preds_local[task], gts_local[task]

            train_met[task] = self.metric_ftns(gts_all[task], preds_all[task])
            print(f"train_met for task {task}:", train_met[task])

            # compute train loss epoch per task
            avg_train_loss[task] = train_loss[task] / n_valid[task] if n_valid[task] > 0 else np.nan

            print(f"train_loss for task {task}:", avg_train_loss[task])

            # log update
            log.update({
                f"train/{task}/loss": avg_train_loss[task], 
                **{f"train/{task}/{k}": v for k, v in train_met[task].items()}
            })

        if self.scheduler_type == "StepLR":
            self.lr_scheduler.step()
        
        return log
    
    def _train_epoch(self, rank, epoch, debug=False):
        dist.barrier()
        train_loss = 0
        self.model.train()

        for batch_idx, batch in enumerate(tqdm(self.train_dataloader, desc='Training')):
            if batch is None:
                continue

            _, images, reports_ids, reports_masks = batch

            if debug and batch_idx >= DEBUG_BATCHES:
                print("Debug mode: break training loop after {} batches".format(DEBUG_BATCHES))
                break

            images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
            # images = (1, N_patches, 1536=1024(UNI)+512(PLIP)), reports_ids = (B, seq_length)
            
            output = self.model(images, reports_ids,  mode='train')    # (B, seq_length-1, vocab_size)

            loss = self.criterion(output, reports_ids, reports_masks)

            train_loss += loss.item()

            loss = loss / self.accum_steps

            loss.backward()

            if (batch_idx + 1) % self.accum_steps == 0:
                torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
                self.optimizer.step()
                self.optimizer.zero_grad()

                if self.scheduler_type=='warmup_cosine' or self.scheduler_type=='OneCycleLR':
                    self.lr_scheduler.step()
        
        # if the number of batches is not multiple of accum_steps
        if (batch_idx + 1) % self.accum_steps != 0:
            torch.nn.utils.clip_grad_value_(self.model.parameters(), 0.1)
            self.optimizer.step()
            self.optimizer.zero_grad()

            if self.scheduler_type=='warmup_cosine' or self.scheduler_type=='OneCycleLR':
                self.lr_scheduler.step()

        log = {'train_loss': train_loss / len(self.train_dataloader)}
        print('train_loss: ', train_loss / len(self.train_dataloader))

        if self.scheduler_type == "StepLR":
            self.lr_scheduler.step()

        return log
    
    def _val_epoch(self, rank, log, debug=False):
        dist.barrier()
        self.model.eval()
        with torch.no_grad():
            val_gts_texts, val_res_texts = [], []
            for batch_idx, batch in enumerate(tqdm(self.val_dataloader, desc='Validating')):
                if batch is None:
                    continue

                _, images, reports_ids, reports_masks = batch
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()

                output = self.model(images, mode='sample')

                val_res_texts.extend(self.model.module.tokenizer.decode_batch(output.cpu().numpy()))              #predict
                val_gts_texts.extend(self.model.module.tokenizer.decode_batch(reports_ids[:,1:].cpu().numpy()))   #ground truth

                if debug and batch_idx >= DEBUG_BATCHES:
                    print("Debug mode: break validation loop after {} batches".format(DEBUG_BATCHES))
                    break
                
            if dist.is_available() and dist.is_initialized():
                val_res_texts = gather_strings(val_res_texts)
                val_gts_texts = gather_strings(val_gts_texts)

            val_met = self.metric_ftns({i: [gt] for i, gt in enumerate(val_gts_texts)},
                                       {i: [re] for i, re in enumerate(val_res_texts)})

            log.update(**{'val_' + k: v for k, v in val_met.items()})
            print("val_met: ",val_met)

        return log

    def _val_epoch_classification(self, rank, log, debug=False):
        dist.barrier()
        self.model.eval()

        val_loss = {}
        gts_local, preds_local = {}, {}
        gts_all, preds_all = {}, {}
        val_met = {}
        for task in self.tasks:
            gts_local[task] = []
            preds_local[task] = []
            gts_all[task] = []
            preds_all[task] = []
            val_met[task] = []
            val_loss[task] = 0.0

        n_valid = {task: 0 for task in self.tasks}

        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(self.val_dataloader, desc='Validating')):
                if debug and batch_idx >= DEBUG_BATCHES:
                    print(f"Debug mode: break validation loop after {DEBUG_BATCHES} batches")
                    break

                model_input = sample["input"]
                model_input = {k: v.cuda(non_blocking=True) for k, v in model_input.items() if isinstance(v, torch.Tensor)}
                output = self.model(model_input)
                output = output["output"]

                # loss_item = sum of the losses of all tasks
                for k, v, in output.items():
                    # labels = sample['tasks'][k]["labels"].cuda(non_blocking=True)
                    labels_cpu = sample["tasks"][k]["labels"]

                    if labels_cpu.item() == -1: # unspecified
                        print(f"{sample['ids'][0]} is skipped for task {k}")
                        continue
                    
                    labels = labels_cpu.cuda(non_blocking=True)
                    # batch loss
                    task_loss = self.criterion[k](v, labels)

                    # batch metrics
                    preds = v.argmax(dim=1)
                    gts_local[k].extend(labels.detach().cpu().tolist())
                    preds_local[k].extend(preds.detach().cpu().tolist())
         
                    val_loss[k] += task_loss.item()
                    n_valid[k] += 1

        # compute metrics
        # gather (o fallback non-DDP)
        # aggregate metrics
        aggregated_metrics = {}
        metric_counts = {}
        for task in self.tasks:
            # compute metrics per task
            if dist.is_available() and dist.is_initialized():
                preds_all[task] = gather_strings(preds_local[task])
                gts_all[task] = gather_strings(gts_local[task])
            else:
                preds_all[task], gts_all[task] = preds_local[task], gts_local[task]

            val_met[task] = self.metric_ftns(gts_all[task], preds_all[task])
            print(f"val_met for task {task}:", val_met[task])
            print(f'val_loss_{task}: ', val_loss[task] / n_valid[task] if n_valid[task] > 0 else np.nan)

            # compute aggregated metric
            for k, v in val_met[task].items():
                if not np.isnan(v):
                    aggregated_metrics[k] = aggregated_metrics.get(k, 0.0) + v
                    metric_counts[k] = metric_counts.get(k, 0) + 1
            
            # log update
            log.update({f'val/{task}/{k}': v for k, v in val_met[task].items()})
            log.update({f'val/{task}/loss': val_loss[task] / n_valid[task] if n_valid[task] > 0 else np.nan})
        
        for k in aggregated_metrics:
            aggregated_metrics[k] /= metric_counts[k]

        log.update({f'val/aggregated/{metr}': avg for metr, avg in aggregated_metrics.items()})

        return log
    
    def _test_epoch(self, rank, log, debug=False):
        dist.barrier()
        self.model.eval()

        with torch.no_grad():
            test_gts_texts, test_res_texts = [], []
            for batch_idx, batch in enumerate(tqdm(self.test_dataloader, desc='Testing')):
                if batch is None:
                    continue

                images_id, images, reports_ids, reports_masks = batch
                images, reports_ids, reports_masks = images.cuda(), reports_ids.cuda(), reports_masks.cuda()
                output = self.model(images,  mode='sample')
                test_res_texts.extend(self.model.module.tokenizer.decode_batch(output.cpu().numpy()))              #predict
                test_gts_texts.extend(self.model.module.tokenizer.decode_batch(reports_ids[:,1:].cpu().numpy()))   #ground truth

                if debug and batch_idx >= DEBUG_BATCHES:
                    print("Debug mode: break testing loop after {} batches".format(DEBUG_BATCHES))
                    break

            if dist.is_available() and dist.is_initialized():
                test_res_texts = gather_strings(test_res_texts)
                test_gts_texts = gather_strings(test_gts_texts)

            #test_gts, test_res = gather_strings(test_gts_ids), gather_strings(test_res_ids)

            test_met = self.metric_ftns({i: [gt] for i, gt in enumerate(test_gts_texts)},
                                        {i: [re] for i, re in enumerate(test_res_texts)})
            log.update(**{'test_' + k: v for k, v in test_met.items()})
            print("test_met: ", test_met)

        return log
    
    def _test_epoch_classification(self, rank, log, debug=False):
        dist.barrier()
        self.model.eval()

        test_loss = {}
        gts_local, preds_local = {}, {}
        gts_all, preds_all = {}, {}
        test_met = {}
        for task in self.tasks:
            gts_local[task] = []
            preds_local[task] = []
            gts_all[task] = []
            preds_all[task] = []
            test_met[task] = []
            test_loss[task] = 0.0

        n_valid = {task: 0 for task in self.tasks}

        with torch.no_grad():
            for batch_idx, sample in enumerate(tqdm(self.test_dataloader, desc='Testing')):
                if debug and batch_idx >= DEBUG_BATCHES:
                    print(f"Debug mode: break test loop after {DEBUG_BATCHES} batches")
                    break

                # images = sample["input"]["features"].cuda(non_blocking=True)
                model_input = sample["input"]
                model_input = {k: v.cuda(non_blocking=True) for k, v in model_input.items() if isinstance(v, torch.Tensor)}

                output = self.model(model_input)
                output = output["output"]

                for k, v, in output.items():
                    # labels = sample['tasks'][k]["labels"].cuda(non_blocking=True)
                    labels_cpu = sample["tasks"][k]["labels"]

                    if labels_cpu.item() == -1: # unspecified
                        print(f"{sample['ids'][0]} is skipped for task {k}")
                        continue
                    
                    labels = labels_cpu.cuda(non_blocking=True)
                    # batch loss
                    task_loss = self.criterion[k](v, labels)

                    # batch metrics
                    preds = v.argmax(dim=1)
                    gts_local[k].extend(labels.detach().cpu().tolist())
                    preds_local[k].extend(preds.detach().cpu().tolist())
                    
                    test_loss[k] += task_loss.item()
                    n_valid[k] += 1

        # compute metrics
        # gather (o fallback non-DDP)
        for task in self.tasks:
            # compute metrics per task
            if dist.is_available() and dist.is_initialized():
                preds_all[task] = gather_strings(preds_local[task])
                gts_all[task] = gather_strings(gts_local[task])
            else:
                preds_all[task], gts_all[task] = preds_local[task], gts_local[task]

            test_met[task] = self.metric_ftns(gts_all[task], preds_all[task])

            print(f"test_met for task {task}:", test_met[task])
            print(f'test_loss_{task}: ', test_loss[task] / n_valid[task] if n_valid[task] > 0 else np.nan)

            cm_plot = make_cm_image(gts_all[task], preds_all[task], self.cfg["tasks"][task]['class_names'], task)

            log.update({f'test/{task}/confusion_matrix': wandb.Image(cm_plot)})
            
            # log update
            log.update({f'test/{task}/{k}': v for k, v in test_met[task].items()})
            log.update({f'test/{task}/loss': test_loss[task] / n_valid[task] if n_valid[task] > 0 else np.nan})

        return log

    
def distributed_concat(tensor, num_total_examples):
    output_tensors = [tensor.clone() for _ in range(torch.distributed.get_world_size())]
    torch.distributed.all_gather(output_tensors, tensor)
    concat = torch.cat(output_tensors, dim=0)
    # truncate the dummy elements added by SequentialDistributedSampler
    return concat[:num_total_examples]

def gather_strings(local_list):
    world = dist.get_world_size()
    gathered = [None for _ in range(world)]
    dist.all_gather_object(gathered, local_list)
    out = []
    for part in gathered:
        out.extend(part)
    return out
