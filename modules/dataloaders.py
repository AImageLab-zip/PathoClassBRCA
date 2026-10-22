import torch
import numpy as np
from torchvision import transforms
from torch.utils.data import DataLoader
from .datasets import TcgaImageDataset, TcgaImageJsonDataset, TcgaImageClassificationDataset
import math

class R2DataLoader(DataLoader):
    def __init__(self, args, tokenizer, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.tokenizer = tokenizer
        self.split = split
        self.use_json_dataset = args.use_json_dataset
        self.classification = args.classification

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'TCGA':
            # WSI-Bench
            if self.use_json_dataset:
                self.dataset = TcgaImageJsonDataset(self.args, self.tokenizer, self.split, transform=self.transform)
            # PathText
            else:
                self.dataset = TcgaImageDataset(self.args, self.tokenizer, self.split, transform=self.transform)

            
        if split == 'train':
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        else:
            self.sampler = SequentialDistributedSampler(self.dataset,self.args.batch_size)

        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'sampler': self.sampler,
        }
        super().__init__(**self.init_kwargs)

    @staticmethod
    def collate_fn(data):
        # Remove samples returned as None by __getitem__
        data = [d for d in data if d is not None]

        # If the whole batch is empty, return None
        if len(data) == 0:
            return None

        images_id, images, reports_ids, reports_masks, _ = zip(*data)

        # batch_size = 1 case
        images = images[0].unsqueeze(0)

        return (
            images_id,
            images,
            torch.LongTensor(reports_ids),
            torch.FloatTensor(reports_masks)
        )
    
class ClassificationDataLoader(DataLoader):
    def __init__(self, args, split, shuffle):
        self.args = args
        self.dataset_name = args.dataset_name
        self.batch_size = args.batch_size
        self.shuffle = shuffle
        self.num_workers = args.num_workers
        self.split = split
        self.tasks = args.tasks

        if split == 'train':
            self.transform = transforms.Compose([
                transforms.Resize(256),
                transforms.RandomCrop(224),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])
        else:
            self.transform = transforms.Compose([
                transforms.Resize((224, 224)),
                transforms.ToTensor(),
                transforms.Normalize((0.485, 0.456, 0.406),
                                     (0.229, 0.224, 0.225))])

        if self.dataset_name == 'TCGA':
            self.dataset = TcgaImageClassificationDataset(self.args, self.split)
            self.class_labels = {}
            for task in self.tasks:
                self.class_labels[task] = [
                    ex["tasks"][task]["label"]
                    for ex in self.dataset.examples
                    if "tasks" in ex and task in ex["tasks"] and "label" in ex["tasks"][task]
                ]

        if split == 'train':
            self.sampler = torch.utils.data.distributed.DistributedSampler(self.dataset)
        else:
            self.sampler = SequentialDistributedSampler(self.dataset, self.args.batch_size)


        self.init_kwargs = {
            'dataset': self.dataset,
            'batch_size': self.batch_size,
            'shuffle': self.shuffle,
            'collate_fn': self.collate_fn,
            'num_workers': self.num_workers,
            'sampler': self.sampler,
            "pin_memory": True,
        }
        if self.num_workers > 0:
            self.init_kwargs.update(persistent_workers=True, prefetch_factor=2)

        super().__init__(**self.init_kwargs)


    @staticmethod
    def collate_fn(batch):
        # Remove samples returned as None by __getitem__
        data = [d for d in batch if d is not None]

        # If the whole batch is empty, return None
        if len(data) == 0:
            return None
        
        images = []
        ids = []
        image_paths = []

        # unione di tutti i task presenti nel batch
        all_tasks = set()
        for ex in data:
            all_tasks.update(ex.get("tasks", {}).keys())

        all_tasks = sorted(all_tasks)

        # accumuliamo per task
        task_labels = {t: [] for t in all_tasks}
        task_text_labels = {t: [] for t in all_tasks}

        for ex in data:
            ids.append(ex["id"])
            image_paths.append(ex["image_path"])

            if ex["input"]["features"] is None:
                ex_id = ex["id"]
                print(f"Invalid features for ID: {ex_id}")
                return None
            
            images.append(ex["input"]["features"])

            tasks_dict = ex.get("tasks", {})
            for t in all_tasks:
                if t in tasks_dict and "label" in tasks_dict[t]:
                    task_labels[t].append(tasks_dict[t]["label"])
                    task_text_labels[t].append(tasks_dict[t].get("text_label", None))
                else:
                    # se manca quel task per quel sample -> ignore_index
                    task_labels[t].append(-1)
                    task_text_labels[t].append(None)

        # stack immagini
        images = torch.stack([torch.as_tensor(img) for img in images], dim=0)

        # costruisci la struttura tasks come vuoi tu:
        tasks = {
            t: {
                "labels": torch.tensor(task_labels[t], dtype=torch.long),
                "text_labels": task_text_labels[t],
            }
            for t in all_tasks
        }

        return {
            "input": {
                "features": images, 
                    },
            "ids": ids,
            "image_paths": image_paths,
            "tasks": tasks,  
        }
    

class SequentialDistributedSampler(torch.utils.data.sampler.Sampler):
    """
    Distributed Sampler that subsamples indicies sequentially,
    making it easier to collate all results at the end.
    Even though we only use this sampler for eval and predict (no training),
    which means that the model params won't have to be synced (i.e. will not hang
    for synchronization even if varied number of forward passes), we still add extra
    samples to the sampler to make it evenly divisible (like in `DistributedSampler`)
    to make it easy to `gather` or `reduce` resulting tensors at the end of the loop.
    """

    def __init__(self, dataset, batch_size, rank=None, num_replicas=None):
        if num_replicas is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            num_replicas = torch.distributed.get_world_size()
        if rank is None:
            if not torch.distributed.is_available():
                raise RuntimeError("Requires distributed package to be available")
            rank = torch.distributed.get_rank()
        self.dataset = dataset
        self.num_replicas = num_replicas
        self.rank = rank
        self.batch_size = batch_size
        self.num_samples = int(math.ceil(len(self.dataset) * 1.0 / self.batch_size / self.num_replicas)) * self.batch_size
        self.total_size = self.num_samples * self.num_replicas

    def __iter__(self):
        indices = list(range(len(self.dataset)))
        # add extra samples to make it evenly divisible
        indices += [indices[-1]] * (self.total_size - len(indices))
        # subsample
        indices = indices[self.rank * self.num_samples : (self.rank + 1) * self.num_samples]
        return iter(indices)

    def __len__(self):
        return self.num_samples