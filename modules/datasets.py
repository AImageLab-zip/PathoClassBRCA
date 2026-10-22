import os
import json
import torch
from torch.utils.data import Dataset
import pandas as pd
import numpy as np

def load_tensor_or_keyed_tensor(path, allowed_keys=None):
    if allowed_keys is None:
        allowed_keys = ["features", "feature", "feats", "x", "tensor", "embeddings", "selected_features"]

    obj = torch.load(path, map_location="cpu")

    if isinstance(obj, torch.Tensor):
        return obj.float()

    if isinstance(obj, dict):
        for key in allowed_keys:
            if key in obj and isinstance(obj[key], torch.Tensor):
                return obj[key].float()

    raise ValueError(f"Could not load tensor from {path}. Checked keys: {allowed_keys}")

# Path Text
class BaseDataset(Dataset):
    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir 
        self.image_dir_plip = args.image_dir_plip
        self.ann_path = args.ann_path
        self.split_path = args.split_path

        self.max_seq_length = args.max_seq_length
        self.max_fea_length = args.max_fea_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform
        self.masking_fixed = args.masking_fixed
        use_padding = args.use_padding

        print(f"{self.split}:", len(pd.read_csv(self.split_path).loc[:, self.split].dropna()))
        cases = self.clean_data(pd.read_csv(self.split_path).loc[:, self.split].dropna())
        print(f"{self.split}:", len(cases))

        
        self.examples = []
        root = self.ann_path

        s=[]
        for dir in os.listdir(root):
            if not dir in cases.keys(): # check whether contained in the split
                continue
            else:
                img_name = cases[dir]
                
            image_path = os.path.join(self.image_dir, img_name)

            if not os.path.exists(image_path+'.pt'):
                print(image_path," not exists.")
                continue
                
            file_name = os.path.join(root, dir, 'annotation')
            with open(file_name, 'r') as f:
                anno = f.read()
            s.append(file_name)
          
            report_ids = tokenizer(anno)[:self.max_seq_length]
            true_report_len = len(report_ids)

            if use_padding:
                pad_id = self.tokenizer.pad_idx
                if true_report_len < self.max_seq_length:
                    padding = [pad_id] * (self.max_seq_length-true_report_len)  
                    report_ids.extend(padding)
                    
                if self.masking_fixed:
                    mask = [1] * true_report_len + [0] * (self.max_seq_length - true_report_len)
                else:
                    mask = [1] * self.max_seq_length
                    
            else:
                mask = [1] * true_report_len

            self.examples.append({'id':dir, 'image_path': image_path+'.pt', 'report': anno, 'split': self.split, 'ids':report_ids, 'mask': mask})

        print(f'The size of {self.split} dataset: {len(self.examples)}')


    def __len__(self):
        return len(self.examples)
    
class TcgaImageDataset(BaseDataset):
    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        
        image = torch.load(image_path)       #(N_patch, 1024)
        image = image[:self.max_fea_length]
        
        if self.image_dir_plip:
            image_plip = torch.load(os.path.join(self.image_dir_plip, image_path.split('/')[-1]))[:self.max_fea_length]
            
            if image.shape[0] != image_plip.shape[0]:
                raise RuntimeError(
                    f"Patch number mismatch for slide {image_path.split('/')[-1]}: "
                    f"image={image.shape}, image_plip={image_plip.shape}, "
                    f"img_path={image_path}, plip_path={os.path.join(self.image_dir_plip, image_path.split('/')[-1])}"
    )
            image = torch.cat((image, image_plip),dim=1) #(N_patch, 1536)

        report_ids = example['ids']
        report_masks = example['mask']

        seq_length = len(report_ids)

        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample


    def clean_data(self,data):
        cases = {}
        for idx in range(len(data)):
            case_name = data[idx]
            case_id = '-'.join(case_name.split('-')[:3]).split('.')[0]
            cases[case_id] = case_id
        return cases 
    
    def filter_df(self,df, filter_dict):
        if len(filter_dict) > 0:
            filter_mask = np.full(len(df), True, bool)
            for key, val in filter_dict.items():
                mask = df[key].isin(val)
                filter_mask = np.logical_and(filter_mask, mask)
            df = df[filter_mask]
        return df

    def df_prep(self,data, label_dict, ignore, label_col):
        if label_col != 'label':
            data['label'] = data[label_col].copy()

        mask = data['label'].isin(ignore)
        data = data[~mask]
        data.reset_index(drop=True, inplace=True)
        for i in data.index:
            key = data.loc[i, 'label']
            data.at[i, 'label'] = label_dict[key]

        return data
    
# WSI Bench
class BaseJsonDataset(Dataset):
    """
    JSON-based dataset for the new train/test files.

    Expected JSON item formats:
    - Test-style:
        {
          "question_id": "...",
          "image": "TCGA-BRCA/<SLIDE_ID>.pt",
          "question": "...",
          "text": "...",
          "metadata": "Report"
        }

    - Train-style:
        {
          "id": "...",
          "image": "TCGA-BRCA/<SLIDE_ID>.pt",
          "conversations": [{"from":"human","value":"..."}, {"from":"gpt","value":"..."}]
        }
    """

    def __init__(self, args, tokenizer, split, transform=None):
        self.image_dir = args.image_dir
        self.image_dir_plip = getattr(args, "image_dir_plip", None)

        # json paths
        self.train_json = getattr(args, "train_json", None)
        self.val_json = getattr(args, "val_json", None)
        self.test_json = getattr(args, "test_json", None)

        self.image_cache = {}      
        self.image_cache_plip = {} 

        self.max_seq_length = args.max_seq_length
        self.max_fea_length = args.max_fea_length
        self.split = split
        self.tokenizer = tokenizer
        self.transform = transform

        self.masking_fixed = getattr(args, "masking_fixed", False)
        use_padding = getattr(args, "use_padding", True)

        # optional filters
        data = self._load_split_json(split)

        self.examples = []
        for item in data:
            img_rel = item.get("image", "")
            if not img_rel:
                continue

            slide_id = os.path.splitext(os.path.basename(img_rel))[0]

            # cohort substitution in image_dir and image_dir_plip if needed
            cohort = img_rel.split("/")[0]
            cohort_basic = cohort.replace("TCGA-", "")

            self.current_image_dir = self.image_dir.replace("{cohort}", cohort_basic)

            self.current_image_dir_plip = (
                self.image_dir_plip.replace("{cohort}", cohort_basic)
                if self.image_dir_plip
                else None
            )

            if not os.path.exists(self.current_image_dir):
                print(f"Warning: Image directory {self.current_image_dir} does not exist. Skipping {slide_id}.")
                continue

            if self.current_image_dir_plip and not os.path.exists(self.current_image_dir_plip):
                print(f"Warning: PLIP image directory {self.current_image_dir_plip} does not exist. Skipping {slide_id}.")
                continue

            image_path = os.path.join(self.current_image_dir, slide_id)

            if not os.path.exists(image_path+'.pt'):
                print(image_path," not exists.")
                continue

            anno = self._get_text(item)
            if not anno:
                continue

            report_ids = tokenizer(anno)[:self.max_seq_length]
            true_report_len = len(report_ids)

            if use_padding:
                pad_id = self.tokenizer.pad_idx
                if true_report_len < self.max_seq_length:
                    padding = [pad_id] * (self.max_seq_length-true_report_len)  
                    report_ids.extend(padding)
                    
                if self.masking_fixed:
                    mask = [1] * true_report_len + [0] * (self.max_seq_length - true_report_len)
                else:
                    mask = [1] * self.max_seq_length
                    
            else:
                mask = [1] * true_report_len

            self.examples.append({'id':slide_id, 'image_path': image_path+'.pt', 'report': anno, 'split': self.split, 'ids':report_ids, 'mask': mask, 'image_dir_plip': self.current_image_dir_plip})

        print(f'The size of {self.split} dataset: {len(self.examples)}')

    def _load_split_json(self, split: str):
        if split == "train":
            if not self.train_json:
                raise ValueError("args.train_json is required for split='train'")
            with open(self.train_json, "r", encoding="utf-8") as f:
                return json.load(f)
        if split == "val":
            if not self.val_json:
                raise ValueError("args.val_json is required for split='val'")
            with open(self.val_json, "r", encoding="utf-8") as f:
                return json.load(f)
        if split == "test":
            if not self.test_json:
                raise ValueError("args.test_json is required for split='test'")
            with open(self.test_json, "r", encoding="utf-8") as f:
                return json.load(f)

        raise ValueError(f"Unsupported split: {split}")

    def _get_text(self, item: dict) -> str:
        # test-style
        if "text" in item and item["text"]:
            return item["text"]

        # train/val-style
        conv = item.get("conversations", [])
        if isinstance(conv, list):
            for msg in conv:
                if msg.get("from") == "gpt" and msg.get("value"):
                    return msg["value"]

        return ""

    def __len__(self):
        return len(self.examples)

class TcgaImageJsonDataset(BaseJsonDataset):

    def __getitem__(self, idx):
        example = self.examples[idx]
        image_id = example['id']
        image_path = example['image_path']
        image_dir_plip = example['image_dir_plip']
        
        # image = torch.load(image_path)       #(N_patch, 1024)
        # image = image[:self.max_fea_length]
        ###################
        if image_path in self.image_cache:
            image = self.image_cache[image_path]
        else:
            image = load_tensor_or_keyed_tensor(image_path)[:self.max_fea_length]       #(N_patch, 1024)
            self.image_cache[image_path] = image

        if image_dir_plip:
            image_plip_path = os.path.join(image_dir_plip, image_path.split('/')[-1])

            if os.path.exists(image_plip_path):
                if image_plip_path in self.image_cache_plip:
                    image_plip = self.image_cache_plip[image_plip_path]
                else:
                    image_plip = load_tensor_or_keyed_tensor(image_plip_path)[:self.max_fea_length]
                    self.image_cache_plip[image_plip_path] = image_plip

                # skip sample
                if image.shape[0] != image_plip.shape[0]:
                    print(
                        f"Patch number mismatch for slide {image_path.split('/')[-1]}: "
                        f"image={image.shape}, image_plip={image_plip.shape}, "
                        f"img_path={image_path}, plip_path={image_plip_path}"
                    )
                    return None

                image = torch.cat((image, image_plip), dim=1)  # (N_patch, 1536)

            else:
                # skip sample
                print(f"Skipping sample {image_id}: PLIP image not found at {image_plip_path}")
                return None

        report_ids = example['ids']
        report_masks = example['mask']

        seq_length = len(report_ids)

        sample = (image_id, image, report_ids, report_masks, seq_length)
        return sample

# Classification Dataset
class BaseDatasetClassification(Dataset):

    def __init__(self, args, split):
        self.image_dir = args.image_dir
        self.image_dir_plip = getattr(args, "image_dir_plip", None) 

        self.image_cache = {}      
        self.image_cache_plip = {}     

        # json paths
        self.train_json = getattr(args, "train_json", None)
        self.val_json = getattr(args, "val_json", None)
        self.test_json = getattr(args, "test_json", None)

        self.max_fea_length = args.max_fea_length
        self.split = split
        self.tasks = args.tasks

        # optional filters
        data = self._load_split_json(split)

        self.examples = []
        for item in data:
            report_id = item.get("id", "")
            if not report_id:
                continue

            img_rel = item.get("image", "")
            if not img_rel:
                continue

            slide_id = os.path.splitext(os.path.basename(img_rel))[0]
            
            image_path = os.path.join(self.image_dir, f"{slide_id}.pt")
            if not os.path.exists(image_path):
                print(f"{image_path} does not exists.")
                continue

            if self.image_dir_plip is not None:
                image_plip_path = os.path.join(self.image_dir_plip, f"{slide_id}.pt")

                if not os.path.exists(image_plip_path):
                    print(f"{image_plip_path} does not exists.")  
                    continue
            else:
                image_plip_path = None

            task_example = {}
            for task in self.tasks:
                sample_item = item.get("tasks", {}).get(task, {})
                label = sample_item.get("label")
                task_example[task] = {
                    'label': -1 if label is None else label,
                    'text_label': sample_item.get("text", ""),
                }

            self.examples.append({
                'id':slide_id, 
                'image_path': image_path, 
                'image_plip_path': image_plip_path,
                'tasks': task_example,
            })

        print(f'The size of {self.split} dataset: {len(self.examples)} for tasks: {self.tasks}')

    def _load_split_json(self, split: str):
        if split == "train":
            if not self.train_json:
                raise ValueError("args.train_json is required for split='train'")
            
            with open(self.train_json, "r", encoding="utf-8") as f:
                return json.load(f)

        if split == "val":
            if not self.val_json:
                raise ValueError("args.val_json is required for split='val'")
            
            with open(self.val_json, "r", encoding="utf-8") as f:
                return json.load(f)

        if split == "test":
            if not self.test_json:
                raise ValueError("args.test_json is required for split='test'")
            
            with open(self.test_json, "r", encoding="utf-8") as f:
                return json.load(f)

        raise ValueError(f"Unsupported split: {split}")
    
    def __len__(self):
        return len(self.examples)
    
class TcgaImageClassificationDataset(BaseDatasetClassification):
    def __getitem__(self, idx):
        example = self.examples[idx]

        ###################
        image_id = example['id']
        image_path = example['image_path']
        image_plip_path = example["image_plip_path"]
        
        image = load_tensor_or_keyed_tensor(image_path)[:self.max_fea_length]

        if image_plip_path is not None: 
            
            # if image_plip_path in self.image_cache_plip:
            #     image_plip = self.image_cache_plip[image_plip_path]
            # else:
            #     image_plip = torch.load(image_plip_path)[:self.max_fea_length]
            #     self.image_cache_plip[image_plip_path] = image_plip

            image_plip = load_tensor_or_keyed_tensor(image_plip_path)[:self.max_fea_length]

            # skip sample
            if image.shape[0] != image_plip.shape[0]:
                print(
                    f"Patch number mismatch for slide {image_path.split('/')[-1]}: "
                    f"image={image.shape}, image_plip={image_plip.shape}, "
                    f"img_path={image_path}, plip_path={image_plip_path}"
                )
                return None

            image = torch.cat((image, image_plip),dim=1) #(N_patch, 1536)


        task_dict = {}
        for task in self.tasks:
            task_dict[task] = {
                'label': example['tasks'][task]['label'], 
                'text_label': example['tasks'][task]['text_label'],
            }

        sample = {
            'input':{
                'features': image,
            },
            'id': image_id, 
            'image_path': image_path, 
            'tasks': task_dict,
        }

        return sample


    

    