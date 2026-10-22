import json
import re
from collections import Counter
import os


def _iter_json_items(json_path: str):
    with open(json_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    for item in data:
        if isinstance(item, dict):
            yield item


def _get_text_from_item(item: dict) -> str:
    # test-style
    if item.get("text"):
        return item["text"]

    # train-style (conversations)
    conv = item.get("conversations", [])
    if isinstance(conv, list):
        for msg in conv:
            if msg.get("from") == "gpt" and msg.get("value"):
                return msg["value"]

    return ""


class Tokenizer(object):
    def __init__(self, args):
        self.ann_path =args.ann_path 
        self.threshold = args.threshold
        #self.dataset_name = args.dataset_name
        self.use_json_dataset = args.use_json_dataset
        self.train_json = getattr(args, "train_json", None)
        self.test_json = getattr(args, "test_json", None)
        self.val_json = getattr(args, "val_json", None)

        # Special tokens
        self.bos_token = '<bos>'
        self.eos_token = '<eos>'
        self.pad_token = '<pad>'
        self.unk_token = '<unk>'

        self.clean_report = self.clean_report_tcga

        if self.use_json_dataset:
            self.token2idx, self.idx2token = self.create_vocabulary_json()
        else:
            self.token2idx, self.idx2token = self.create_vocabulary()


        self.pad_idx = self.token2idx[self.pad_token]  
        self.bos_idx = self.token2idx[self.bos_token]  
        self.eos_idx = self.token2idx[self.eos_token]   

    
    def create_vocabulary_json(self):
        if not self.train_json:
            raise ValueError("train_json is required to build the vocabulary.")

        total_tokens = []
        for item in _iter_json_items(self.train_json):
            anno = _get_text_from_item(item)
            if anno:
                total_tokens.extend(self.clean_report(anno).split())

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        vocab.sort()

        token2idx = {token: idx for idx, token in enumerate(vocab)}
        idx2token = {idx: token for idx, token in enumerate(vocab)}
        return token2idx, idx2token
    
    def create_vocabulary(self):
        # NOTE: following the original MiGen/HistGen/BiGen implementations. Kept as is to reproduce the paper results
        total_tokens = []
        root = self.ann_path

        for dir in os.listdir(root):
            file_name = os.path.join(root, dir, 'annotation')

            with open(file_name, 'r') as f:
                anno = f.read()

            tokens = self.clean_report(anno).split()

            for token in tokens:
                total_tokens.append(token)

        counter = Counter(total_tokens)
        vocab = [k for k, v in counter.items() if v >= self.threshold] + [self.pad_token, self.bos_token, self.eos_token, self.unk_token]
        vocab.sort()
        token2idx, idx2token = {}, {}

        for idx, token in enumerate(vocab):
            token2idx[token] = idx 
            idx2token[idx] = token

        return token2idx, idx2token

    def clean_report_tcga(self, report):
        report_cleaner = lambda t: (t.replace('\n', ' ').replace('  ', ' ') \
            .replace('  ', ' ').replace('  ', ' ')\
            .replace(' 10. ', ' ').replace(' 11. ', ' ').replace(' 12. ', ' ').replace(' 13. ', ' ').replace(' 14.', ' ')    \
            .replace(' 1. ', ' ').replace(' 2. ', ' ') \
            .replace(' 3. ', ' ').replace(' 4. ', ' ').replace(' 5. ', ' ').replace(' 6. ', ' ').replace(' 7. ', ' ').replace(' 8. ', ' ') .replace(' 9. ', ' ')   \
            .strip().lower() + ' ').split('. ')
        
        # NOTE: inside the character class, ":-\[" is a range (':' to '['), so '<', '=', '>', '@'
        # are removed while '-' is kept. Inherited from the original code, kept for reproducibility.
        sent_cleaner = lambda t: re.sub(r'[#,?;*!^&_+():-\[\]{}]', '', t.replace('"', '').
                                    replace('\\', '').replace("'", '').strip().lower())
        tokens = [sent_cleaner(sent) for sent in report_cleaner(report) if sent_cleaner(sent) != []]
        report = ' . '.join(tokens) 
        return report

    def get_token_by_id(self, id):
        return self.idx2token[id]

    def get_id_by_token(self, token):
        if token not in self.token2idx:
            return self.token2idx['<unk>']
        return self.token2idx[token]

    def get_vocab_size(self):
        return len(self.token2idx)

    def __call__(self, report):
        tokens = self.clean_report(report).split()
        ids = []

        for token in tokens:
            ids.append(self.get_id_by_token(token))

        ids = [self.get_id_by_token(self.bos_token)] + ids + [self.get_id_by_token(self.eos_token)]
        return ids


    def decode(self, ids):
        tokens = []
        for idx in ids:
            # end of sequence
            if idx == self.eos_idx:
                break
            # skip padding and BOS
            if idx in (self.pad_idx, self.bos_idx):
                continue
            # Add token to list
            tokens.append(self.idx2token[idx])
        return ' '.join(tokens)


    def decode_batch(self, ids_batch):
        out = []
        for ids in ids_batch:
            out.append(self.decode(ids))
        return out
