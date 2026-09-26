# Copyright (c) Meta Platforms, Inc. and affiliates.
# This software may be used and distributed according to the terms of the Llama 2 Community License Agreement.
"""Whole-conversation packing and fixed left-padding (Appendix C.3)."""
import torch
from torch.utils.data import Dataset


class BucketPaddingCollator:
    def __init__(self, base_collator, pad_token_id, max_length=16384):
        self.pad_token_id = pad_token_id
        self.max_length = max_length

    def __call__(self, features):
        batch = {key: [] for key in ('input_ids', 'attention_mask', 'labels')}
        for sample in features:
            length = len(sample['input_ids'])
            if not 0 < length <= self.max_length:
                raise ValueError(f'Sample length {length} exceeds context limit {self.max_length}')
            padding = self.max_length-length
            for key, value in [('input_ids', self.pad_token_id), ('attention_mask', 0), ('labels', -100)]:
                batch[key].append([value]*padding + list(sample[key]))
        return {key: torch.tensor(value, dtype=torch.long) for key, value in batch.items()}


class ConcatDataset(Dataset):
    def __init__(self, dataset, chunk_size=16384, pad_token_id=None):
        self.samples = []
        self.dropped_too_long = 0
        buffer = {key: [] for key in ('input_ids','attention_mask','labels')}
        for sample in dataset:
            sample = {key: list(sample[key]) for key in buffer}
            if len(sample['input_ids']) > chunk_size:
                self.dropped_too_long += 1
                continue
            if not sample['input_ids']:
                continue
            # Flipped samples already end in a masked pad separator.
            if pad_token_id is not None and sample['input_ids'][-1] != pad_token_id:
                sample['input_ids'].append(pad_token_id)
                sample['attention_mask'].append(1)
                sample['labels'].append(-100)
            if len(sample['input_ids']) > chunk_size:
                self.dropped_too_long += 1
                continue
            if buffer['input_ids'] and len(buffer['input_ids'])+len(sample['input_ids']) > chunk_size:
                self.samples.append(buffer)
                buffer = {key: [] for key in buffer}
            for key in buffer:
                buffer[key].extend(sample[key])
        if buffer['input_ids']:
            self.samples.append(buffer)

    def __getitem__(self, idx):
        return self.samples[idx]

    def __len__(self):
        return len(self.samples)


class PadForDistributedEvaluation(Dataset):
    """Equal numbers of FSDP forwards, without duplicating or discarding scored tokens."""
    def __init__(self, dataset, multiple, pad_token_id):
        self.dataset = dataset
        self.multiple = multiple
        self.pad_token_id = pad_token_id

    def __len__(self):
        return ((len(self.dataset)+self.multiple-1)//self.multiple)*self.multiple

    def __getitem__(self, index):
        if index < len(self.dataset):
            return self.dataset[index]
        return {'input_ids':[self.pad_token_id], 'attention_mask':[1], 'labels':[-100]}
