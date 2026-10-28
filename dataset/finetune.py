import os
import wfdb
import numpy as np
import pandas as pd
from PIL import Image
from typing import Iterable, Literal, Optional

import torch
import torchvision.transforms as transforms
from torch.utils.data import Dataset, DataLoader

import util.transforms as T
from util.misc import get_rank, get_world_size
from util.transforms import get_transforms_from_config, get_rand_augment_from_config

class ImageSignal_Dataset(Dataset):
    def __init__(self, image_path, signal_path, csv_path, use_labels, image_transform, signal_transform, original_fs, target_fs=250):

        csv = pd.read_csv(csv_path)

        self.image_path = image_path
        self.signal_path = signal_path
        
        self.file_path = csv['filename_hr']
        
        if use_labels:
            self.labels = csv.iloc[:, 6:].values
            self.label_name = list(csv.columns[6:])
            
        # ptb-xl super class의 경우, norm을 포함한 멀티 레이블 제거
        self.file_list, self.gt_list = [], []
        for idx, filename in enumerate(self.file_path):
            label_sum = np.sum(self.labels[idx])
            if self.labels[idx][1] == 1.0 and label_sum > 1:
                continue
            self.file_list.append(filename)
            self.gt_list.append(self.labels[idx])
        
        self.image_transform = image_transform
        self.signal_transform = signal_transform

        self.original_fs = original_fs
        if original_fs != target_fs:
            self.resample = T.Resample(target_fs=target_fs)
        else:
            self.resample = None
        
        self.crop_area = (0, 450, 2200, 1700-70)

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, idx):
        file = self.file_list[idx]
        image_path = os.path.join(self.image_path, f'{file[11:]}-0.png')
        signal_path = os.path.join(self.signal_path, file)
        
        # image
        image = Image.open(image_path).convert('RGB')
        image = image.crop(self.crop_area)
        image = self.image_transform(image)
        
        # signal
        signal = wfdb.rdsamp(signal_path)[0]
        signal = signal.T
        if self.resample is not None:
            signal = self.resample(signal, self.original_fs)
        if self.signal_transform:
            signal = self.signal_transform(signal)
            
        if hasattr(self, 'gt_list'):
            label = torch.tensor(self.gt_list[idx], dtype= torch.float) # multi-class는 dtype=long
            return image, signal, label
        else:
            return image, signal


def build_dataset(cfg, split):

    signal_path = cfg.get('signal_path', None)
    image_path = cfg.get('image_path', None)
    csv_path = cfg.get(f'{split}_csv', None)
    labels = cfg.get('labels', None)
    original_fs = cfg.get('fs', 500)

    if split == "train":
        transform = get_transforms_from_config(cfg["train_transforms"])
        randaug_config = cfg.get("rand_augment", {})
        use_randaug = randaug_config.get("use", False)
        if use_randaug:
            randaug_kwargs = randaug_config.get("kwargs", {})
            transform.append(get_rand_augment_from_config(randaug_kwargs))
    else:
        transform = get_transforms_from_config(cfg["eval_transforms"])

    signal_transform = T.Compose(transform + [T.ToTensor()])
    image_transform = transforms.Compose([transforms.Resize((224, 224)),
                                          transforms.ToTensor()])

    dataset = ImageSignal_Dataset(image_path,
                                  signal_path,
                                  csv_path,
                                  labels,
                                  image_transform,
                                  signal_transform,
                                  original_fs,
                                  target_fs=250)

    return dataset

def get_dataloader(dataset: Dataset,
                   is_distributed: bool = False,
                   dist_eval: bool = False,
                   mode: Literal["train", "eval"] = "train",
                   **kwargs) -> DataLoader:
    is_train = mode == "train"
    if is_distributed and (is_train or dist_eval):
        num_tasks = get_world_size()
        global_rank = get_rank()
        if not is_train and len(dataset) % num_tasks != 0:
            print('Warning: Enabling distributed evaluation with an eval dataset not divisible by process number. '
                  'This will slightly alter validation results as extra duplicate entries are added to achieve '
                  'equal num of samples per-process.')
        # shuffle=True to reduce monitor bias even if it is for validation.
        # https://github.com/facebookresearch/mae/blob/main/main_finetune.py#L189
        sampler = torch.utils.data.distributed.DistributedSampler(dataset,
                                                                  num_replicas=num_tasks,
                                                                  rank=global_rank,
                                                                  shuffle=True)
    elif is_train:
        sampler = torch.utils.data.RandomSampler(dataset)
    else:
        sampler = torch.utils.data.SequentialSampler(dataset)

    return DataLoader(dataset,
                      sampler=sampler,
                      drop_last=is_train,
                      **kwargs)