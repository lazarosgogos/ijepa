import os
import subprocess
import time
import numpy as np
from logging import getLogger

import torch
import torchvision
from PIL import Image

logger = getLogger()


# -----------------------------
# CSV LABEL PARSER
# -----------------------------
def load_val_labels(csv_path):
    mapping = {}

    with open(csv_path, 'r') as f:
        next(f)
        for line in f:
            img_id, pred = line.strip().split(',', 1)
            class_id = pred.split(' ')[0]
            mapping[img_id] = class_id

    return mapping


# -----------------------------
# DATASET
# -----------------------------
class ImageNet(torch.utils.data.Dataset):

    def __init__(
        self,
        root,
        image_folder='imagenet_full_size/061417/',
        tar_file='imagenet_full_size-061417.tar.gz',
        transform=None,
        train=True,
        job_id=None,
        local_rank=None,
        copy_data=True,
        index_targets=False,
        train_suffix='train/',
        val_suffix='val/',
        val_label_csv=None,
        train_dir_override=None,
    ):
        self.transform = transform
        self.train = train

        suffix = train_suffix if train else val_suffix

        data_path = None
        if copy_data:
            data_path = copy_imgnt_locally(
                root=root,
                suffix=suffix,
                image_folder=image_folder,
                tar_file=tar_file,
                job_id=job_id,
                local_rank=local_rank
            )

        if (not copy_data) or (data_path is None):
            data_path = os.path.join(root, image_folder, suffix)

        logger.info(f'data-path {data_path}')

        # -----------------------------
        # TRAIN: use ImageFolder
        # -----------------------------
        if train:
            dataset = torchvision.datasets.ImageFolder(
                root=data_path,
                transform=transform
            )

            self.samples = dataset.samples
            self.targets = np.array([s[1] for s in dataset.samples])
            self.classes = dataset.classes
            self.class_to_idx = dataset.class_to_idx
            self.loader = dataset.loader

        # -----------------------------
        # VAL: custom loader
        # -----------------------------
        else:
            if val_label_csv is None:
                raise ValueError("val_label_csv is required for validation")

            val_map = load_val_labels(val_label_csv)

            # build class mapping from train dir
            # train_dir = os.path.join(root, image_folder) #, train_suffix)
            train_dir = train_dir_override if train_dir_override else os.path.join(root, image_folder)

            classes = sorted(
                entry.name for entry in os.scandir(train_dir) if entry.is_dir()
            )
            class_to_idx = {cls: i for i, cls in enumerate(classes)}

            self.classes = classes
            self.class_to_idx = class_to_idx
            self.loader = torchvision.datasets.folder.default_loader

            samples = []
            targets = []

            for fname, class_id in val_map.items():
                path = os.path.join(data_path, fname + ".JPEG")

                if not os.path.exists(path):
                    continue

                if class_id not in class_to_idx:
                    continue

                target = class_to_idx[class_id]
                samples.append((path, target))
                targets.append(target)

            self.samples = samples
            self.targets = np.array(targets)

            logger.info(f'Validation samples loaded: {len(self.samples)}')

        # -----------------------------
        # INDEX TARGETS (KNN)
        # -----------------------------
        if index_targets:
            self.target_indices = []
            for t in range(len(self.classes)):
                indices = np.where(self.targets == t)[0].tolist()
                self.target_indices.append(indices)

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.loader(path)

        if self.transform is not None:
            img = self.transform(img)

        return img, target


# -----------------------------
# SUBSET
# -----------------------------
class ImageNetSubset(object):

    def __init__(self, dataset, subset_file):
        self.dataset = dataset
        self.filter_dataset_(subset_file)

    def filter_dataset_(self, subset_file):
        new_samples = []

        with open(subset_file, 'r') as f:
            for line in f:
                img = line.strip()
                class_name = img.split('_')[0]
                target = self.dataset.class_to_idx[class_name]

                path = os.path.join(self.dataset.root, class_name, img)
                new_samples.append((path, target))

        self.samples = new_samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, index):
        path, target = self.samples[index]
        img = self.dataset.loader(path)

        if self.dataset.transform:
            img = self.dataset.transform(img)

        return img, target


# -----------------------------
# UNSUPERVISED LOADER
# -----------------------------
def make_imagenet1k(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    shuffle=False
):
    dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training,
        copy_data=copy_data,
        index_targets=False
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=collator,
        pin_memory=pin_mem,
        num_workers=num_workers,
    )

    return dataset, loader, sampler


# -----------------------------
# SUPERVISED LOADER
# -----------------------------
def make_imagenet1k_supervised(
    transform,
    batch_size,
    collator=None,
    pin_mem=True,
    num_workers=8,
    world_size=1,
    rank=0,
    root_path=None,
    image_folder=None,
    training=True,
    copy_data=False,
    drop_last=True,
    subset_file=None,
    shuffle=False,
    train_suffix="train/",
    val_suffix="val/",
    val_label_csv=None,
):

    dataset = ImageNet(
        root=root_path,
        image_folder=image_folder,
        transform=transform,
        train=training,
        copy_data=copy_data,
        index_targets=True,
        train_suffix=train_suffix,
        val_suffix=val_suffix,
        val_label_csv=val_label_csv,
    )

    sampler = torch.utils.data.distributed.DistributedSampler(
        dataset,
        num_replicas=world_size,
        rank=rank,
        shuffle=shuffle
    )

    loader = torch.utils.data.DataLoader(
        dataset,
        sampler=sampler,
        batch_size=batch_size,
        drop_last=drop_last,
        collate_fn=torch.utils.data.default_collate,
        pin_memory=pin_mem,
        num_workers=num_workers,
    )

    return dataset, loader, sampler


# -----------------------------
# COPY FUNCTION
# -----------------------------
def copy_imgnt_locally(
    root,
    suffix,
    image_folder='imagenet_full_size/061417/',
    tar_file='imagenet_full_size-061417.tar.gz',
    job_id=None,
    local_rank=None
):
    if job_id is None or local_rank is None:
        return None

    source_file = os.path.join(root, tar_file)
    target = f'/scratch/slurm_tmpdir/{job_id}/'
    data_path = os.path.join(target, image_folder, suffix)

    signal = os.path.join(target, 'copy_signal.txt')

    if not os.path.exists(data_path):
        if local_rank == 0:
            subprocess.run(['tar', '-xf', source_file, '-C', target])
            with open(signal, 'w') as f:
                f.write('done')
        else:
            while not os.path.exists(signal):
                time.sleep(30)

    return data_path
