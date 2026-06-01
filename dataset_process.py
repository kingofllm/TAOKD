import os
import io
import glob
import torch
import pandas as pd
import pyarrow.parquet as pq
from typing import Dict, Sequence
from dataclasses import dataclass
from torchvision import transforms
from torch.utils.data import IterableDataset, DataLoader, get_worker_info
import numpy as np
import random
import cv2
import math

cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

class StreamingSupervisedDataset(IterableDataset):
    def __init__(self, parquet_files, dataset_type="cifar", transform=None, use_coarse=False):
        if isinstance(parquet_files, str):
            parquet_files = [parquet_files]

        self.parquet_files = parquet_files
        self.dataset_type = dataset_type
        self.transform = transform
        self.use_coarse = use_coarse

        self.total_len = 0
        print(f"[INFO] Scanning {len(parquet_files)} files to calculate total length...")
        for f in parquet_files:
            self.total_len += pq.read_metadata(f).num_rows
        print(f"[INFO] Total samples: {self.total_len}")

    def __len__(self):
        return self.total_len

    def _process_image(self, img_bytes):
        np_arr = np.frombuffer(img_bytes, np.uint8)
        img = cv2.imdecode(np_arr, cv2.IMREAD_COLOR)
        if img is None:
            return None
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        return img

    def __iter__(self):
        worker_info = get_worker_info()
        if worker_info is None:
            my_files = self.parquet_files
        else:
            my_files = [
                f for i, f in enumerate(self.parquet_files)
                if i % worker_info.num_workers == worker_info.id
            ]

        for file_path in my_files:
            # print(f"[Worker {worker_info.id if worker_info else 0}] Loading: {os.path.basename(file_path)}")

            try:
                df = pq.read_table(file_path).to_pandas()
                df = df.sample(frac=1.0).reset_index(drop=True)
                for _, row in df.iterrows():
                    if "img" in row:
                        img_bytes = row["img"]["bytes"]
                    elif "image" in row:
                        img_bytes = row["image"]["bytes"]
                    else:
                        continue

                    img_array = self._process_image(img_bytes)
                    if img_array is None:
                        continue

                    from PIL import Image
                    img = Image.fromarray(img_array)

                    if self.transform:
                        img = self.transform(img)
                    result = {"image": img, "idx": -1}

                    if self.dataset_type == "cifar":
                        result["fine_idx"] = int(row.get("fine_label", -1))
                        result["coarse_idx"] = int(row.get("coarse_label", -1))
                    elif self.dataset_type in ["stanford-cars", "tiny-imagenet", "nabirds"]:
                        result["labels"] = int(row.get("label", -1))
                    yield result

                del df

            except Exception as e:
                print(f"Error loading file {file_path}: {e}")
                continue


@dataclass
class DataCollatorForSupervisedDataset:
    def __call__(self, instances: Sequence[Dict]) -> Dict[str, torch.Tensor]:
        instances = [i for i in instances if i is not None]
        if len(instances) == 0:
            return None

        images = torch.stack([inst["image"] for inst in instances])
        keys = instances[0].keys()
        batch = {"images": images}
        if "fine_idx" in keys:
            batch["fine_idx"] = torch.tensor([inst["fine_idx"] for inst in instances])
        if "coarse_idx" in keys:
            batch["coarse_idx"] = torch.tensor([inst["coarse_idx"] for inst in instances])
        if "labels" in keys:
            batch["labels"] = torch.tensor([inst["labels"] for inst in instances])

        return batch

transform_cifar_train = transforms.Compose([
    transforms.RandomCrop(32, padding=4),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

transform_cifar_test = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.5071, 0.4867, 0.4408], std=[0.2675, 0.2565, 0.2761])
])

transform_stanford_cars_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_stanford_cars_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_tiny_imagenet_train = transforms.Compose([
    transforms.RandomResizedCrop(64),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_tiny_imagenet_test = transforms.Compose([
    transforms.Resize(64),
    transforms.CenterCrop(64),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_nabirds_train = transforms.Compose([
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

transform_nabirds_test = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])


def make_supervised_data_module(data_args) -> Dict:
    dataset_type = data_args.dataset_type.lower()

    def worker_init_fn(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    if dataset_type == "cifar":
        train_files = sorted(glob.glob(os.path.join(data_args.train_dir, "*.parquet")))
        test_files = sorted(glob.glob(os.path.join(data_args.test_dir, "*.parquet")))

        if len(train_files) == 0:
            raise ValueError(f"No .parquet files found in {data_args.train_dir}")

        print(f"[INFO] Found {len(train_files)} training files.")

        train_dataset = StreamingSupervisedDataset(
            parquet_files=train_files,
            dataset_type="cifar",
            transform=transform_cifar_train,
            use_coarse=data_args.use_coarse
        )
        test_dataset = StreamingSupervisedDataset(
            parquet_files=test_files,
            dataset_type="cifar",
            transform=transform_cifar_test,
            use_coarse=data_args.use_coarse
        )

    elif dataset_type == "stanford-cars":
        train_files = sorted(glob.glob(os.path.join(data_args.train_dir, "*.parquet")))
        test_files = sorted(glob.glob(os.path.join(data_args.test_dir, "*.parquet")))

        if len(train_files) == 0:
            raise ValueError(f"No .parquet files found in {data_args.train_dir}")

        print(f"[INFO] Found {len(train_files)} training files.")

        train_dataset = StreamingSupervisedDataset(
            parquet_files=train_files,
            dataset_type=dataset_type,
            transform=transform_stanford_cars_train
        )
        test_dataset = StreamingSupervisedDataset(
            parquet_files=test_files,
            dataset_type=dataset_type,
            transform=transform_stanford_cars_test
        )
    elif dataset_type == "tiny-imagenet":
        train_files = sorted(glob.glob(os.path.join(data_args.train_dir, "*.parquet")))
        test_files = sorted(glob.glob(os.path.join(data_args.test_dir, "*.parquet")))
        if len(train_files) == 0:
            raise ValueError(f"No .parquet files found in {data_args.train_dir}")

        print(f"[INFO] Found {len(train_files)} training files.")

        train_dataset = StreamingSupervisedDataset(
            parquet_files=train_files,
            dataset_type=dataset_type,
            transform=transform_tiny_imagenet_train
        )
        test_dataset = StreamingSupervisedDataset(
            parquet_files=test_files,
            dataset_type=dataset_type,
            transform=transform_tiny_imagenet_test
        )
    elif dataset_type == "nabirds":
        train_files = sorted(glob.glob(os.path.join(data_args.train_dir, "*.parquet")))
        test_files = sorted(glob.glob(os.path.join(data_args.test_dir, "*.parquet")))
        if len(train_files) == 0:
            raise ValueError(f"No .parquet files found in {data_args.train_dir}")

        print(f"[INFO] Found {len(train_files)} training files.")

        train_dataset = StreamingSupervisedDataset(
            parquet_files=train_files,
            dataset_type=dataset_type,
            transform=transform_nabirds_train
        )
        test_dataset = StreamingSupervisedDataset(
            parquet_files=test_files,
            dataset_type=dataset_type,
            transform=transform_nabirds_test
        )
    else:
        raise ValueError(f"Unsupported dataset type: {dataset_type}")

    data_collator = DataCollatorForSupervisedDataset()

    train_dataloader = DataLoader(
        train_dataset,
        batch_size=data_args.batch_size,
        shuffle=False,
        num_workers=data_args.num_workers,
        collate_fn=data_collator,
        pin_memory=True,
        worker_init_fn=worker_init_fn,
        drop_last=True
    )

    test_dataloader = DataLoader(
        test_dataset,
        batch_size=data_args.batch_size,
        shuffle=False,
        num_workers=data_args.num_workers,
        collate_fn=data_collator,
        pin_memory=True
    )

    return {"train_dataloader": train_dataloader, "test_dataloader": test_dataloader}