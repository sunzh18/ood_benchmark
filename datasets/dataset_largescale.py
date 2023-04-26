from torch.utils.data import Dataset
from PIL import Image
import io
import os
import random
import numpy as np
import pickle


def pil_loader(img_str, str='RGB'):
    with Image.open(img_str) as img:
        img = img.convert(str)
    return img


class DatasetWithMeta(Dataset):
    def __init__(self, root_dir, meta_file, transform=None):
        super(DatasetWithMeta, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        with open(meta_file) as f:
            lines = f.readlines()
        self.images = []
        self.cls_idx = []
        self.classes = set()

        for line in lines:
            segs = line.strip().split(' ')
            self.images.append(' '.join(segs[:-1]))
            self.cls_idx.append(int(segs[-1]))
            self.classes.add(int(segs[-1]))
        self.num = len(self.images)
        # self.classes = len(self.cls_set)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = os.path.join(self.root_dir, self.images[idx])

        try:
            img = pil_loader(filename)
        except:
            print(filename)
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # transform
        if self.transform is not None:
            img = self.transform(img)
        return img, self.cls_idx[idx]


class DatasetWithMetaGroup(Dataset):
    def __init__(self, root_dir, meta_file, transform=None, num_group=8):
        super(DatasetWithMetaGroup, self).__init__()
        self.root_dir = root_dir
        self.transform = transform
        with open(meta_file) as f:
            lines = f.readlines()
        self.images = []
        self.cls_idx = []
        self.classes = set()
        self.num_group = num_group

        for line in lines:
            segs = line.strip().split(' ')
            self.images.append(' '.join(segs[:-2]))

            group_idx = int(segs[-2])
            sub_cls_idx = int(segs[-1])

            self.cls_idx.append((group_idx, sub_cls_idx))
            self.classes.add((group_idx, sub_cls_idx))

        self.num = len(self.images)
        # self.classes = len(self.cls_set)

    def __len__(self):
        return self.num

    def __getitem__(self, idx):
        filename = os.path.join(self.root_dir, self.images[idx])

        try:
            img = pil_loader(filename)
        except:
            print(filename)
            return self.__getitem__(random.randint(0, self.__len__() - 1))

        # transform
        if self.transform is not None:
            img = self.transform(img)

        group_id, cls_id = self.cls_idx[idx]
        labels = np.zeros(self.num_group, dtype=np.int)
        labels[group_id] = cls_id + 1

        return img, labels


class PartialDataset(Dataset):
    def __init__(self, parent_ds, offset, length):
        self.parent_ds = parent_ds
        self.offset = offset
        self.length = length
        assert len(parent_ds) >= offset + length, Exception("Parent Dataset not long enough")
        super(PartialDataset, self).__init__()

    def __len__(self):
        return self.length

    def __getitem__(self, i):
        return self.parent_ds[i + self.offset]


def validation_split(dataset, val_share=0.1):
    """
       Split a (training and vaidation combined) dataset into training and validation.
       Note that to be statistically sound, the items in the dataset should be statistically
       independent (e.g. not sorted by class, not several instances of the same dataset that
       could end up in either set).

       inputs:
          dataset:   ("training") dataset to split into training and validation
          val_share: fraction of validation data (should be 0<val_share<1, default: 0.1)
       returns: input dataset split into test_ds, val_ds

    """
    val_offset = int(len(dataset) * (1 - val_share))
    return PartialDataset(dataset, 0, val_offset), PartialDataset(dataset, val_offset, len(dataset) - val_offset)