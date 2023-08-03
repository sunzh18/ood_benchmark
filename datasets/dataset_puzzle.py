import torch.utils.data as data
import os
import torch
import os.path
import numpy as np
from torchvision.datasets.utils import check_integrity, download_url
import os.path
import pickle
from typing import Any, Callable, Optional, Tuple

import numpy as np
from PIL import Image

from torchvision.datasets import CIFAR10, CIFAR100, ImageFolder
import random


# 将图像切分成3x3块
def split_image(img):
    w, h = img.size
    tile_size = w//3, h//3
    tiles = [img.crop((i*tile_size[0], j*tile_size[1], (i+1)*tile_size[0], (j+1)*tile_size[1])) 
             for i in range(3) for j in range(3)]
    return tiles

# 随机打乱图像块
def shuffle_tiles(tiles):
    random.shuffle(tiles)
    return tiles

# 将图像块拼接生成新的图像
def recompose_image(shuffled_tiles):
    new_img = Image.new('RGB', (shuffled_tiles[0].size[0] * 3, shuffled_tiles[0].size[1] * 3))  
    for i in range(9):
        new_img.paste(shuffled_tiles[i], (i%3 * shuffled_tiles[i].size[0], i//3 * shuffled_tiles[i].size[1]))
    return new_img

# 读入原始图像
# img = Image.open('image.jpg') 

# 拼图变换  
# tiles = split_image(img)
# shuffled_tiles = shuffle_tiles(tiles)
# new_img = recompose_image(shuffled_tiles)

# 保存新图像
# new_img.save('puzzle_image.jpg')

class Puzzle_CIFAR10(CIFAR10):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        # print(img.size)
        noise = np.random.normal(0, 0.3, (3, img.size[0], img.size[1]))
        
        # tiles = split_image(img)
        # shuffled_tiles = shuffle_tiles(tiles)
        # img = recompose_image(shuffled_tiles)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        img = img + torch.tensor(noise)
        img = img.float()
        return img, target


class Puzzle_CIFAR100(CIFAR100):

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, target) where target is index of the target class.
        """
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)
        # print(img.size)
        noise = np.random.normal(0, 0.3, (3, img.size[0], img.size[1]))
        
        # tiles = split_image(img)
        # shuffled_tiles = shuffle_tiles(tiles)
        # img = recompose_image(shuffled_tiles)
        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)
        img = img + torch.tensor(noise)
        img = img.float()
        return img, target


class Puzzle_imagenet(ImageFolder):     

    def __getitem__(self, index: int) -> Tuple[Any, Any]:
        """
        Args:
            index (int): Index

        Returns:
            tuple: (sample, target) where target is class_index of the target class.
        """
        path, target = self.samples[index]
        sample = self.loader(path)
        
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        # print(sample.shape)
        noise = np.random.normal(0, 0.5, sample.shape)
        sample = sample + torch.tensor(noise)
        sample = sample.float()
        return sample, target   