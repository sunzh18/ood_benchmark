from utils import log
from models.resnetv2 import * 
import torch
import torch.nn as nn
from torchsummary import summary
import time
import csv
import numpy as np

from utils.test_utils import mk_id_ood, get_measures
import os

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from utils.mahalanobis_lib import get_Mahalanobis_score
from utils.data_loader import get_dataloader_in, get_dataloader_out, cifar_out_datasets, imagenet_out_datasets
from utils.model_loader import get_model

from utils.cal_score import *
from argparser import *

import torchvision
from torchvision import transforms
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
from scipy.stats import norm
from scipy.stats import laplace
import torch.nn.functional as F


def get_features(args, model, dataloader):
    features = []
    for b, (x, y) in enumerate(dataloader):
        with torch.no_grad():
            x = x.cuda()            
            # print(x.size())
            # feature = model.forward_features(x)
            outputs = model.features(x)
            feature = F.adaptive_avg_pool2d(outputs, 1)
            feature = feature.view(feature.size(0), -1)
            # print(feature.size())
            features.extend(feature.data.cpu().numpy())
            # x = feature[feature>=0]
            # print(x.size())

    features = np.array(features)
    # x = np.transpose(features)
    print(features.shape)

    return features


def extact_mean_std(args, model):
    for key, v in model.state_dict().items():
        # resnet
        if key == 'layer4.1.bn2.weight':
            # print(f'var: {v}')
            std = v
        if key == 'layer4.1.bn2.bias':
            # print(f'mean: {v}')
            mean = v
        

        # print(f'{key}')
        #wideresnet, densenet
        if key == 'bn1.weight':
            # print(f'var: {v}')
            std = v
        if key == 'bn1.bias':
            # print(f'mean: {v}')
            mean = v

    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    if not os.path.isdir(file_folder):
        os.makedirs(file_folder)
    
    torch.save(mean, f'{file_folder}/{args.model}_features_mean.pt')
    torch.save(std, f'{file_folder}/{args.model}_features_std.pt')
    print(mean)
    print(std)

kwargs = {'num_workers': 2, 'pin_memory': True}
imagesize = 32
transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def main(args):
    if args.in_dataset == "CIFAR-10":
        data_path = '/data/Public/Datasets/cifar10'
        # Data loading code
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform_test)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, **kwargs)

        num_classes = 10

    elif args.in_dataset == "CIFAR-100":
        data_path = '/data/Public/Datasets/cifar100'
        # Data loading code
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False, transform=transform_test)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, **kwargs)

        num_classes = 100

    elif args.in_dataset == "imagenet100":
        root = '/data/Public/Datasets/ImageNet100'
        # Data loading code
        trainset = torchvision.datasets.ImageFolder(os.path.join(root, 'train'), transform_test_largescale)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, **kwargs)

        num_classes = 100

       
    elif args.in_dataset == "imagenet":
        root = '/data/Public/Datasets/ilsvrc2012'
        # Data loading code
        trainset = torchvision.datasets.ImageFolder(os.path.join(root, 'train'), transform_test_largescale)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=True, **kwargs)

        num_classes = 1000

    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=False)
    checkpoint = torch.load(
            f'{args.model_path}/{args.name}/{args.in_dataset}/{args.model}_parameter.pth.tar')

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()
    extact_mean_std(args, model)

    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    if not os.path.isdir(file_folder):
        os.makedirs(file_folder)
    
    features = get_features(args, model, train_dataloader)

    np.save(f"{file_folder}/{args.model}_feat_stat.npy", features.mean(0))
    # np.save(f"{file_folder}/{args.model}_feat_std.npy", features.std(0))

    # info = np.load(f"{args.in_dataset}_{args.model}_feat_stat.npy")
    # mean = features.mean(0)
    # print(mean.shape)

    # print(mean)
    # print(info)



    

    




if __name__ == "__main__":
    parser = get_argparser()

    
    main(parser.parse_args())
