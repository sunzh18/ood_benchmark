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


def main(args):
    in_dataset = args.in_dataset

    loader_in_dict = get_dataloader_in(args, split=('train','val'))
    train_dataloader, test_dataloader, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
    train_set, test_set = loader_in_dict.train_dataset, loader_in_dict.val_dataset
    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=False)
    checkpoint = torch.load(
            f'{args.model_path}/{args.name}/{args.in_dataset}/{args.model}_parameter.pth.tar')

    model.load_state_dict(checkpoint['state_dict'])
    model.eval()

    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    if not os.path.isdir(file_folder):
        os.makedirs(file_folder)
    
    features = get_features(args, model, train_dataloader)

    np.save(f"{file_folder}/{args.model}_feat_mean.npy", features.mean(0))
    np.save(f"{file_folder}/{args.model}_feat_std.npy", features.std(0))

    info = np.load(f"{args.in_dataset}_{args.model}_feat_stat.npy")
    mean = features.mean(0)
    print(mean.shape)

    print(mean)
    print(info)



    

    




if __name__ == "__main__":
    parser = get_argparser()

    
    main(parser.parse_args())
