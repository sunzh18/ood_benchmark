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

def extact_mean_std(args, model):
    for key, v in model.state_dict().items():
        # resnet
        if key == 'layer4.1.bn2.weight':
            # print(f'var: {v}')
            std = v
        if key == 'layer4.1.bn2.bias':
            # print(f'mean: {v}')
            mean = v
        
        #wideresnet
        # if key == 'bn1.weight':
        #     # print(f'var: {v}')
        #     std = v
        # if key == 'bn1.bias':
        #     # print(f'mean: {v}')
        #     mean = v
    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    if not os.path.isdir(file_folder):
        os.makedirs(file_folder)
    
    torch.save(mean, f'{file_folder}/{args.model}_features_mean.pt')
    torch.save(std, f'{file_folder}/{args.model}_features_std.pt')
    print(mean)
    print(std)


def get_features(args, model, dataloader, channel=-1):
    features = []
    for b, (x, y) in enumerate(dataloader):
        with torch.no_grad():
            x = x.cuda()            
            # print(x.size())
            feature = model.forward_features(x)
            # print(feature.size())
            if channel == -1:
                f1 = feature[:,:5] 
            else:
                f1 = feature[:,channel]
            features.extend(f1.data.cpu().numpy())
    features = np.array(features)
    # x = np.transpose(features)
    print(features.shape, features)

    return features


def get_BATS_features(args, model, dataloader, lam, feature_std, feature_mean, channel=-1):
    features = []
    for b, (x, y) in enumerate(dataloader):
        with torch.no_grad():
            x = x.cuda()            
            # print(x.size())
            feature = model.forward_features(x)
            feature = torch.where(feature<(feature_std*lam+feature_mean),feature,feature_std*lam+feature_mean)
            feature = torch.where(feature>(-feature_std*lam+feature_mean),feature,-feature_std*lam+feature_mean)
            # print(feature.size())
            if channel == -1:
                f1 = feature[:,:5] 
            else:
                f1 = feature[:,channel]
            features.extend(f1.data.cpu().numpy())
    features = np.array(features)
    # x = np.transpose(features)
    print(features.shape, features)

    return features

def feature_dis(args):
    path = f'draw_analysis/feature/contrast_method/{args.model}'
    if not os.path.isdir(path):
        os.makedirs(path)

    in_dataset = args.in_dataset
    
    loader_in_dict = get_dataloader_in(args, split=('train','val'))
    train_dataloader, test_dataloader, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
    train_set, test_set = loader_in_dict.train_dataset, loader_in_dict.val_dataset
    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    baseline_model = get_model(args, num_classes, load_ckpt=False)
    checkpoint = torch.load(f'{args.model_path}/baseline/{args.in_dataset}/{args.model}_parameter.pth')
    baseline_model.load_state_dict(checkpoint['state_dict'])

    KD_random_model = get_model(args, num_classes, load_ckpt=False)
    checkpoint = torch.load(f'{args.model_path}/KD_random_init/{args.in_dataset}/{args.model}_parameter.pth')
    KD_random_model.load_state_dict(checkpoint['state_dict'])

    KD_teach_model = get_model(args, num_classes, load_ckpt=False)
    checkpoint = torch.load(f'{args.model_path}/KD_teacher_init/{args.in_dataset}/{args.model}_parameter.pth')
    KD_teach_model.load_state_dict(checkpoint['state_dict'])


    baseline_model.eval()
    KD_random_model.eval()
    KD_teach_model.eval()

    channel = 1
    base_features = get_features(args, baseline_model, test_dataloader, channel)
    KD_random_features = get_features(args, KD_random_model, test_dataloader, channel)
    KD_teach_features = get_features(args, KD_teach_model, test_dataloader, channel)

    save_pic_filename=f'{path}/{args.in_dataset}_{channel}_channel.png'
    if channel == -1:
        save_pic_filename=f'{path}/{args.in_dataset}_kdeplot.png'
    # plt.figure(figsize=(8, 4))
    ax = sns.kdeplot(data=base_features, label='base', fill=True, common_norm=True, alpha=.5, linewidth=0, legend=True)
    sns.kdeplot(data=KD_random_features, label='KD_random_init', fill=True, common_norm=True, alpha=.5, linewidth=0, legend=True)
    sns.kdeplot(data=KD_teach_features, label='KD_teacher_init', fill=True, common_norm=True, alpha=.5, linewidth=0, legend=True)
    plt.xlabel("feature act")
    ax.legend(loc="upper right")
    # plt.legend(pic,["sinx","cosx"],shadow=True,fancybox="blue")
    # plt.xlim(0, m * 10)
    plt.savefig(save_pic_filename,dpi=600)
    
    plt.close() 

def BATS_feature_dis(args):
    path = f'draw_analysis/feature/contrast_method_BATS/{args.model}'
    if not os.path.isdir(path):
        os.makedirs(path)

    in_dataset = args.in_dataset
    
    loader_in_dict = get_dataloader_in(args, split=('train','val'))
    train_dataloader, test_dataloader, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
    train_set, test_set = loader_in_dict.train_dataset, loader_in_dict.val_dataset
    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    if args.bats:
        if args.in_dataset == 'imagenet':
            feature_std=torch.load(f"checkpoints/feature/vit_features_std.pt").cuda()
            feature_mean=torch.load(f"checkpoints/feature/vit_features_mean.pt").cuda()        
            lam = 1.05
        elif args.in_dataset == 'CIFAR-10':
            feature_std=torch.load(f"checkpoints/feature/baseline/{args.in_dataset}/{args.model}_features_std.pt").cuda()
            feature_mean=torch.load(f"checkpoints/feature/baseline/{args.in_dataset}/{args.model}_features_mean.pt").cuda()
            # lam = 3.25
            if args.model == 'wrn':
                lam = 2.25
            elif args.model == 'resnet18':
                lam = 3.3

        elif args.in_dataset == 'CIFAR-100':
            feature_std=torch.load(f"checkpoints/feature/baseline/{args.in_dataset}/{args.model}_features_std.pt").cuda()
            feature_mean=torch.load(f"checkpoints/feature/baseline/{args.in_dataset}/{args.model}_features_mean.pt").cuda()
            # lam = 3.25
            if args.model == 'wrn':
                lam = 1.5
            elif args.model == 'resnet18':
                lam = 1.35

    baseline_model = get_model(args, num_classes, load_ckpt=False)
    checkpoint = torch.load(f'{args.model_path}/baseline/{args.in_dataset}/{args.model}_parameter.pth')
    baseline_model.load_state_dict(checkpoint['state_dict'])

    KD_random_model = get_model(args, num_classes, load_ckpt=False)
    checkpoint = torch.load(f'{args.model_path}/KD_random_init/{args.in_dataset}/{args.model}_parameter.pth')
    KD_random_model.load_state_dict(checkpoint['state_dict'])

    KD_teach_model = get_model(args, num_classes, load_ckpt=False)
    checkpoint = torch.load(f'{args.model_path}/KD_teacher_init/{args.in_dataset}/{args.model}_parameter.pth')
    KD_teach_model.load_state_dict(checkpoint['state_dict'])


    baseline_model.eval()
    KD_random_model.eval()
    KD_teach_model.eval()

    channel = 1
    base_features = get_BATS_features(args, baseline_model, test_dataloader, lam, feature_std, feature_mean ,channel)
    KD_random_features = get_features(args, KD_random_model, test_dataloader, channel)
    KD_teach_features = get_features(args, KD_teach_model, test_dataloader, channel)

    save_pic_filename=f'{path}/{args.in_dataset}_{channel}_channel.png'
    if channel == -1:
        save_pic_filename=f'{path}/{args.in_dataset}_kdeplot.png'
    # plt.figure(figsize=(8, 4))
    ax = sns.kdeplot(data=base_features, label='base_BATS', fill=True, common_norm=True, alpha=.5, linewidth=0, legend=True)
    sns.kdeplot(data=KD_random_features, label='KD_random_init', fill=True, common_norm=True, alpha=.5, linewidth=0, legend=True)
    sns.kdeplot(data=KD_teach_features, label='KD_teacher_init', fill=True, common_norm=True, alpha=.5, linewidth=0, legend=True)
    plt.xlabel("feature act")
    ax.legend(loc="upper right")
    # plt.legend(pic,["sinx","cosx"],shadow=True,fancybox="blue")
    # plt.xlim(0, m * 10)
    plt.savefig(save_pic_filename,dpi=600)
    
    plt.close() 

def analysis_feature(args, model, dataloader, channel=-1):

    features = get_features(args, model, dataloader, channel)
    # fmean = np.mean(features, axis=0)
    # m = min(fmean)
    # print(fmean.shape, fmean)
    # print(m)
    path = f'draw_analysis/feature/{args.name}/{args.model}'
    if not os.path.isdir(path):
        os.makedirs(path)
    save_pic_filename=f'{path}/{args.in_dataset}_kdeplot_{channel}_channel.png'
    if channel == -1:
        save_pic_filename=f'{path}/{args.in_dataset}_kdeplot.png'
    # plt.figure(figsize=(8, 4))
    sns.kdeplot(data=features, label='channel', fill=True, common_norm=True, alpha=.5, linewidth=0)
    plt.xlabel("feature act")
    # plt.xlim(0, m * 10)
    plt.savefig(save_pic_filename,dpi=600)
    
    plt.close()


    

# sns.kdeplot

# sns.set_palette("hls")
# #sns.set_style("whitegrid")
# plt.figure(dpi=120)
# sns.set_style("whitegrid")
# sns.set_style("dark")
# sns.set_style("darkgrid")
# sns.set_style("white")
# sns.set_style("ticks")
# save_pic_filename='sns_kdeplot.png'
# plt.figure(figsize=(8, 4))
# x=np.random.normal(size=100)
# print(x.shape)
# sns.kdeplot(x,fill=True)
# plt.savefig(save_pic_filename,dpi=600)
# plt.close() 

# sns.set(style='dark')
# sns.set_style("dark", {"axes.facecolor": "#e9f3ea"})







def main(args):
    in_dataset = args.in_dataset

    loader_in_dict = get_dataloader_in(args, split=('train','val'))
    train_dataloader, test_dataloader, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
    train_set, test_set = loader_in_dict.train_dataset, loader_in_dict.val_dataset
    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=load_ckpt)

    if args.in_dataset == 'CIFAR-10':
        # feature_std=torch.load("checkpoints/feature/BATS/CIFAR-10/resnet18_features_std.pt").cuda()
        # feature_mean=torch.load("checkpoints/feature/BATS/CIFAR-10/resnet18_features_mean.pt").cuda()

        # feature_std2=torch.load("checkpoints/feature/KD_teacher_init/CIFAR-10/resnet18_features_std.pt").cuda()
        # feature_mean2=torch.load("checkpoints/feature/KD_teacher_init/CIFAR-10/resnet18_features_mean.pt").cuda()
        lam = 3.25

    # print(feature_std - feature_std2)
    # print(feature_mean - feature_mean2)
    model.eval()

    
    channel = 2
    analysis_feature(args, model, test_dataloader, channel)











    # print(model)
    # for name, p in model.named_parameters():
    #     print(f'{name}')
    # extact_mean_std(args, model)
    # for key, v in model.state_dict().items():
    #     if key == 'layer4.1.bn2.weight':
    #         # print(f'var: {v}')
    #         std = v
    #     if key == 'layer4.1.bn2.bias':
    #         # print(f'mean: {v}')
    #         mean = v
    #     if key == 'bn1.weight':
    #         # print(f'var: {v}')
    #         std = v
    #     if key == 'bn1.bias':
    #         # print(f'mean: {v}')
    #         mean = v
        # if key == 'layer4.1.bn2.running_mean':
        #     print(f'running mean: {v}')
        # if key == 'layer4.1.bn2.running_var':
        #     print(f'running var: {v}')
        # print(f'{key}: {v.size()}')
    # print(feature_std.size(), feature_mean.size())

    # print(feature_mean, mean, feature_mean - mean)
    # print(feature_std, std, feature_std - std)

    

    




if __name__ == "__main__":
    parser = get_argparser()
    if parser.parse_args().bats:
        BATS_feature_dis(parser.parse_args())
    else:  
        feature_dis(parser.parse_args())

    
    # main(parser.parse_args())
