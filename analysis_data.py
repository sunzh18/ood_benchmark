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


def run_eval(model, in_loader, out_loader, args, num_classes, out_dataset):
    # switch to evaluate mode
    model.eval()

    if args.score == 'MSP':
        
        in_scores = iterate_data_msp(in_loader, model)
        
        out_scores = iterate_data_msp(out_loader, model)
    elif args.score == 'ODIN':
        
        in_scores = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, None)
        
        out_scores = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin, None)
    elif args.score == 'Energy':
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)    
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy)
    elif args.score == 'Mahalanobis':
        sample_mean, precision, lr_weights, lr_bias, magnitude = np.load(
            os.path.join(args.mahalanobis_param_path, 'results.npy'), allow_pickle=True)
        sample_mean = [s.cuda() for s in sample_mean]
        precision = [p.cuda() for p in precision]

        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                   [0, 0, 1, 1])

        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        temp_x = torch.rand(2, 3, 480, 480)
        temp_x = Variable(temp_x).cuda()
        temp_list = model(x=temp_x, layer_index='all')[1]
        num_output = len(temp_list)

        
        in_scores = iterate_data_mahalanobis(in_loader, model, num_classes, sample_mean, precision,
                                             num_output, magnitude, regressor, None)
        
        out_scores = iterate_data_mahalanobis(out_loader, model, num_classes, sample_mean, precision,
                                              num_output, magnitude, regressor, None)
    elif args.score == 'KL_Div':
        
        in_dist_logits, in_labels = iterate_data_kl_div(in_loader, model)
        
        out_dist_logits, _ = iterate_data_kl_div(out_loader, model)

        class_mean_logits = []
        for c in range(num_classes):
            selected_idx = (in_labels == c)
            selected_logits = in_dist_logits[selected_idx]
            class_mean_logits.append(np.mean(selected_logits, axis=0))
        class_mean_logits = np.array(class_mean_logits)

        in_scores = []
        for i, logit in enumerate(in_dist_logits):
            min_div = float('inf')
            for c_mean in class_mean_logits:
                cur_div = kl(logit, c_mean)
                if cur_div < min_div:
                    min_div = cur_div
            in_scores.append(-min_div)
        in_scores = np.array(in_scores)

        out_scores = []
        for i, logit in enumerate(out_dist_logits):
            min_div = float('inf')
            for c_mean in class_mean_logits:
                cur_div = kl(logit, c_mean)
                if cur_div < min_div:
                    min_div = cur_div
            out_scores.append(-min_div)
        out_scores = np.array(out_scores)
    else:
        raise ValueError("Unknown score type {}".format(args.score))

    # in_examples = in_scores.reshape((-1, 1))
    # out_examples = out_scores.reshape((-1, 1))

    # in_examples = in_scores
    # out_examples = out_scores

    print(in_scores.shape, out_scores.shape)
    return in_scores, out_scores



def analysis_score(args, in_examples, out_examples, out_dataset):
    # fmean = np.mean(features, axis=0)
    # m = min(fmean)
    # print(fmean.shape, fmean)
    # print(m)
    path = f'analysis_score/{args.name}/{args.in_dataset}/{args.model}/{args.score}'
    if not os.path.isdir(path):
        os.makedirs(path)
    save_pic_filename=f'{path}/{args.in_dataset}_{out_dataset}.png'
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data=in_examples, label=f'{args.in_dataset}', color='crimson', fill=True, common_norm=True, alpha=.5, linewidth=0, legend=True, ax = ax)
    sns.kdeplot(data=out_examples, label=f'{out_dataset}', color='limegreen', fill=True, common_norm=True, alpha=.5, linewidth=0, legend=True, ax = ax)
    plt.xlabel("score")
    ax.legend(loc="upper right")
    # plt.xlim(0, m * 10)
    plt.savefig(save_pic_filename,dpi=600)
    
    plt.close()


def get_features(args, model, dataloader):
    features = []
    for b, (x, y) in enumerate(dataloader):
        with torch.no_grad():
            x = x.cuda()            
            # print(x.size())
            feature = model.forward_features(x)
            # print(feature.size())
            features.extend(feature.data.cpu().numpy())
            # x = feature[feature>=0]
            # print(x.size())

    features = np.array(features)
    # x = np.transpose(features)
    # print(features.shape, features)

    return features
    
def analysis_actnum(args, model, in_loader, out_loader, out_dataset):
    model.eval()

    in_features = get_features(args, model, in_loader)
    out_features = get_features(args, model, out_loader)
    
    in_actnum = np.sum(np.greater(in_features, 0), axis=1)
    out_actnum = np.sum(np.greater(out_features, 0), axis=1)

    print(in_actnum.shape, out_actnum.shape)

    path = f'analysis_feature/actnum/{args.name}/{args.in_dataset}/{args.model}'
    if not os.path.isdir(path):
        os.makedirs(path)
    save_pic_filename=f'{path}/{args.in_dataset}_{out_dataset}.png'
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data=in_actnum, label=f'{args.in_dataset}', color='crimson', fill=True, common_norm=True, alpha=.5, linewidth=0, legend=True, ax = ax)
    sns.kdeplot(data=out_actnum, label=f'{out_dataset}', color='limegreen', fill=True, common_norm=True, alpha=.5, linewidth=0, legend=True, ax = ax)
    # plt.title("")
    plt.xlabel("act num")
    ax.legend(loc="upper right")
    # plt.xlim(0, m * 10)
    plt.savefig(save_pic_filename,dpi=600)
    
    plt.close()


def analysis_actvalue(args, model, in_loader, out_loader, out_dataset, max=50):
    model.eval()

    in_features = get_features(args, model, in_loader)
    out_features = get_features(args, model, out_loader)
    
    # 取出每行中第max大的元素
    in_actvalue = np.sort(in_features, axis=1)[:, -max]
    out_actvalue = np.sort(out_features, axis=1)[:, -max]

    print(in_actvalue.shape, out_actvalue.shape)

    path = f'analysis_feature/actvalue/{args.name}/{args.in_dataset}/{args.model}/maxnum{max}'
    if not os.path.isdir(path):
        os.makedirs(path)
    save_pic_filename=f'{path}/{args.in_dataset}_{out_dataset}.png'
    fig, ax = plt.subplots(figsize=(8, 4))
    sns.kdeplot(data=in_actvalue, label=f'{args.in_dataset}', color='crimson', fill=True, common_norm=True, alpha=.5, linewidth=0, legend=True, ax = ax)
    sns.kdeplot(data=out_actvalue, label=f'{out_dataset}', color='limegreen', fill=True, common_norm=True, alpha=.5, linewidth=0, legend=True, ax = ax)
    plt.title(f"max={max}")
    plt.xlabel("act value")
    ax.legend(loc="upper right")
    # plt.xlim(0, m * 10)
    plt.savefig(save_pic_filename,dpi=600)
    
    plt.close()




def main(args):
    in_dataset = args.in_dataset

    loader_in_dict = get_dataloader_in(args, split=('train','val'))
    train_dataloader, in_loader, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
    train_set, test_set = loader_in_dict.train_dataset, loader_in_dict.val_dataset
    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=load_ckpt)

    if args.out_dataset is not None:
        out_dataset = args.out_dataset
        loader_out_dict = get_dataloader_out(args, (None, out_dataset), split=('val'))
        out_loader = loader_out_dict.val_ood_loader

        in_set, out_set = loader_in_dict.val_dataset, loader_out_dict.val_dataset
        analysis_actnum(args, model, in_loader, out_loader, out_dataset)
        analysis_actvalue(args, model, in_loader, out_loader, out_dataset)
        # in_scores, out_scores = run_eval(model, in_loader, out_loader, args, num_classes=num_classes, out_dataset=out_dataset)
        # analysis_score(args, in_scores, out_scores, out_dataset)
    
    else:
        out_datasets = []
        if in_dataset == "imagenet":
            out_datasets = imagenet_out_datasets
        else:
            out_datasets = cifar_out_datasets
        for out_dataset in out_datasets:
            loader_out_dict = get_dataloader_out(args, (None, out_dataset), split=('val'))
            out_loader = loader_out_dict.val_ood_loader

            in_set, out_set = loader_in_dict.val_dataset, loader_out_dict.val_dataset
            # analysis_actnum(args, model, in_loader, out_loader, out_dataset)
            for i in [50, 100, 150, 200, 300, 400, 10]:     
                analysis_actvalue(args, model, in_loader, out_loader, out_dataset, i)
            # in_scores, out_scores = run_eval(model, in_loader, out_loader, args, num_classes=num_classes, out_dataset=out_dataset)
            # analysis_score(args, in_scores, out_scores, out_dataset)



if __name__ == "__main__":
    parser = get_argparser()

    main(parser.parse_args())
