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
from utils.mahalanobis_lib import sample_estimator
from utils.data_loader import get_dataloader_in, get_dataloader_out, cifar_out_datasets, imagenet_out_datasets
from utils.model_loader import get_model

from utils.my_cal_score import *
from utils.cal_score import *
from argparser import *

from tqdm import tqdm
import torchvision
from torchvision import transforms
import seaborn as sns
import matplotlib as mpl
import matplotlib.pyplot as plt
import pandas as pd
from scipy.stats import norm
from scipy.stats import laplace
import torch.nn.functional as F
envpath = '/home/2022/zhuohao/miniconda3/envs/python39/lib/python3.9/site-packages/cv2/qt/plugins/platforms'
os.environ['QT_QPA_PLATFORM_PLUGIN_PATH'] = envpath

def get_class_feature(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    print()
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            # print(feature_list[i])
            temp = torch.zeros([int(feature_list[i])])
            # print(temp.shape)
            temp_list.append(temp)
        list_features.append(temp_list)

    for data, target in tqdm(train_loader):
        total += data.size(0)
        # print(total)
        # if total > 50000:
        #     break
        # data = data.cuda()
        data = Variable(data)
        data = data.cuda()
        output, out_features = model.feature_list(data)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            # print(out_features[i].shape)
            out_features[i] = torch.mean(out_features[i].data, 2)
            # out_features[i] = out_features[i].view(out_features[i].size(0), -1)
            # print(out_features[i].shape)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(data.size(0)):
            label = pred[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        # temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        temp_list = torch.Tensor(num_classes, int(num_feature)).cpu()

        for j in range(num_classes):
            # list_features[out_count][j] = list_features[out_count][j].clip(max=1.0)
            # print(list_features[out_count][j])
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    return sample_class_mean

def get_features(args, model, dataloader, mask=None):
    features = []
    for b, (x, y) in enumerate(dataloader):
        with torch.no_grad():
            x = x.cuda()            
            # print(x.size())
            feature = model.forward_features(x)
            # outputs = model.features(x)
            # feature = F.adaptive_avg_pool2d(outputs, 1)
            # feature = feature.view(feature.size(0), -1)
            # print(feature.size())
            # features.extend(feature.data.cpu().numpy())
            features.extend(feature.data.cpu().numpy())
            # x = feature[feature>=0]
            # print(x.size())

    features = np.array(features)
    # x = np.transpose(features)
    print(features.shape)

    return features

def extact_mean_std(args, model):
    for key, v in model.state_dict().items():
        # print(key)
        if key == 'classifier.1.weight':
            fc_w = v
            print(v.shape)
        if key == 'fc.weight':
            fc_w = v
            print(v.shape)
            # print(f'fc: {v}')
        
    return fc_w.cpu().numpy()

def get_class_mean_precision(args, model, num_classes, train_loader):
    if args.in_dataset == "CIFAR-10" or args.in_dataset == "CIFAR-100":
        temp_x = torch.rand(2,3,32,32).cuda()
    else:
        temp_x = torch.rand(2,3,224,224).cuda()
    temp_x = Variable(temp_x)
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)

    print(num_output)

    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        print(out.shape)
        print(out.size(1))
        feature_list[count] = out.size(1)
        count += 1
    class_mean = get_class_feature(model, num_classes, feature_list, train_loader)
    for mean in class_mean:
        print(mean.shape)
        print(mean)

    # print(class_mean.shape, precision.shape)
    return class_mean[0]
    

def get_class_mean(args):
    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    class_mean = np.load(f"{file_folder}/{args.model}_class_mean.npy")
    print(class_mean.shape)
    # print(class_mean)
    class_mean = np.squeeze(class_mean)
    print(class_mean.shape)
    # print(class_mean)
    np.save(f"{file_folder}/{args.model}_class_mean.npy", class_mean)

    p = 0
    if args.p:
        p = args.p
    thresh = np.percentile(class_mean, p, axis=1)
    # print(thresh.shape, thresh)
    mask = np.zeros_like(class_mean)
    print(mask.shape)
    for i in range(mask.shape[0]):
        mask[i] = np.where(class_mean[i] >= thresh[i],1,0)
        # class_mean[i,:] = class_mean[i,:] * mask[i,:]
    # mask = np.where(class_mean>thresh,1,0)

    # print(mask)
    index = np.argwhere(mask == 1)
    mask = torch.tensor(mask)
    return mask, torch.tensor(class_mean)
    # train_acc = test_model(model, train_dataloader, mask)
    # print(f'tran_acc = {train_acc}')

def get_class_mean2(args, fc_w):
    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    class_mean = np.load(f"{file_folder}/{args.model}_class_mean.npy")
    print(class_mean.shape, fc_w.shape)
    class_mean = np.squeeze(class_mean)

    # np.save(f"{file_folder}/{args.model}_class_mean.npy", class_mean)
    p = 0
    if args.p:
        p = args.p
    thresh = np.percentile(fc_w, p, axis=1)
    # print(thresh.shape, thresh)
    mask = np.zeros_like(fc_w)
    print(mask.shape)
    for i in range(mask.shape[0]):
        mask[i] = np.where(fc_w[i] >= thresh[i],1,0)
        class_mean[i,:] = class_mean[i,:] * mask[i,:]

    # mask = np.where(class_mean>thresh,1,0)

    # print(mask)
    index = np.argwhere(mask == 1)
    mask = torch.tensor(mask)
    return mask, torch.tensor(class_mean)

def get_class_mean3(args, fc_w):
    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    class_mean = np.load(f"{file_folder}/{args.model}_class_mean.npy")
    print(class_mean.shape, fc_w.shape)
    # class_mean = np.squeeze(class_mean)

    # np.save(f"{file_folder}/{args.model}_class_mean.npy", class_mean)
    p = 0
    if args.p:
        p = args.p
    
    fc_w = fc_w / fc_w.sum(axis=1)[:, None]
    fc_w = np.exp(fc_w)
    thresh = np.percentile(fc_w, p, axis=1)
    # print(thresh.shape, thresh)
    mask = np.zeros_like(fc_w)
    print(mask.shape)
    for i in range(mask.shape[0]):
        mask[i] = np.where(fc_w[i] >= thresh[i],1,0) * fc_w[i]
        # print(mask[i])
        class_mean[i,:] = class_mean[i,:] * mask[i,:]

    # mask = np.where(class_mean>thresh,1,0)

    # print(mask)
    index = np.argwhere(mask == 1)
    mask = torch.tensor(mask)
    return mask, torch.tensor(class_mean)

def get_class_mean4(args, fc_w):
    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    class_mean = np.load(f"{file_folder}/{args.model}_class_mean.npy")
    print(class_mean.shape, fc_w.shape)
    class_mean = np.squeeze(class_mean)

    # np.save(f"{file_folder}/{args.model}_class_mean.npy", class_mean)
    p = 0
    if args.p:
        p = args.p
    thresh = np.percentile(fc_w, p, axis=1)
    # print(thresh.shape, thresh)
    mask = np.zeros_like(fc_w)
    print(mask.shape)
    for i in range(mask.shape[0]):
        mask[i] = np.where(fc_w[i] >= thresh[i],1,0)
        # class_mean[i,:] = class_mean[i,:] * mask[i,:]
    # print(thresh)
    print(p, thresh.mean(0))
    # mask = np.where(class_mean>thresh,1,0)

    # print(mask)
    index = np.argwhere(mask == 1)
    mask = torch.tensor(mask)
    return thresh


def draw_feature(args, in_class_mean, out_class_mean, fc, save_dir, out_dataset, classid):
    # dim = 128
    # v1 = np.random.rand(dim)*3
    # v2 = np.random.rand(dim)*3
    # w = np.random.rand(dim)
    # w = np.array([0.1, 0.2, 0.3, 0.4, 0.5])  # replace with your actual data

    # sort indices by the weights
    sorted_indices = np.argsort(fc)
    sorted_indices = sorted_indices[::-1]  # reverse the order

    # sort v1 and v2 according to the sorted indices
    in_sorted = in_class_mean[sorted_indices]
    out_sorted = out_class_mean[sorted_indices]

    # plot
    plt.figure(figsize=(10, 6))
    plt.ylim(0, 6)
    # plt.plot(v1_sorted, color='yellow', label='v1')
    # plt.plot(v2_sorted, color='green', label='v2')

    plt.bar(np.arange(len(in_sorted)), in_sorted, color='red', label='v1')
    plt.bar(np.arange(len(out_sorted)), out_sorted, color='green', label='v2')

    plt.xlabel('Channel')
    plt.ylabel('Activation')

    ax = plt.gca().twinx()

    plt.plot(fc[sorted_indices], color='blue', label='w')
    plt.ylabel('Classifer weight')

    # 同时显示左右两个图例
    lines, labels = plt.gca().get_legend_handles_labels()
    lines2, labels2 = ax.get_legend_handles_labels()
    plt.legend(lines + lines2, labels + labels2)

    import matplotlib.patches as mpatches
    id_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='red', edgecolor='black')
    ood_patch = mpatches.Rectangle((0, 0), 1, 1, facecolor='green', edgecolor='black')

    # 添加右上角的label
    plt.legend([id_patch, ood_patch], [f'ID:{args.in_dataset}', f'OOD:{out_dataset}'], loc='upper right')

    plt.grid(True)
    save_dir = os.path.join(save_dir, args.in_dataset, out_dataset)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filename = os.path.join(save_dir, f'{classid}.pdf')
    plt.savefig(filename)

def draw_sensitivity(args, auc, fpr95, sota, p, save_dir):

    p = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]

    # 以上内容是模拟数据 请忽略
    # 画图部分开始
    plt.figure(figsize=(10, 6))

    plt.plot(p, auc, 'o-', color='red', label='ours')
    plt.plot(p, sota, '--', color='blue', label='LINe')

    plt.xlabel('Pruning percentile')
    plt.ylabel('AUROC')

    # plt.plot(fpr95, color='red', label='FPR95')
    # plt.ylabel('FPR95')

    # ax2.plot(p, fpr95, 's-', color='red', label='FPR95')
    # ax2.set_ylabel('FPR95')

    # plt.xlim(0, 1.0)
    plt.xticks(p, [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99])


    # plt.title('Training and Validation Accuracy over Epochs')
    plt.legend(loc='lower right')
    plt.grid(True)

    filename = os.path.join(save_dir, f'{args.in_dataset}_{args.score}.pdf')
    plt.savefig(filename)

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

def test_train(args):
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

    
    # checkpoint = torch.load(
    #         f'{args.model_path}/{args.name}/{args.in_dataset}/{args.model}_parameter.pth.tar')

    # model.load_state_dict(checkpoint['state_dict'])
    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    model.eval()
    mask = get_class_mean(args)

def sensitivity(args):
    args.logdir='plot/sensitivity'
    # logger = log.setup_logger(args)
    in_dataset = args.in_dataset
    args.score = 'my_score23'
    save_dir = os.path.join(args.logdir, args.name, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    filepath = os.path.join('sensitivity_result', args.name, args.model)
    filename = os.path.join(filepath, f"{args.in_dataset}_{args.score}.csv")
    data_array = []
    with open(filename) as csv_file:
        csv_reader = csv.reader(csv_file)
        for row in csv_reader:
            data_array.append(row)
        print(data_array)
    
    Auc = []
    Fpr95 = []
    for i in range(len(data_array)):
        if i != 0:
            data = data_array[i]
            auc = float(data[2])
            fpr95 = float(data[3])
            Auc.append(auc)
            Fpr95.append(fpr95)
    Auc = np.array(Auc)
    Fpr95 = np.array(Fpr95)

    if args.in_dataset == "CIFAR-10":
        sota = [96.57] * len(Auc)

    elif args.in_dataset == "CIFAR-100":
        sota = [88.71] * len(Auc)
            
    elif args.in_dataset == "imagenet":
        sota = [95.02] * len(Auc)

    p = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 0.99]
    draw_sensitivity(args, Auc, Fpr95, sota, p, save_dir)
    # args.logdir='sensitivity_result'
    # logger = log.setup_logger(args)
    # args.p = 80
    # loader_in_dict = get_dataloader_in(args, split=('val'))
    # in_loader, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    # args.num_classes = num_classes
    # in_scores=None

    # load_ckpt = False
    # if args.model_path != None:
    #     load_ckpt = True
    # model = get_model(args, num_classes, load_ckpt=load_ckpt)
    # model.eval()
    # fc_w = extact_mean_std(args, model)
    # for x in p:
    #     args.p = x * 100
    #     thresh = get_class_mean4(args, fc_w)
    #     logger.info("percentile: {}, thresh: {}".format(args.p, thresh.mean(0)))
    
    # mask, class_mean = get_class_mean4(args, fc_w)
    
    

def main(args):
    args.logdir='plot/feature_fc'
    # logger = log.setup_logger(args)
    in_dataset = args.in_dataset

    save_dir = os.path.join(args.logdir, args.name, args.model)
    if not os.path.exists(save_dir):
        os.makedirs(save_dir)
    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    if not os.path.isdir(file_folder):
        os.makedirs(file_folder)
    loader_in_dict = get_dataloader_in(args, split=('val'))
    in_loader, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
    in_class_mean = np.load(f"{file_folder}/{args.model}_class_mean.npy")
    fc_w = extact_mean_std(args, model)
    model.eval()
    # in_right, in_sum = analysis_act_num(model, in_loader, args, mask)
    if args.out_dataset is not None:
        out_dataset = args.out_dataset
        loader_out_dict = get_dataloader_out(args, (None, out_dataset), split=('val'))
        out_loader = loader_out_dict.val_ood_loader
    
    else:
        out_datasets = []
        AUroc, AUPR_in, AUPR_out, Fpr95 = [], [], [], []
        if in_dataset == "imagenet":
            out_datasets = imagenet_out_datasets
        else:
            out_datasets = cifar_out_datasets
        for out_dataset in out_datasets:
            if out_dataset == 'SVHN':
                continue
            loader_out_dict = get_dataloader_out(args, (None, out_dataset), split=('val'))
            out_loader = loader_out_dict.val_ood_loader

            filename = os.path.join(file_folder, f'{out_dataset}_{args.model}_class_mean.npy')
            if not os.path.exists(filename):
                out_class_mean = get_class_mean_precision(args, model, num_classes, out_loader)
                np.save(filename, out_class_mean)

            out_class_mean = np.load(filename)

            for i in tqdm(range(num_classes)):
                in_feature = in_class_mean[i]
                out_feature = out_class_mean[i]
                fc = fc_w[i]
                # np.random.rand(3)
                x = np.random.binomial(1, p=0.4, size=1)
                if x == 1:
                    if np.linalg.norm(x=out_feature, ord=1) != 0:
                        draw_feature(args, in_feature, out_feature, fc, save_dir, out_dataset, i)

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    
    if args.in_dataset == "CIFAR-10":
        if args.model == 'densenet':
            args.threshold = 1.6
        elif args.model == 'resnet18':
            args.threshold = 0.8
        args.p_a = 90
        args.p_w = 90

    elif args.in_dataset == "CIFAR-100":
        if args.model == 'densenet':
            args.threshold = 1.7
        elif args.model == 'resnet18':
            args.threshold = 0.8
        args.p_a = 10
        args.p_w = 90
            
    elif args.in_dataset == "imagenet":
        args.threshold = 0.8
        if args.model == 'mobilenet':
            args.threshold = 0.2
        args.p_a = 10
        args.p_w = 10
    
    # args.threshold = 1e5
    # analysis(args)
    # analysis_confidence(args)
    # analysis_feature(args)
    # analysis_cos(args)
    # main(args)
    sensitivity(args)
    # test_train(args)
    # test_mask(args)