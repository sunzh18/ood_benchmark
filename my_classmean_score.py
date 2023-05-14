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

from utils.cal_score import *
from argparser import *

from tqdm import tqdm
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
            feature = model.forward_features(x)
            # outputs = model.features(x)
            # feature = F.adaptive_avg_pool2d(outputs, 1)
            # feature = feature.view(feature.size(0), -1)
            # print(feature.size())
            # features.extend(feature.data.cpu().numpy())
            features.extend(feature)
            # x = feature[feature>=0]
            # print(x.size())

    # features = np.array(features)
    # # x = np.transpose(features)
    # print(features.shape)

    return features


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
    # mask = np.where(class_mean>thresh,1,0)
    
    # print(mask)
    index = np.argwhere(mask == 1)
    mask = torch.tensor(mask)
    return mask
    # train_acc = test_model(model, train_dataloader, mask)
    # print(f'tran_acc = {train_acc}')
    

def test_model(model, data_loader, mask):
    num=0
    accu=0
    
    print("test: ")
    with torch.no_grad():
        for data,target in tqdm(data_loader):
            data, target = data.cuda(), target.cuda()
    #         print(data.shape, data)
            output = model(data)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()
            y_label = target.cpu().numpy()
            accu += (pred_y == y_label).sum()
    #         print(pred_y, y_label)
    #         print(accu1 / num1, accu2 / num2)

            num += len(y_label)
    accu /= num
    # print("loss:",Loss,"accuracy:",accu)
    # print("loss:",Loss,"accuracy:",accu)
    return accu


def test_model_mask(model, data_loader, mask):
    num=0
    accu=0
    
    print("test: ")
    with torch.no_grad():
        for data,target in tqdm(data_loader):
            data, target = data.cuda(), target.cuda()
    #         print(data.shape, data)
            output = model(data)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_features(data)
            counter_cp = 0
            cp = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                cp[counter_cp,:] = feature[counter_cp,:] * mask[idx,:].cuda()     
                counter_cp = counter_cp + 1

            logits = model.forward_head(cp)
            pred_y = torch.max(logits, 1)[1].cpu().numpy()
            y_label = target.cpu().numpy()
            accu += (pred_y == y_label).sum()
    #         print(pred_y, y_label)
    #         print(accu1 / num1, accu2 / num2)

            num += len(y_label)
    accu /= num
    # print("loss:",Loss,"accuracy:",accu)
    # print("loss:",Loss,"accuracy:",accu)
    return accu


def iterate_data_my(data_loader, model, temper, mask):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_features(x)
            counter_cp = 0
            cp = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                cp[counter_cp,:] = feature[counter_cp,:] * mask[idx,:].cuda()     
                counter_cp = counter_cp + 1

            logits = model.forward_head(cp)
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

def iterate_data_my2(data_loader, model, temper, mask, p):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_features(x)
            np_feature = feature.cpu().numpy()
            thresh = np.percentile(np_feature, p, axis=1)
            counter_cp = 0
            cp = torch.zeros(feature.shape).cuda()
            feature_prun = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                cp[counter_cp,:] = feature[counter_cp,:] * mask[idx,:].cuda() 
                feature_prun[counter_cp] = torch.tensor(np.where(np_feature[counter_cp] >= thresh[counter_cp],np_feature[counter_cp],0)).cuda()
                
                counter_cp = counter_cp + 1

            # print(feature_prun, cp, feature)
            m1 = np.array(feature_prun.cpu().numpy() > 0)
            m2 = np.array(feature_prun.cpu().numpy() == cp.cpu().numpy())
            right = (m1 * m2).sum(axis=1)
            count = np.sum(cp.cpu().numpy()>0,axis=1)
            Right.extend(right)
            Sum.extend(count)

    return np.array(Right)

def iterate_data_my3(data_loader, model, temper, mask, p):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_features(x)
            np_feature = feature.cpu().numpy()
            thresh = np.percentile(np_feature, p, axis=1)
            counter_cp = 0
            cp = torch.zeros(feature.shape).cuda()
            feature_prun = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                cp[counter_cp,:] = feature[counter_cp,:] * mask[idx,:].cuda() 
                feature_prun[counter_cp] = torch.tensor(np.where(np_feature[counter_cp] >= thresh[counter_cp],np_feature[counter_cp],0)).cuda()
                
                counter_cp = counter_cp + 1

            # print(feature_prun, cp, feature)
            m1 = np.array(feature_prun.cpu().numpy() > 0)
            m2 = np.array(feature_prun.cpu().numpy() == cp.cpu().numpy())
            right = (m1 * m2).sum(axis=1)
            count = np.sum(cp.cpu().numpy()>0,axis=1)
            scale = torch.tensor(right / count + 1.0).float().cuda()
            cp = cp * torch.exp(scale[:, None])
            # print(cp.type(), scale.type())
            logits = model.forward_head(cp)
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my4(data_loader, model, temper, mask, p):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_features(x)
            np_feature = feature.cpu().numpy()
            thresh = np.percentile(np_feature, p, axis=1)
            counter_cp = 0
            cp = torch.zeros(feature.shape).cuda()
            feature_prun = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                cp[counter_cp,:] = feature[counter_cp,:] * mask[idx,:].cuda() 
                feature_prun[counter_cp] = torch.tensor(np.where(np_feature[counter_cp] >= thresh[counter_cp],np_feature[counter_cp],0)).cuda()
                
                counter_cp = counter_cp + 1

            # print(feature_prun, cp, feature)
            m1 = np.array(feature_prun.cpu().numpy() > 0)
            m2 = np.array(feature_prun.cpu().numpy() == cp.cpu().numpy())
            right = (m1 * m2).sum(axis=1)
            count = np.sum(cp.cpu().numpy()>0,axis=1)
            scale = torch.tensor(right / count + 1.0).float().cuda()
            cp = cp * torch.exp(scale[:, None])
            # print(cp.type(), scale.type())
            logits = model.forward_head(cp)
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_ashp(data_loader, model, temper, p):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_features(x)
            np_feature = feature.cpu().numpy()
            thresh = np.percentile(np_feature, p, axis=1)
            counter_cp = 0
            feature_prun = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                feature_prun[counter_cp] = torch.tensor(np.where(np_feature[counter_cp] >= thresh[counter_cp],np_feature[counter_cp],0)).cuda()
                counter_cp = counter_cp + 1

            # print(feature_prun, cp, feature)
            
            # scale = torch.tensor(right / count).float().cuda()
            # cp = cp * torch.exp(scale[:, None])
            # print(cp.type(), scale.type())
            logits = model.forward_head(feature_prun)
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_ashs(data_loader, model, temper, p):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)
            
            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_features(x)
            s1 = feature.sum(dim=1)
            np_feature = feature.cpu().numpy()
            thresh = np.percentile(np_feature, p, axis=1)
            counter_cp = 0
            feature_prun = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                feature_prun[counter_cp] = torch.tensor(np.where(np_feature[counter_cp] >= thresh[counter_cp],np_feature[counter_cp],0)).cuda()
                counter_cp = counter_cp + 1

            # print(feature_prun, cp, feature)
            
            #
            # print(cp.type(), scale.type())
            s2 = feature_prun.sum(dim=1)
            scale = s1 / s2
            feature_prun = feature_prun * torch.exp(scale[:, None])
            logits = model.forward_head(feature_prun)
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def test_mask(args):
    args.num_classes = num_classes

    loader_in_dict = get_dataloader_in(args, split=('val'))
    in_loader, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    model.eval()

    mask = get_class_mean(args)
    # val_acc = test_model_mask(model, in_loader, mask)
    # print(f'mask_val_acc = {val_acc}')
    
    # val_acc = test_model(model, in_loader, mask)
    # print(f'orign_val_acc = {val_acc}')
    return 


def run_eval(model, in_loader, out_loader, logger, args, num_classes, out_dataset, mask):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()

    if args.score == 'react':
        # args.threshold = 1.25
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_react(in_loader, model, args.temperature_energy, args.threshold)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_react(out_loader, model, args.temperature_energy, args.threshold)

    elif args.score == 'ash-p':
        p = 0
        if args.p:
            p = args.p
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_ashp(in_loader, model, args.temperature_energy, p)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_ashp(out_loader, model, args.temperature_energy, p)
    
    elif args.score == 'ash-s':
        p = 0
        if args.p:
            p = args.p
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_ashs(in_loader, model, args.temperature_energy, p)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_ashs(out_loader, model, args.temperature_energy, p)
        analysis_score(args, in_scores, out_scores, out_dataset)

    elif args.score == 'my_score':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_my(in_loader, model, args.temperature_energy, mask)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_my(out_loader, model, args.temperature_energy, mask)
    elif args.score == 'my_score2':
        p = 0
        if args.p:
            p = args.p
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_my2(in_loader, model, args.temperature_energy, mask, p)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_my2(out_loader, model, args.temperature_energy, mask, p)
    elif args.score == 'my_score3':
        p = 0
        if args.p:
            p = args.p
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_my3(in_loader, model, args.temperature_energy, mask, p)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_my3(out_loader, model, args.temperature_energy, mask, p)

        analysis_score(args, in_scores, out_scores, out_dataset)
    
    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    
    # result_path = os.path.join(args.logdir, args.name, args.model, f"{args.in_dataset}_{args.score}.csv")
    # fp = open(result_path,'a+')
    # result = []

    # result.append(f'p: {args.p}')
    # result.append(out_dataset)
    # result.append("{:.4f}".format(auroc))
    # result.append("{:.4f}".format(aupr_in))
    # result.append("{:.4f}".format(aupr_out))
    # result.append("{:.4f}".format(fpr95))
    # context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
    # context.writerow(result)
    # fp.close()

    logger.info('============Results for {}============'.format(args.score))
    logger.info('=======in dataset: {}; ood dataset: {}============'.format(args.in_dataset, out_dataset))
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))

    logger.flush()
    return auroc, aupr_in, aupr_out, fpr95


def analysis_act_num(model, data_loader, args, mask):
    p = 0
    if args.p:
        p = args.p
    Right = []
    Sum = []
    with torch.no_grad():
        for data,target in tqdm(data_loader):
            data, target = data.cuda(), target.cuda()
    #         print(data.shape, data)
            output = model(data)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_features(data)
            np_feature = feature.cpu().numpy()
            thresh = np.percentile(np_feature, p, axis=1)
            counter_cp = 0
            cp = torch.zeros(feature.shape).cuda()
            feature_prun = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                cp[counter_cp,:] = feature[counter_cp,:] * mask[idx,:].cuda() 
                feature_prun[counter_cp] = torch.tensor(np.where(np_feature[counter_cp] >= thresh[counter_cp],np_feature[counter_cp],0)).cuda()
                
                counter_cp = counter_cp + 1

            # print(feature_prun, cp, feature)
            # m1 = np.array(feature_prun.cpu().numpy() > 0)
            m1 = np.array(cp.cpu().numpy() > 0)
            m2 = np.array(feature_prun.cpu().numpy() == cp.cpu().numpy())
            right = (m1 * m2).sum(axis=1)
            count = np.sum(cp.cpu().numpy()>0,axis=1)
            Right.extend(right)
            Sum.extend(count)
    #         print(f'right:{right}, count:{count}')
    # #         logits = model.forward_head(cp)
    #         pred_y = torch.max(logits, 1)[1].cpu().numpy()
    #         y_label = target.cpu().numpy()
    #         accu += (pred_y == y_label).sum()
    # #         print(pred_y, y_label)
    # #         print(accu1 / num1, accu2 / num2)

    #         num += len(y_label)
    Right = np.array(Right)
    Sum = np.array(Sum)
    print(Right.shape, Sum.shape)
    return Right, Sum


def analysis_act_value(model, data_loader, args, mask):
    p = 0
    if args.p:
        p = args.p
    class_prun = []
    max_prun = []
    with torch.no_grad():
        for data,target in tqdm(data_loader):
            data, target = data.cuda(), target.cuda()
    #         print(data.shape, data)
            output = model(data)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_features(data)
            np_feature = feature.cpu().numpy()
            thresh = np.percentile(np_feature, p, axis=1)
            counter_cp = 0
            cp = torch.zeros(feature.shape).cuda()
            feature_prun = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                cp[counter_cp,:] = feature[counter_cp,:] * mask[idx,:].cuda() 
                feature_prun[counter_cp] = torch.tensor(np.where(np_feature[counter_cp] >= thresh[counter_cp],np_feature[counter_cp],0)).cuda()
                
                counter_cp = counter_cp + 1

            # print(feature_prun, cp, feature)
            s1 = cp.sum(dim=1)
            s2 = feature_prun.sum(dim=1)
            class_prun.extend(s1.cpu().numpy())
            max_prun.extend(s2.cpu().numpy())
    #         print(f'right:{right}, count:{count}')
    # #         logits = model.forward_head(cp)
    #         pred_y = torch.max(logits, 1)[1].cpu().numpy()
    #         y_label = target.cpu().numpy()
    #         accu += (pred_y == y_label).sum()
    # #         print(pred_y, y_label)
    # #         print(accu1 / num1, accu2 / num2)

    #         num += len(y_label)
    class_prun = np.array(class_prun)
    max_prun = np.array(max_prun)
    print(class_prun.shape, max_prun.shape)
    return class_prun, max_prun

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

def analysis(args):
    # args.logdir='analysis_feature/prun_num'
    args.logdir='analysis_feature/prun_value'
    logger = log.setup_logger(args)
    in_dataset = args.in_dataset

    in_save_dir = os.path.join(args.logdir, args.name, args.model)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    loader_in_dict = get_dataloader_in(args, split=('val'))
    in_loader, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    mask = get_class_mean(args)
    model.eval()
    # in_right, in_sum = analysis_act_num(model, in_loader, args, mask)
    in_class_prun, in_max_prun = analysis_act_value(model, in_loader, args, mask)

    logger.info(f'top_p: {args.p}')
    # print(f'in_data: {args.in_dataset}, mean_in_num: {in_right.mean(0)}, sum: {in_sum.mean(0)}')
    # logger.info(f'in_data: {args.in_dataset}, mean_in_num: {in_right.mean(0)}, sum: {in_sum.mean(0)}')

    print(f'in_data: {args.in_dataset}, mean_in_class_prun: {in_class_prun.mean(0)}, max_prun: {in_max_prun.mean(0)}')
    logger.info(f'in_data: {args.in_dataset}, mean_in_class_prun: {in_class_prun.mean(0)}, max_prun: {in_max_prun.mean(0)}')

    if args.out_dataset is not None:
        out_dataset = args.out_dataset
        loader_out_dict = get_dataloader_out(args, (None, out_dataset), split=('val'))
        out_loader = loader_out_dict.val_ood_loader

        in_set, out_set = loader_in_dict.val_dataset, loader_out_dict.val_dataset
        start_time = time.time()
        out_right, out_sum = analysis_act_num(model, out_loader, args, mask)
        in_class_prun, in_max_prun = analysis_act_value(model, out_loader, args, mask)
        end_time = time.time()

    
    else:
        out_datasets = []
        AUroc, AUPR_in, AUPR_out, Fpr95 = [], [], [], []
        if in_dataset == "imagenet":
            out_datasets = imagenet_out_datasets
        else:
            out_datasets = cifar_out_datasets
        for out_dataset in out_datasets:
            loader_out_dict = get_dataloader_out(args, (None, out_dataset), split=('val'))
            out_loader = loader_out_dict.val_ood_loader
            
            in_set, out_set = loader_in_dict.val_dataset, loader_out_dict.val_dataset
            start_time = time.time()
            # out_right, out_sum = analysis_act_num(model, out_loader, args, mask)
            out_class_prun, out_max_prun = analysis_act_value(model, out_loader, args, mask)
            # analysis_score(args, in_right, out_right, out_dataset)
            # print(f'out_data: {out_dataset}, mean_out_num: {out_right.mean(0)}, sum: {out_sum.mean(0)}')
            # logger.info(f'out_data: {out_dataset}, mean_out_num: {out_right.mean(0)}, sum: {out_sum.mean(0)}')

            print(f'in_data: {args.in_dataset}, mean_out_class_prun: {out_class_prun.mean(0)}, max_prun: {out_max_prun.mean(0)}')
            logger.info(f'in_data: {args.in_dataset}, mean_out_class_prun: {out_class_prun.mean(0)}, max_prun: {out_max_prun.mean(0)}')
            end_time = time.time()


def main(args):
    logger = log.setup_logger(args)

    result_path = os.path.join(args.logdir, args.name, args.model, f"{args.in_dataset}_{args.score}.csv")
    if not os.path.exists(result_path):
        fp = open(result_path,'a+')
        result = []
        result.append('model')
        result.append('out-dataset')
        result.append('AUROC')
        result.append('AUPR (In)')
        result.append('AUPR (Out)')
        result.append('FPR95')
        context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
        context.writerow(result)
        fp.close()


    in_dataset = args.in_dataset

    in_save_dir = os.path.join(args.logdir, args.name, args.model)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    loader_in_dict = get_dataloader_in(args, split=('val'))
    in_loader, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    mask = get_class_mean(args)
    model.eval()

    if args.out_dataset is not None:
        out_dataset = args.out_dataset
        loader_out_dict = get_dataloader_out(args, (None, out_dataset), split=('val'))
        out_loader = loader_out_dict.val_ood_loader

        in_set, out_set = loader_in_dict.val_dataset, loader_out_dict.val_dataset
        logger.info(f"Using an in-distribution set with {len(in_set)} images.")
        logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")


        start_time = time.time()
        run_eval(model, in_loader, out_loader, logger, args, num_classes=num_classes, out_dataset=out_dataset, mask=mask)
        end_time = time.time()

        logger.info("Total running time: {}".format(end_time - start_time))
    
    else:
        out_datasets = []
        AUroc, AUPR_in, AUPR_out, Fpr95 = [], [], [], []
        if in_dataset == "imagenet":
            out_datasets = imagenet_out_datasets
        else:
            out_datasets = cifar_out_datasets
        for out_dataset in out_datasets:
            loader_out_dict = get_dataloader_out(args, (None, out_dataset), split=('val'))
            out_loader = loader_out_dict.val_ood_loader

            in_set, out_set = loader_in_dict.val_dataset, loader_out_dict.val_dataset
            logger.info(f"Using an in-distribution set with {len(in_set)} images.")
            logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

            start_time = time.time()
            auroc, aupr_in, aupr_out, fpr95 = run_eval(model, in_loader, out_loader, logger, args, num_classes=num_classes, out_dataset=out_dataset, mask=mask)
            end_time = time.time()

            logger.info("Total running time: {}".format(end_time - start_time))

            AUroc.append(auroc)
            AUPR_in.append(aupr_in)
            AUPR_out.append(aupr_out)
            Fpr95.append(fpr95)
        avg_auroc = sum(AUroc) / len(AUroc)
        avg_aupr_in = sum(AUPR_in) / len(AUPR_in)
        avg_aupr_out = sum(AUPR_out) / len(AUPR_out)
        avg_fpr95 = sum(Fpr95) / len(Fpr95)

        result_path = os.path.join(args.logdir, args.name, args.model, f"{args.in_dataset}_{args.score}.csv")
        fp = open(result_path,'a+')
        result = []

        result.append(f'p: {args.p}')
        result.append('Average')
        result.append("{:.4f}".format(avg_auroc))
        result.append("{:.4f}".format(avg_aupr_in))
        result.append("{:.4f}".format(avg_aupr_out))
        result.append("{:.4f}".format(avg_fpr95))
        context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
        context.writerow(result)
        fp.close()

        logger.info('============Results for {}============'.format(args.score))
        logger.info('=======in dataset: {}; ood dataset: Average============'.format(args.in_dataset))
        logger.info('Average AUROC: {}'.format(avg_auroc))
        logger.info('Average AUPR (In): {}'.format(avg_aupr_in))
        logger.info('Average AUPR (Out): {}'.format(avg_aupr_out))
        logger.info('Average FPR95: {}'.format(avg_fpr95))

    


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
    train_acc = test_model_mask(model, train_dataloader, mask)
    print(f'mask_train_acc = {train_acc}')
    model.eval()
    in_right, in_sum = analysis_act_num(model, train_dataloader, args, mask)

    print(f'train_data: {args.in_dataset}, mean_in_num: {in_right.mean(0)}, sum: {in_sum.mean(0)}')




if __name__ == "__main__":
    parser = get_argparser()

    analysis(parser.parse_args())
    # main(parser.parse_args())
    # test_train(parser.parse_args())
    