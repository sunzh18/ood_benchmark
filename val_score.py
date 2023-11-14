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
from utils.data_loader import get_dataloader_in, get_dataloader_out, get_dataloader_noise, cifar_out_datasets, imagenet_out_datasets
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
        class_mean[i,:] = class_mean[i,:] * mask[i,:]
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

    # mask = np.where(class_mean>thresh,1,0)

    # print(mask)
    index = np.argwhere(mask == 1)
    mask = torch.tensor(mask)
    return mask, torch.tensor(class_mean)

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


def test_model_mask(model, data_loader, mask, p):
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


def test_mask(args):
    

    loader_in_dict = get_dataloader_in(args, split=('val'))
    in_loader, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    args.num_classes = num_classes
    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    model.eval()

    fc_w = extact_mean_std(args, model)
    mask, class_mean = get_class_mean2(args, fc_w)
    val_acc = test_model_mask(model, in_loader, mask, args.p)
    print(f'mask_val_acc = {val_acc}')
    
    val_acc = test_model(model, in_loader, mask)
    print(f'orign_val_acc = {val_acc}')
    return 


def run_eval(model, in_loader, out_loader, logger, args, num_classes, out_dataset, mask, class_mean, in_scores=None):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()

    if args.score == 'react':
        # args.threshold = 1.25
        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_react(in_loader, model, args.temperature_energy, args.threshold)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_react(out_loader, model, args.temperature_energy, args.threshold)

    elif args.score == 'ash-p':
        p = 0
        if args.p:
            p = args.p
        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_ashp(in_loader, model, args.temperature_energy, p)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_ashp(out_loader, model, args.temperature_energy, p)
    
    elif args.score == 'ash-s':
        p = 0
        if args.p:
            p = args.p
        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_ashs(in_loader, model, args.temperature_energy, p)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_ashs(out_loader, model, args.temperature_energy, p)
        analysis_score(args, in_scores, out_scores, out_dataset)


    elif args.score == 'myodin':
        p = 0
        if args.p:
            p = args.p
        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_myodin(in_loader, model, args.epsilon_odin, args.temperature_energy, mask, p, args.threshold, class_mean)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_myodin(out_loader, model, args.epsilon_odin, args.temperature_energy, mask, p, args.threshold, class_mean)
        analysis_score(args, in_scores, out_scores, out_dataset)

    elif args.score == 'mymsp':
        p = 0
        if args.p:
            p = args.p
        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_mymsp(in_loader, model, mask, p, args.threshold, class_mean)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_mymsp(out_loader, model, mask, p, args.threshold, class_mean)
        analysis_score(args, in_scores, out_scores, out_dataset)

    elif args.score == 'myLINE':
       
        args.threshold = 0.8  #0.8
        p = 0
        args.p = 0

        info = np.load(f"cache/{args.name}/{args.in_dataset}_{args.model}_meanshap_class.npy")
        model = get_model(args, num_classes, load_ckpt=True, info=info, LU=True)
        model.eval()
        fc_w = extact_mean_std(args, model)
        mask, class_mean = get_class_mean2(args, fc_w)
        # mask, class_mean = get_class_mean(args)
        class_mean = class_mean.cuda()
        # class_mean = class_mean.clip(max=args.threshold)
        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_myLINE(in_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_myLINE(out_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)
        analysis_score(args, in_scores, out_scores, out_dataset)

    elif args.score == 'simodin':
        p = 0
        if args.p:
            p = args.p
        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_simodin(in_loader, model, args.epsilon_odin, args.temperature_energy, mask, p, args.threshold, class_mean)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_simodin(out_loader, model, args.epsilon_odin, args.temperature_energy, mask, p, args.threshold, class_mean)
        analysis_score(args, in_scores, out_scores, out_dataset)

    elif args.score == 'my_score21':
        p = 0
        if args.p:
            p = args.p
        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_my21(in_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_my21(out_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)
        analysis_score(args, in_scores, out_scores, out_dataset)

    elif args.score == 'my_score22':
        p = 0
        if args.p:
            p = args.p
        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_my22(in_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_my22(out_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)
        analysis_score(args, in_scores, out_scores, out_dataset)
    
    elif args.score == 'my_score23':
        p = 0
        if args.p:
            p = args.p
        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_my23(in_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_my23(out_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)

    elif args.score == 'FM':
        p = 0
        if args.p:
            p = args.p
        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_my23_ablation(in_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_my23_ablation(out_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)

    elif args.score == 'cosine':
        p = 0
        if args.p:
            p = args.p
        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_cosine(in_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_cosine(out_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)
        analysis_score(args, in_scores, out_scores, out_dataset)

    elif args.score == 'ablation':
        p = 0
        if args.p:
            p = args.p

        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_ablation(in_loader, model, args.temperature_energy, mask, args.threshold, class_mean, args.cos)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_ablation(out_loader, model, args.temperature_energy, mask, args.threshold, class_mean, args.cos)
        analysis_score(args, in_scores, out_scores, out_dataset)
    
    elif args.score == 'reactmsp':
        p = 0
        if args.p:
            p = args.p

        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_reactmsp(in_loader, model, args.threshold)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_reactmsp(out_loader, model, args.threshold)
        analysis_score(args, in_scores, out_scores, out_dataset)
    
    elif args.score == 'reactodin':
        p = 0
        if args.p:
            p = args.p

        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_reactodin(in_loader, model, args.epsilon_odin, args.temperature_odin, args.threshold)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_reactodin(out_loader, model, args.epsilon_odin, args.temperature_odin, args.threshold)
        analysis_score(args, in_scores, out_scores, out_dataset)

    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    auroc, aupr_in, aupr_out, fpr95 = auroc*100, aupr_in*100, aupr_out*100, fpr95*100

    # if args.in_dataset == "imagenet":
    #     result_path = os.path.join(args.logdir, args.name, args.model, f"{args.in_dataset}_{args.score}.csv")
    #     fp = open(result_path,'a+')
    #     result = []

    #     result.append(f'p: {args.p}')
    #     result.append(out_dataset)
    #     result.append("{:.4f}".format(auroc))
    #     result.append("{:.4f}".format(aupr_in))
    #     result.append("{:.4f}".format(aupr_out))
    #     result.append("{:.4f}".format(fpr95))
    #     context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
    #     context.writerow(result)
    #     fp.close()

    logger.info('============Results for {}============'.format(args.score))
    logger.info('=======in dataset: {}; ood dataset: {}============'.format(args.in_dataset, out_dataset))
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))

    logger.flush()
    return auroc, aupr_in, aupr_out, fpr95, in_scores


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

    sns.kdeplot(data=in_examples, label='{}, mean{:.1f}, std{:.1f}'.format(args.in_dataset, in_examples.mean(0), in_examples.std(0)), color='crimson', fill=True, common_norm=True, alpha=.5, linewidth=0, legend=True, ax = ax)
    sns.kdeplot(data=out_examples, label='{}, mean{:.1f}, std{:.1f}'.format(out_dataset, out_examples.mean(0), out_examples.std(0)), color='limegreen', fill=True, common_norm=True, alpha=.5, linewidth=0, legend=True, ax = ax)
    plt.xlabel("score")
    ax.legend(loc="upper right")
    # plt.xlim(0, m * 10)
    plt.title('{}:p={}'.format(args.score ,args.p/100))
    plt.savefig(save_pic_filename,dpi=600)
    
    plt.close()


def analysis_sensitivity(args):
    args.logdir='sensitivity_result'
    logger = log.setup_logger(args)
    result_path = os.path.join('sensitivity_result', args.name, args.model, f"{args.in_dataset}_gaussian_{args.score}.csv")
    if not os.path.exists(result_path):
        fp = open(result_path,'a+')
        result = []
        result.append('AUROC')
        for p in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            result.append(f'p: {p}')
        result.append('FPR95')
        for p in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
            result.append(f'p: {p}')
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
    in_scores=None

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    model.eval()

    fc_w = extact_mean_std(args, model)

    loader_out_dict = get_dataloader_noise(args)
    out_loader = loader_out_dict.val_ood_loader
    # [0, 10, 20, 30, 40, 50, 60, 70, 80, 90, 99]

    In_Score = []

    file_folder = os.path.join('sensitivity_result', args.name, args.model)
    In_Score = np.load(f"{file_folder}/{args.in_dataset}_{args.score}.npy")

    num = 0
    AUroc, AUPR_in, AUPR_out, Fpr95 = [], [], [], []
    for p in [0, 10, 20, 30, 40, 50, 60, 70, 80, 90]:
        args.p = p
        mask, class_mean = get_class_mean4(args, fc_w)
        class_mean = class_mean.cuda()
        # in_scores=None
        in_scores = In_Score[num]
        num += 1


        in_set, out_set = loader_in_dict.val_dataset, loader_out_dict.val_dataset
        logger.info(f"Using an in-distribution set with {len(in_set)} images.")
        logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")


        start_time = time.time()
        auroc, aupr_in, aupr_out, fpr95, in_scores = run_eval(model, in_loader, out_loader, logger, args, num_classes=num_classes, out_dataset='gaussian', mask=mask, class_mean=class_mean, in_scores=in_scores)
        end_time = time.time()
        logger.info("Total running time: {}".format(end_time - start_time))
        AUroc.append(auroc)
        AUPR_in.append(aupr_in)
        AUPR_out.append(aupr_out)
        Fpr95.append(fpr95)
        # In_Score.append(in_scores)
        
    result_path = os.path.join(args.logdir, args.name, args.model, f"{args.in_dataset}_gaussian_{args.score}.csv")
    fp = open(result_path,'a+')
    result = []

    result.append('AUROC')
    for auc in AUroc:
        result.append("{:.4f}".format(auc))
    result.append('FPR95')
    for fpr95 in Fpr95:
        result.append("{:.4f}".format(fpr95))
    context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
    context.writerow(result)
    fp.close()

    
    # In_Score = np.array(In_Score)

    # np.save(f"{file_folder}/{args.in_dataset}_{args.score}.npy", In_Score)

def analysis_react_sensitivity(args):
    args.logdir='sensitivity_result'
    logger = log.setup_logger(args)
    result_path = os.path.join('sensitivity_result', args.name, args.model, f"react_{args.in_dataset}_{args.score}.csv")
    if not os.path.exists(result_path):
        fp = open(result_path,'a+')
        result = []
        result.append('model')
        result.append('out-dataset')
        result.append('FPR95')
        result.append('AUROC')
        context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
        context.writerow(result)
        fp.close()

    result_path = os.path.join(args.logdir, args.name, args.model, f"react_{args.in_dataset}_{args.score}.txt")
    if not os.path.exists(result_path):
        with open(result_path, 'a+', encoding='utf-8') as f:
            f.write('method  ')
            f.write('FPR95  ')
            f.write('AUROC\n')

    in_dataset = args.in_dataset

    in_save_dir = os.path.join(args.logdir, args.name, args.model)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    loader_in_dict = get_dataloader_in(args, split=('val'))
    in_loader, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    args.num_classes = num_classes
    in_scores=None

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    model.eval()
    
    fc_w = extact_mean_std(args, model)
    mask, class_mean = get_class_mean4(args, fc_w)
    class_mean = class_mean.cuda()
    for threshold in [0.1, 0.5, 0.8, 1.0, 1.5, 2.25, 1e5]:
        args.threshold = threshold
        
        in_scores=None
        if args.out_dataset is not None:
            out_dataset = args.out_dataset
            loader_out_dict = get_dataloader_out(args, (None, out_dataset), split=('val'))
            out_loader = loader_out_dict.val_ood_loader

            in_set, out_set = loader_in_dict.val_dataset, loader_out_dict.val_dataset


            start_time = time.time()
            run_eval(model, in_loader, out_loader, logger, args, num_classes=num_classes, out_dataset=out_dataset, mask=mask, class_mean=class_mean, in_scores=in_scores)
            print(in_scores)
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
                auroc, aupr_in, aupr_out, fpr95, in_scores = run_eval(model, in_loader, out_loader, logger, args, num_classes=num_classes, out_dataset=out_dataset, mask=mask, class_mean=class_mean, in_scores=in_scores)
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

            result_path = os.path.join(args.logdir, args.name, args.model, f"react_{args.in_dataset}_{args.score}.csv")
            fp = open(result_path,'a+')
            result = []

            result.append(f'p: {args.p}/threshold{args.threshold}')
            result.append('Average')
            result.append("{:.2f}".format(avg_fpr95))
            result.append("{:.2f}".format(avg_auroc))
            context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
            context.writerow(result)
            fp.close()

            result_path = os.path.join(args.logdir, args.name, args.model, f"react_{args.in_dataset}_{args.score}.txt")
            with open(result_path, 'a+', encoding='utf-8') as f:
                f.write("threshold={:.2f} & ".format(args.threshold))
                f.write("{:.2f} & ".format(avg_fpr95))
                f.write("{:.2f}\n".format(avg_auroc))


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


if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    
    if args.in_dataset == "CIFAR-10":
        if args.model == 'densenet':
            args.threshold = 1.6
            # args.threshold = 1.5      #react
            # args.threshold = 1.2
        elif args.model == 'resnet18':
            args.threshold = 0.8
        args.p_a = 90
        args.p_w = 90
        args.threshold = 1e5

    elif args.in_dataset == "CIFAR-100":
        if args.model == 'densenet':
            args.threshold = 1.6
            # args.threshold = 2.25       #react
            # args.threshold = 1.9
        elif args.model == 'resnet18':
            args.threshold = 0.8
        args.p_a = 10
        args.p_w = 90
        args.threshold = 1e5
            
    elif args.in_dataset == "imagenet":
        args.threshold = 0.8
        # args.threshold = 0.84
        if args.model == 'mobilenet':
            args.threshold = 0.2
        args.p_a = 10
        args.p_w = 10
    
    args.threshold = 1e5
    # analysis(args)
    # analysis_confidence(args)
    # analysis_feature(args)
    # analysis_cos(args)
    # analysis_react_sensitivity(args)
    analysis_sensitivity(args)
    # test_train(args)
    # test_mask(args)
