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
import pandas as pd
from scipy.stats import norm
from scipy.stats import laplace
import torch.nn.functional as F



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
            scale = torch.tensor(right / count).float().cuda()
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
            s1 = cp.sum(dim=1)
            # s2 = feature_prun.sum(dim=1)

            confs.extend(s1.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my5(data_loader, model, temper, mask, p):
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
            s1 = cp.sum(dim=1)
            s2 = feature_prun.sum(dim=1)
            # rate = s1/s2
            rate = s2/(s2-s1)
            # rate = torch.exp(rate)
            confs.extend(rate.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my6(data_loader, model, temper, mask, p):
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
            # s1 = cp.sum(dim=1)
            # s2 = feature_prun.sum(dim=1)
            s1 = cp.norm(p=1, dim=1)
            s2 = feature_prun.norm(p=1, dim=1)
            # scale = s2/(s2-s1)
            scale = s1/s2
            cp = cp * torch.exp(scale[:, None])
            # scale = scale + 1
            # cp = cp * scale[:, None]
            # print(cp.type(), scale.type())
            logits = model.forward_head(cp)
            conf = temper * (torch.logsumexp(logits / (temper), dim=1))
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my7(data_loader, model, temper, mask, p):
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
            s1 = cp.sum(dim=1)
            s2 = feature_prun.sum(dim=1)
            # scale = s1/s2
            scale = s2/(s2-s1)
            feature_prun = feature_prun * scale[:, None]
            # print(cp.type(), scale.type())
            logits = model.forward_head(feature_prun)
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my8(data_loader, model, temper, mask, p):

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
            s1 = feature.sum(dim=1)
            s2 = feature_prun.sum(dim=1)
            scale = s1/s2
            confs.extend(scale.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my9(data_loader, model, temper, mask, p):
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
            s1 = cp.sum(dim=1)
            s2 = feature_prun.sum(dim=1)
            scale = s1/s2
            # scale = s2/(s2-s1)
            # cp = cp * scale[:, None]
            cp = cp * torch.exp(scale[:, None])
            # print(cp.type(), scale.type())
            logits = model.forward_head(cp)
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            scale2 = s1/s2
            conf = conf * scale
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my10(data_loader, model, temper, mask, p):
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
                feature_prun[counter_cp,:] = feature_prun[counter_cp,:] * mask[idx,:].cuda()
                counter_cp = counter_cp + 1

            # print(feature_prun, cp, feature)
            s1 = cp.sum(dim=1)
            s2 = feature_prun.sum(dim=1)
            # scale = s1/s2
            scale = s1/s2
            feature_prun = feature_prun * scale[:, None]
            # print(cp.type(), scale.type())
            logits = model.forward_head(feature_prun)
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            # scale2 = s1/s2
            # conf = conf * scale
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my11(data_loader, model, temper, mask, p, threshold):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_threshold_features(x, threshold)
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
            scale = s1/s2
            # scale = s2/(s2-s1)
            # cp = cp * scale[:, None]
            # cp = cp * torch.exp(scale[:, None])
            # print(cp.type(), scale.type())
            logits = model.forward_head(cp)
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            scale2 = s1/s2
            # conf = conf * scale
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my12(data_loader, model, temper, mask, p, threshold, class_mean):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_threshold_features(x, threshold)
            np_feature = feature.cpu().numpy()
            thresh = np.percentile(np_feature, p, axis=1)
            counter_cp = 0
            cp = torch.zeros(feature.shape).cuda()
            class_mask = torch.zeros(feature.shape).cuda()
            feature_prun = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                cp[counter_cp,:] = feature[counter_cp,:] * mask[idx,:].cuda() 
                # feature_prun[counter_cp] = torch.tensor(np.where(np_feature[counter_cp] >= thresh[counter_cp],np_feature[counter_cp],0)).cuda() 
                class_mask[counter_cp,:] = class_mean[idx,:].cuda() 
                counter_cp = counter_cp + 1

            cos_sim = F.cosine_similarity(class_mask, cp, dim=1)
            # conf = conf * scale
            # cos_sim = torch.exp(cos_sim)
            confs.extend(cos_sim.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my13(data_loader, model, temper, mask, p, threshold, class_mean):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_threshold_features(x, threshold)
            np_feature = feature.cpu().numpy()
            thresh = np.percentile(np_feature, p, axis=1)
            counter_cp = 0
            cp = torch.zeros(feature.shape).cuda()
            class_mask = torch.zeros(feature.shape).cuda()
            feature_prun = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                cp[counter_cp,:] = feature[counter_cp,:] * mask[idx,:].cuda() 
                # feature_prun[counter_cp] = torch.tensor(np.where(np_feature[counter_cp] >= thresh[counter_cp],np_feature[counter_cp],0)).cuda() 
                class_mask[counter_cp,:] = class_mean[idx,:].cuda() 
                counter_cp = counter_cp + 1

            cos_sim = F.cosine_similarity(class_mask, cp, dim=1)

            # conf = conf * scale
            cp = cp * torch.exp(cos_sim[:, None])
            # cp = cp * scale[:, None]
            # print(cp.type(), scale.type())
            logits = model.forward_head(cp)
            conf = temper * (torch.logsumexp(logits / (temper), dim=1))
            conf = conf * cos_sim
            confs.extend(conf.data.cpu().numpy())
           
    return np.array(confs)

def iterate_data_my14(data_loader, model, temper, mask, p, threshold):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            # feature = model.forward_features(x)
            feature = model.forward_threshold_features(x, threshold)
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
            scale = s1/s2
            # scale = s2/(s2-s1)
            # cp = cp * scale[:, None]
            # cp = cp * scale[:, None]
            # print(cp.type(), scale.type())
            logits = model.forward_head(cp)
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            scale2 = s1/s2
            # conf = conf * scale
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my15(data_loader, model, temper, mask, p, threshold):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            # feature = model.forward_features(x)
            feature = model.forward_threshold_features(x, threshold)
            logits = model.forward_head(feature)
            np_feature = feature.cpu().numpy()
            thresh = np.percentile(np_feature, p, axis=1)
            counter_cp = 0
            cp = torch.zeros(feature.shape).cuda()
            feature_prun = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                cp[counter_cp,:] = feature[counter_cp,:] * mask[idx,:].cuda() 
                feature_prun[counter_cp] = torch.tensor(np.where(np_feature[counter_cp] >= thresh[counter_cp],np_feature[counter_cp],0)).cuda()
                counter_cp = counter_cp + 1

            logits2 = model.forward_head(cp)
            counter_cp = 0

            for idx in pred_y:
                logits[counter_cp, idx] = logits2[counter_cp, idx]
                counter_cp = counter_cp + 1
                
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            # conf = conf * scale
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my16(data_loader, model, temper, mask, p, threshold):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()
            # feature = model.forward_features(x)
            feature = model.forward_threshold_features(x, threshold)
            logits = model.forward_head_mask(feature, mask.cuda())
            conf = temper * (torch.logsumexp(logits / temper, dim=1))

            # conf = conf * scale
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my17(data_loader, model, temper, mask, p, threshold, class_mean):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_threshold_features(x, threshold)
            np_feature = feature.cpu().numpy()
            thresh = np.percentile(np_feature, p, axis=1)
            counter_cp = 0
            cp = torch.zeros(feature.shape).cuda()
            class_mask = torch.zeros(feature.shape).cuda()
            feature_prun = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                cp[counter_cp,:] = feature[counter_cp,:] * mask[idx,:].cuda() 
                feature_prun[counter_cp] = torch.tensor(np.where(np_feature[counter_cp] >= thresh[counter_cp],np_feature[counter_cp],0)).cuda()
                class_mask[counter_cp,:] = class_mean[idx,:].cuda() 
                counter_cp = counter_cp + 1

            # print(feature_prun, cp, feature)
            L = mask.sum(dim=1)[0]
            cos_sim = F.cosine_similarity(class_mask, cp, dim=1)
            s1 = cp.sum(dim=1)
            s2 = feature_prun.sum(dim=1)
            s3 = class_mask.sum(dim=1)
            scale = s1/s2
            scale2 = s1/s3
            act_rate = s1/L
            # scale = s2/(s2-s1)
            # cp = cp * scale[:, None]
            # cp = cp * torch.exp(scale[:, None])
            # print(cp.type(), scale.type())
            logits = model.forward_head(cp)
            l2_dis = 1 / torch.pairwise_distance(class_mask, cp)
            counter_cp = 0

            for idx in pred_y:
                # logits[counter_cp, idx] = logits[counter_cp, idx] * torch.exp(scale[counter_cp])
                logits[counter_cp, idx] = logits[counter_cp, idx] * torch.exp(cos_sim[counter_cp])
                # logits[counter_cp, idx] = logits[counter_cp, idx] * torch.exp(cos_sim[counter_cp]/10)
                # logits[counter_cp, idx] = logits[counter_cp, idx] * torch.exp(l2_dis[counter_cp])
                # logits[counter_cp, idx] = logits[counter_cp, idx] * torch.exp(act_rate[counter_cp])
                # logits[counter_cp, idx] = logits[counter_cp, idx] * torch.exp(scale2[counter_cp])
                counter_cp = counter_cp + 1

            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            conf = conf * cos_sim
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my18(data_loader, model, temper, mask, p, threshold, class_mean):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_threshold_features(x, threshold)
            np_feature = feature.cpu().numpy()
            thresh = np.percentile(np_feature, p, axis=1)
            counter_cp = 0
            cp = torch.zeros(feature.shape).cuda()
            feature_prun = torch.zeros(feature.shape).cuda()
            class_mask = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                cp[counter_cp,:] = feature[counter_cp,:] * mask[idx,:].cuda() 
                feature_prun[counter_cp] = torch.tensor(np.where(np_feature[counter_cp] >= thresh[counter_cp],np_feature[counter_cp],0)).cuda()
                class_mask[counter_cp,:] = class_mean[idx,:].cuda() 
                counter_cp = counter_cp + 1

            # print(feature_prun, cp, feature)
            L = mask.sum(dim=1)[0]
            s1 = cp.sum(dim=1)
            s2 = feature_prun.sum(dim=1)
            rate = s1/s2
            l = cp.shape[1]
            l2_dis = 1 / torch.pairwise_distance(class_mask, cp)
            

            confs.extend(l2_dis.cpu().numpy())

    return np.array(confs)

def iterate_data_my19(data_loader, model, temper, mask, p, threshold, class_mean):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            feature = model.forward_threshold_features(x, threshold)
            np_feature = feature.cpu().numpy()
            thresh = np.percentile(np_feature, p, axis=1)
            counter_cp = 0
            cp = torch.zeros(feature.shape).cuda()
            feature_prun = torch.zeros(feature.shape).cuda()
            class_mask = torch.zeros(feature.shape).cuda()
            for idx in pred_y:
                cp[counter_cp,:] = feature[counter_cp,:] * mask[idx,:].cuda() 
                counter_cp = counter_cp + 1

            logits = cp @ class_mean.t() / (cp.norm(2,1) * class_mean.norm(2,1))
            
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            confs.extend(conf.cpu().numpy())

    return np.array(confs)

