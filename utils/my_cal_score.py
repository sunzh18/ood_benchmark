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
# from compute_gradcam import *

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
            rate = cos_sim
            # conf = conf * scale
            # cos_sim = torch.exp(cos_sim)
            confs.extend(rate.data.cpu().numpy())

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
            # for idx in pred_y:
            #     cp[counter_cp,:] = feature[counter_cp,:] * mask[idx,:].cuda() 
            #     feature_prun[counter_cp] = torch.tensor(np.where(np_feature[counter_cp] >= thresh[counter_cp],np_feature[counter_cp],0)).cuda()
            #     # class_mask[counter_cp,:] = class_mean[idx,:].cuda() 
            #     counter_cp = counter_cp + 1

            # print(feature_prun, cp, feature)
            cp = feature * mask[pred_y,:].cuda()
            class_mask = class_mean[pred_y,:].cuda() 
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
            # logits = logits * torch.exp(cos_sim[:, None])
            l2_dis = 1 / torch.pairwise_distance(class_mask, cp)
            counter_cp = 0
            
            # for idx in pred_y:
            #     # logits[counter_cp, idx] = logits[counter_cp, idx] * torch.exp(scale[counter_cp])
            #     logits[counter_cp, idx] = logits[counter_cp, idx] * torch.exp(cos_sim[counter_cp])
            #     # logits[counter_cp, idx] = logits[counter_cp, idx] * torch.exp(cos_sim[counter_cp]/10)
            #     # logits[counter_cp, idx] = logits[counter_cp, idx] * torch.exp(l2_dis[counter_cp])
            #     # logits[counter_cp, idx] = logits[counter_cp, idx] * torch.exp(act_rate[counter_cp])
            #     # logits[counter_cp, idx] = logits[counter_cp, idx] * torch.exp(scale2[counter_cp])
            #     counter_cp = counter_cp + 1
            # v = torch.sum(torch.mul(class_mask,cp),dim=1)
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

def iterate_data_my20(data_loader, model, temper, mask, p, threshold, class_mean):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model.forward_threshold(x, threshold)
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            # feature = model.forward_threshold_features(x, threshold)
            feature = model.forward_features(x)
            cp = torch.zeros(feature.shape).cuda()
            class_mask = torch.zeros(feature.shape).cuda()
            cp = feature * mask[pred_y,:].cuda()
            class_mask = class_mean[pred_y,:].cuda() 

            cos_sim = F.cosine_similarity(class_mask, cp, dim=1)

            cp = cp.clip(max=threshold)
            logits = model.forward_head(cp)
            # logits = logits * torch.exp(cos_sim[:, None])
            logits = logits * cos_sim[:, None]

            # v = torch.sum(torch.mul(class_mask,cp),dim=1)
            conf = temper * (torch.logsumexp((logits) / temper, dim=1))
            # conf = conf * torch.exp(cos_sim)
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_myodin(data_loader, model, epsilon, temper, mask, p, threshold, class_mean):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    temper = 800
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)   
        output = model(x)  
         
        # output = model.forward_threshold(x, threshold) 
        pred_y = torch.max(output, 1)[1].cpu().numpy()  

        feature = model.forward_features(x)
        cp = torch.zeros(feature.shape).cuda()
        class_mask = torch.zeros(feature.shape).cuda()

        cp = feature * mask[pred_y,:].cuda()
        class_mask = class_mean[pred_y,:].cuda() 
        cos_sim = F.cosine_similarity(class_mask, feature, dim=1)

        cp = cp.clip(max=threshold)
        outputs = model.forward_head(cp)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

        outputs = (outputs * cos_sim[:, None]) / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
                

        # feature =  model.forward_threshold_features(tempInputs, threshold)
        feature = model.forward_features(tempInputs)
        cp = feature * mask[maxIndexTemp,:].cuda()
        class_mask = class_mean[maxIndexTemp,:].cuda() 
        cos_sim = F.cosine_similarity(class_mask, feature, dim=1)
        cp = cp.clip(max=threshold)
        outputs = model.forward_head(cp)
        outputs = outputs * cos_sim[:, None]
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))

    return np.array(confs)

def iterate_data_mymsp(data_loader, model, mask, p, threshold, class_mean):
    m = torch.nn.Softmax(dim=-1).cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model(x)  
            # output = model.forward_threshold(x, threshold) 
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            # feature = model.forward_threshold_features(x, threshold)
            feature = model.forward_features(x)
            cp = torch.zeros(feature.shape).cuda()
            class_mask = torch.zeros(feature.shape).cuda()
            cp = feature * mask[pred_y,:].cuda()
            class_mask = class_mean[pred_y,:].cuda() 

            cos_sim = F.cosine_similarity(class_mask, feature, dim=1)

            cp = cp.clip(max=threshold)
            logits = model.forward_head(cp)
            # logits = logits * torch.exp(cos_sim[:, None])
            logits = logits * cos_sim[:, None]
            conf, _ = torch.max(m(logits), dim=-1)
            # conf = conf * cos_sim
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_simodin(data_loader, model, epsilon, temper, mask, p, threshold, class_mean):
    confs = []
    criterion = torch.nn.CrossEntropyLoss().cuda()
    temper = 800
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)      
        feature = model.forward_threshold_features(x, threshold)
        outputs = model.forward_head(feature)

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)

        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward(retain_graph=True)

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
                

        feature =  model.forward_threshold_features(tempInputs, threshold)
        outputs = model.forward_head(feature)
        class_mask = torch.zeros(feature.shape).cuda()

        class_mask = class_mean[maxIndexTemp,:].cuda() 
        cos_sim = F.cosine_similarity(class_mask, feature, dim=1)
        outputs = outputs * cos_sim[:, None]
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))

    return np.array(confs)

def iterate_data_myLINE(data_loader, model, temper, mask, p, threshold, class_mean):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            # compute output, measure accuracy and record loss.
            logits, feature = model.forward_LINE(x, threshold)
            pred_y = torch.max(logits, 1)[1].cpu().numpy()
            mask = model.fc.mask_f

            class_mask = torch.zeros(feature.shape).cuda()
            class_mask = class_mean[pred_y,:].cuda() 
            # counter_cp = 0
            # for idx in pred_y:
            #     class_mask[counter_cp,:] = class_mask[counter_cp,:] * mask[idx,:].cuda()     
            #     counter_cp = counter_cp + 1
            # cos_sim = F.cosine_similarity(class_mask, feature, dim=1)

            cos_sim = F.cosine_similarity(class_mask, feature, dim=1)
            logits = logits * cos_sim[:, None]

            # v = torch.sum(torch.mul(class_mask,cp),dim=1)
            conf = temper * (torch.logsumexp((logits) / temper, dim=1))
            # conf = conf * cos_sim
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my21(data_loader, model, temper, mask, p, threshold, class_mean):
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
            cp = torch.zeros(feature.shape).cuda()
            class_mask = torch.zeros(feature.shape).cuda()
            cp = feature * mask[pred_y,:].cuda()
            class_mask = class_mean[pred_y,:].cuda() 
            L = mask.sum(dim=1)[0]
            cos_sim = F.cosine_similarity(class_mask, cp, dim=1)

            # logits = model.forward_head(cp)
            logits = model.forward_head_mask(cp, mask.cuda())
            # logits = logits * torch.exp(cos_sim[:, None])
            logits = logits * cos_sim[:, None]

            # v = torch.sum(torch.mul(class_mask,cp),dim=1)
            conf = temper * (torch.logsumexp((logits) / temper, dim=1))
            # conf = conf * cos_sim
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_camscore(data_loader, model, temper, mask, p, threshold, class_mean):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)   
        output = model(x)
        # idx = np.argmax(output.cpu().data.numpy(), axis=1)
        class_loss, idx = torch.max(output, 1)[0], torch.max(output, 1)[1].cpu().numpy()
        # backward
        model.zero_grad()
        # class_loss = output[0,id]
        class_loss.backward()
        # 生成cam
        # grads = grad_block[b].cpu().data.numpy().squeeze()
        # feature_map = feaure_block[b].cpu().data.numpy().squeeze()
        # weights = np.mean(grads, axis=1)
        _, _, H, W = x.shape
        # cam = np.zeros(feature_map.shape[1:], dtype=np.float32)  # 二维，用于叠加
        # grads = grads.reshape([grads.shape[0], -1])

        
        # weights = np.mean(grads, axis=1)	
        # for i, w in enumerate(weights):
        #     cam += w * feature_map[i, :, :]	# 特征图加权和
        cam = np.maximum(cam, 0)
        cam = cam / cam.max()
        # cam = cv2.resize(cam, (W, H))
        input_mask = (torch.exp(torch.tensor(cam))-1).cuda()
        x = x[:,:,] * input_mask

        output = model(x)  
#         print(output.shape, output)

        pred_y = torch.max(output, 1)[1].cpu().numpy()

        feature = model.forward_threshold_features(x, threshold)
        cp = torch.zeros(feature.shape).cuda()
        class_mask = torch.zeros(feature.shape).cuda()
        cp = feature * mask[pred_y,:].cuda()
        class_mask = class_mean[pred_y,:].cuda() 
        L = mask.sum(dim=1)[0]
        cos_sim = F.cosine_similarity(class_mask, cp, dim=1)

        # logits = model.forward_head(cp)
        logits = model.forward_head(cp)
        # logits = logits * torch.exp(cos_sim[:, None])
        logits = logits * cos_sim[:, None]

        # v = torch.sum(torch.mul(class_mask,cp),dim=1)
        conf = temper * (torch.logsumexp((logits) / temper, dim=1))
        # conf = conf * cos_sim
        confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my22(data_loader, model, temper, mask, p, threshold, class_mean):
    Right = []
    Sum = []
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            output = model.forward_threshold(x, threshold)
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()
            value, _ = torch.max(m(output), dim=-1)

            feature = model.features(x)
            B, C, H, W = feature.shape
            feature = torch.flatten(feature, 2)
            weight = model.fc.weight[pred_y, :]
            cammask = feature * weight[:, :, None]
            cammask = cammask.sum(1)

            cammask = torch.clamp(cammask,min=0.0)
            # print(cammask.shape, cammask.mean(1).shape)
            # cammask = cammask / cammask.mean(1)[:, None]
            cammask = cammask / cammask.sum(1)[:, None]
            cammask = torch.exp(cammask)
            
            thresh_min = np.percentile(cammask.cpu().numpy(), 20, axis=1)
            thresh_max = np.percentile(cammask.cpu().numpy(), 40, axis=1)
            # print(thresh.shape, thresh)
            # for i in range(cammask.shape[0]):
            #     # print(1-value[i].cpu())
            #     # thresh = np.percentile(cammask[i].cpu().numpy(), 30*(1-value[i].cpu().numpy()))
                
            #     # cammask[i] = torch.clamp(cammask[i],min=thresh_min[i], max=thresh_max[i])
            #     cammask[i] = torch.where(cammask[i]>=thresh_max[i],cammask[i],0)
                # cammask[i] = torch.where(cammask[i]>=thresh,1,0)
                
                
            feature = feature * cammask[:, None, :]
            feature = torch.mean(feature, 2)
            # feature = feature.clip(max=threshold)
            # print(feature.shape)
            cp = torch.zeros(feature.shape).cuda()
            class_mask = torch.zeros(feature.shape).cuda()
            cp = feature * mask[pred_y,:].cuda()
            class_mask = class_mean[pred_y,:].cuda() 
            cos_sim = F.cosine_similarity(class_mask, cp, dim=1)
            cp = cp.clip(max=threshold)
            # logits = model.forward_head_mask(cp, mask.cuda())
            logits = model.forward_head(cp)
            # logits = logits * torch.exp(cos_sim[:, None])
            logits = logits * cos_sim[:, None]

            # v = torch.sum(torch.mul(class_mask,cp),dim=1)
            conf = temper * (torch.logsumexp((logits) / temper, dim=1))
            # conf = conf * cos_sim
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

def iterate_data_my23(data_loader, model, temper, mask, p, threshold, class_mean):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()        
            output = model(x)      
            # output = model.forward_threshold(x, threshold)
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            # feature = model.forward_threshold_features(x, threshold)
            feature = model.forward_features(x)
            cp = torch.zeros(feature.shape).cuda()
            class_mask = torch.zeros(feature.shape).cuda()
            cp = feature * mask[pred_y,:].cuda()
            class_mask = class_mean[pred_y,:].cuda() 

            cos_sim = F.cosine_similarity(class_mask, feature, dim=1)

            cp = cp.clip(max=threshold)
            
            logits = model.forward_head(cp)
            # logits = logits * torch.exp(cos_sim[:, None])
            logits = logits * cos_sim[:, None]

            # v = torch.sum(torch.mul(class_mask,cp),dim=1)
            conf = temper * (torch.logsumexp((logits) / temper, dim=1))
            # conf = conf * torch.exp(cos_sim)
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)


def iterate_data_my23_ablation(data_loader, model, temper, mask, p, threshold, class_mean):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()        
            output = model(x)      
            # output = model.forward_threshold(x, threshold)
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            # feature = model.forward_threshold_features(x, threshold)
            feature = model.forward_features(x)
            cp = torch.zeros(feature.shape).cuda()
            class_mask = torch.zeros(feature.shape).cuda()
            cp = feature * mask[pred_y,:].cuda()
            
            logits = model.forward_head(cp)

            # v = torch.sum(torch.mul(class_mask,cp),dim=1)
            conf = temper * (torch.logsumexp((logits) / temper, dim=1))
            # conf = conf * torch.exp(cos_sim)
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)


def iterate_data_cosine(data_loader, model, temper, mask, p, threshold, class_mean):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()        
            output = model(x)      
            # output = model.forward_threshold(x, threshold)
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            # feature = model.forward_threshold_features(x, threshold)
            feature = model.forward_features(x)
            cp = torch.zeros(feature.shape).cuda()
            class_mask = torch.zeros(feature.shape).cuda()
            cp = feature * mask[pred_y,:].cuda()
            class_mask = class_mean[pred_y,:].cuda() 

            # cos_sim = F.cosine_similarity(class_mask, feature, dim=1)


            cos_sim = F.cosine_similarity(class_mask, cp, dim=1)
            rate = cos_sim
            # conf = conf * scale
            # cos_sim = torch.exp(cos_sim)
            confs.extend(rate.data.cpu().numpy())
    return np.array(confs)

def iterate_data_ablation(data_loader, model, temper, mask, threshold, class_mean, cos=1):
    Right = []
    Sum = []
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()        
            output = model(x)      
            # output = model.forward_threshold(x, threshold)
    #         print(output.shape, output)

            pred_y = torch.max(output, 1)[1].cpu().numpy()

            # feature = model.forward_threshold_features(x, threshold)
            feature = model.forward_features(x)
            cp = torch.zeros(feature.shape).cuda()
            class_mask = torch.zeros(feature.shape).cuda()
            cp = feature * mask[pred_y,:].cuda()
            class_mask = class_mean[pred_y,:].cuda() 

            cos_sim = F.cosine_similarity(class_mask, feature, dim=1)

            cp = cp.clip(max=threshold)
            
            logits = model.forward_head(cp)
            # logits = logits * torch.exp(cos_sim[:, None])

            if cos == 1:
                # print('1')
                logits = logits * cos_sim[:, None]

            # v = torch.sum(torch.mul(class_mask,cp),dim=1)
            conf = temper * (torch.logsumexp((logits) / temper, dim=1))
            # conf = conf * torch.exp(cos_sim)
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)


def iterate_data_reactodin(data_loader, model, epsilon, temper, threshold):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        outputs = model.forward_threshold(x, threshold) 

        maxIndexTemp = np.argmax(outputs.data.cpu().numpy(), axis=1)
        outputs = outputs / temper

        labels = Variable(torch.LongTensor(maxIndexTemp).cuda())
        loss = criterion(outputs, labels)
        loss.backward()

        # Normalizing the gradient to binary in {0, 1}
        gradient = torch.ge(x.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        # Adding small perturbations to images
        tempInputs = torch.add(x.data, -epsilon, gradient)
        # tempInputs = torch.add(x.data, gradient, -epsilon)
        outputs = model.forward_threshold(Variable(tempInputs), threshold)
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        # if b % 100 == 0:
        #     logger.info('{} batches processed'.format(b))
        # debug
        # if b > 500:
        #    break

    return np.array(confs)


def iterate_data_reactmsp(data_loader, model, threshold):
    m = torch.nn.Softmax(dim=-1).cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            # output = model(x)  
            logits = model.forward_threshold(x, threshold) 

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)
