from utils import log
from models.resnetv2 import * 
import torch
import time

import numpy as np

import os

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from utils.mahalanobis_lib import get_Mahalanobis_score


def iterate_data_msp(data_loader, model):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_odin(data_loader, model, epsilon, temper, logger):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        outputs = model(x)

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
        outputs = model(Variable(tempInputs))
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


def iterate_data_energy(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

def iterate_data_react(data_loader, model, temper, threshold):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model.forward_threshold(x, threshold)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)

def iterate_data_LINE(data_loader, model, temper, threshold):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits, _ = model.forward_LINE(x, threshold)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def iterate_data_gradnorm(data_loader, model, temperature, num_classes):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        inputs = Variable(x.cuda(), requires_grad=True)

        model.zero_grad()
        outputs = model(inputs)  
        targets = torch.ones((inputs.shape[0], num_classes)).cuda()
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward(retain_graph=True)
        
        # if num_classes==1000:
        #     layer_grad = model.head.weight.grad.data
        # else:
        #     layer_grad = model.fc.weight.grad.data
        layer_grad = model.fc.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(layer_grad_norm)

    return np.array(confs)

def iterate_data_mahalanobis(data_loader, model, num_classes, sample_mean, precision,
                             num_output, magnitude, regressor, logger):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        # if b % 10 == 0:
        #     logger.info('{} batches processed'.format(b))
        x = x.cuda()

        Mahalanobis_scores = get_Mahalanobis_score(x, model, num_classes, sample_mean, precision, num_output, magnitude)
        scores = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
        confs.extend(scores)
    return np.array(confs)


def iterate_data_kl_div(data_loader, model):
    probs, labels = [], []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            prob = m(logits)
            probs.extend(prob.data.cpu().numpy())
            labels.extend(y.numpy())

    return np.array(probs), np.array(labels)


def kl(p, q):
    """Kullback-Leibler divergence D(P || Q) for discrete distributions
    Parameters
    ----------
    p, q : array-like, dtype=float, shape=n
    Discrete probability distributions.
    """
    # p = np.asarray(p, dtype=np.float)
    # q = np.asarray(q, dtype=np.float)

    return np.sum(np.where(p != 0, p * np.log(p / q), 0))


def iterate_data_dice(data_loader, model, temper):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            # compute output, measure accuracy and record loss.
            logits = model(x)

            conf = temper * torch.logsumexp(logits / temper, dim=1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)




def bats_iterate_data_msp(data_loader, model, lam=None, feature_std=None, feature_mean=None, bats=False):
    confs = []
    m = torch.nn.Softmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()
            features = model.forward_features(x)
            
            if bats:
                features = torch.where(features<(feature_std*lam+feature_mean),features,feature_std*lam+feature_mean)
                features = torch.where(features>(-feature_std*lam+feature_mean),features,-feature_std*lam+feature_mean)
            
            logits = model.forward_head(features)
            # logits = model(x)

            conf, _ = torch.max(m(logits), dim=-1)
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def bats_iterate_data_odin(data_loader, model, epsilon, temper, logger, lam=None, feature_std=None, feature_mean=None, bats=False):
    criterion = torch.nn.CrossEntropyLoss().cuda()
    confs = []
    for b, (x, y) in enumerate(data_loader):
        x = Variable(x.cuda(), requires_grad=True)
        
        features = model.forward_features(x)
        if bats:
            features = torch.where(features<(feature_std*lam+feature_mean),features,feature_std*lam+feature_mean)
            features = torch.where(features>(-feature_std*lam+feature_mean),features,-feature_std*lam+feature_mean)
        
        outputs = model.forward_head(features)

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
                
        features = model.forward_features(Variable(tempInputs))
        
        if bats:
            features = torch.where(features<(feature_std*lam+feature_mean),features,feature_std*lam+feature_mean)
            features = torch.where(features>(-feature_std*lam+feature_mean),features,-feature_std*lam+feature_mean)
        
        outputs = model.forward_head(features)
        
        outputs = outputs / temper
        # Calculating the confidence after adding perturbations
        nnOutputs = outputs.data.cpu()
        nnOutputs = nnOutputs.numpy()
        nnOutputs = nnOutputs - np.max(nnOutputs, axis=1, keepdims=True)
        nnOutputs = np.exp(nnOutputs) / np.sum(np.exp(nnOutputs), axis=1, keepdims=True)

        confs.extend(np.max(nnOutputs, axis=1))
        if b % 100 == 0:
            logger.info('{} batches processed'.format(b))

    return np.array(confs)

def bats_iterate_data_energy(data_loader, model, temper, lam=None, feature_std=None, feature_mean=None, bats=False):
    confs = []
    for b, (x, y) in enumerate(data_loader):
        with torch.no_grad():
            x = x.cuda()            
            # print(x.size())
            features = model.forward_features(x)
            # print(features.size())
            # f = model.features(x)
            # print(f.size())
            if bats:
                features = torch.where(features<(feature_std*lam+feature_mean),features,feature_std*lam+feature_mean)
                features = torch.where(features>(-feature_std*lam+feature_mean),features,-feature_std*lam+feature_mean)
            
            logits = model.forward_head(features)
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            
            confs.extend(conf.data.cpu().numpy())
    return np.array(confs)


def bats_iterate_data_gradnorm(data_loader, model, temperature, num_classes, lam=None, feature_std=None, feature_mean=None, bats=False):
    confs = []
    logsoftmax = torch.nn.LogSoftmax(dim=-1).cuda()
    for b, (x, y) in enumerate(data_loader):
        if b % 100 == 0:
            print('{} batches processed'.format(b))
        inputs = Variable(x.cuda(), requires_grad=True)

        model.zero_grad()
        
        features = model.forward_features(inputs)
        if bats:
            features = torch.where(features<(feature_std*lam+feature_mean),features,feature_std*lam+feature_mean)
            features = torch.where(features>(-feature_std*lam+feature_mean),features,-feature_std*lam+feature_mean)        
        outputs = model.forward_head(features)
        
        targets = torch.ones((inputs.shape[0], num_classes)).cuda()
        outputs = outputs / temperature
        loss = torch.mean(torch.sum(-targets * logsoftmax(outputs), dim=-1))

        loss.backward(retain_graph=True)
        
        # if num_classes==1000:
        #     layer_grad = model.head.weight.grad.data
        # else:
        #     layer_grad = model.fc.weight.grad.data
        layer_grad = model.fc.weight.grad.data

        layer_grad_norm = torch.sum(torch.abs(layer_grad)).cpu().numpy()
        confs.append(layer_grad_norm)

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
            scale = s1 / s2 - 1.0
            feature_prun = feature_prun * torch.exp(scale[:, None])
            logits = model.forward_head(feature_prun)
            conf = temper * (torch.logsumexp(logits / temper, dim=1))
            confs.extend(conf.data.cpu().numpy())

    return np.array(confs)

