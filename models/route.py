import torch
import torch.nn as nn
import numpy as np


class RouteDICE(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, p=90, conv1x1=False, info=None):
        super(RouteDICE, self).__init__(in_features, out_features, bias)
        if conv1x1:
            self.weight = nn.Parameter(torch.Tensor(out_features, in_features, 1, 1))
        self.p = p
        self.info = info
        self.masked_w = None

    def calculate_mask_weight(self):
        self.contrib = self.info[None, :] * self.weight.data.cpu().numpy()

        # print('fc',self.weight.shape)
        # print('contrib', self.contrib.shape)

        # self.contrib = np.abs(self.contrib)
        # self.contrib = np.random.rand(*self.contrib.shape)
        # self.contrib = self.info[None, :]
        # self.contrib = np.random.rand(*self.info[None, :].shape)
        self.thresh = np.percentile(self.contrib, self.p)

        mask = torch.Tensor((self.contrib > self.thresh))
        self.masked_w = (self.weight.squeeze().cpu() * mask).cuda()

    def forward(self, input):
        if self.masked_w is None:
            self.calculate_mask_weight()

        vote = input[:, None, :] * self.masked_w.cuda()
        if self.bias is not None:
            out = vote.sum(2) + self.bias
        else:
            out = vote.sum(2)
        return out

class RouteLUNCH(nn.Linear):

    def __init__(self, in_features, out_features, bias=True, p_w=10, p_a=10, conv1x1=False, info=None, clip_threshold = 1e10):
        super(RouteLUNCH, self).__init__(in_features, out_features, bias)
        self.p = p_a
        self.weight_p = p_w
        self.clip_threshold = clip_threshold
        self.info = info
        self.masked_w = None
        self.mask_f = None
        self.l_weight = self.weight.data.cuda()

    def calculate_shap_value(self):
        self.contrib = self.info.T
        self.mask_f = torch.zeros(self.out_features,self.in_features)
        self.masked_w = torch.zeros((self.out_features,self.out_features,self.in_features))

        for class_num in range(self.out_features):
            self.matrix = abs(self.contrib[class_num,:]) * self.weight.data.cpu().numpy()
            self.thresh = np.percentile(self.matrix, self.weight_p)
            mask_w = torch.Tensor((self.matrix > self.thresh))
            self.masked_w[class_num,:,:] = (self.weight.squeeze().cpu() * mask_w).cuda()
            self.class_thresh = np.percentile(self.contrib[class_num,:], self.p)
            self.mask_f[class_num,:] = torch.Tensor((self.contrib[class_num,:] > self.class_thresh))

    def forward(self, input):    
        if self.masked_w is None:
            self.calculate_shap_value()
        pre = input[:, None, :] * self.weight.data.cuda()
        if self.bias is not None:
            pred = pre.sum(2) + self.bias
        else:
            pred = pre.sum(2)
        pred = torch.nn.functional.softmax(pred, dim=1)   
        preds = np.argmax(pred.cpu().detach().numpy(), axis=1)
       
        counter_cp = 0
        cp = torch.zeros((len(input), self.in_features)).cuda()
        for idx in preds:
            cp[counter_cp,:] = input[counter_cp,:] * self.mask_f[idx,:].cuda()     
            counter_cp = counter_cp + 1
