
import torch
from torch import nn
import torch.nn.functional as nnF
import math


class Net(nn.Module):

    def __init__(self, densenet, num_class, freeze_conv=False, n_extra_info=0, p_dropout=0.5, neurons_class=256,
                 feat_reducer=None, classifier=None, dim=32):

        super(Net, self).__init__()
        self.features = nn.Sequential(*list(densenet.children())[:-1])
        # freezing the convolution layers
        if freeze_conv:
            for param in self.features.parameters():
                param.requires_grad = False

        # if feat_reducer is None:
        #     self.feat_reducer = nn.Sequential(
        #         nn.Linear(1024, neurons_class),
        #         nn.BatchNorm1d(neurons_class),
        #         nn.ReLU(),
        #         nn.Dropout(p=p_dropout)
        #     )
        # else:
        #     self.feat_reducer = feat_reducer

        self.mu_embeddings = nn.Parameter(torch.zeros(1, 1024))
        self.fc1 = nn.Linear(1024, dim)
        self.fc2 = nn.Linear(dim, 1024)

        if classifier is None:
            self.classifier = nn.Sequential(
                nn.Linear(neurons_class + n_extra_info, num_class)
            )
        else:
            self.classifier = classifier

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
            elif isinstance(m, nn.Linear):
                m.bias.data.zero_()

        self.fc1.bias.requires_grad = False
        self.fc2.bias.requires_grad = False

    def cal_SigmalMatrix_VIM(self, x):
        resd = x - self.mu_embeddings
        fc1 = self.fc1(resd)
        fc1_w = self.fc1.weight
        fc2 = self.fc2(fc1)
        fc2_w = self.fc2.weight
        new_x = fc2+self.mu_embeddings
        r = fc1_w @ fc2_w
        lw = fc1_w - fc2_w.t()
        return new_x, r, lw

    def forward(self, img, extra_info=None):

        xf = self.features(img)
        x = nnF.relu(xf, inplace=True)
        x = nnF.adaptive_avg_pool2d(x, (1, 1)).view(xf.size(0), -1)

        # x = self.feat_reducer(x)
        re_x, r, lw = self.cal_SigmalMatrix_VIM(x)

        if extra_info is not None:
            agg = torch.cat((x, extra_info), dim=1)
        else:
            agg = x

        x = self.classifier(agg)

        return x, agg, re_x, r, lw
