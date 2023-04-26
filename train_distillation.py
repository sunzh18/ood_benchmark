from __future__ import print_function
import shutil
import torch.backends.cudnn as cudnn

import csv
import heapq
import random
from utils import log
import time
import math

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
import torchvision
import torchvision.models as models
import torchvision.transforms as transforms
from torch.cuda.amp import GradScaler, autocast
from torch.optim.lr_scheduler import MultiStepLR, StepLR
from torch.utils.data import DataLoader, Subset, random_split
from tqdm import tqdm

from argparser import *
from datasets.dataset_largescale import validation_split
from utils.data_loader import get_dataloader_in, get_dataloader_out
from utils.model_loader import get_model
import wandb
import os

cudnn.benchmark = True

class transfer_conv(nn.Module):
    def __init__(self, in_feature, out_feature):
        super().__init__()
        self.in_feature = in_feature
        self.out_feature = out_feature
        self.Connectors = nn.Sequential(
            nn.Conv2d(in_feature, out_feature, kernel_size=1, stride=1, padding=0, bias=False),
            nn.BatchNorm2d(out_feature), nn.ReLU())
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()
    def forward(self, student):
        student = self.Connectors(student)
        return student

class statm_loss(nn.Module):
    def __init__(self, eps=2):
        super(statm_loss, self).__init__()
        self.eps = eps

    def forward(self,x, y):
        # x = x.view(x.size(0),x.size(1),-1)
        # y = y.view(y.size(0),y.size(1),-1)
        # x_mean = x.mean(dim=2)
        # y_mean = y.mean(dim=2)
        x_mean = x.mean(dim=1)
        y_mean = y.mean(dim=1)
        # print(x_mean.size(),y_mean.size())
        # mean_gap = (x_mean-y_mean).pow(2).mean(1)
        mean_gap = (x_mean-y_mean).pow(2)
        return mean_gap.mean()

# 测试网络正确率
def test_model(model, data_loader):
    criterion = nn.CrossEntropyLoss()                     #交叉熵损失  
    num=0
    accu=0
    Loss=0
    
    print("test: ")
    with torch.no_grad():
        for data,target in tqdm(data_loader):
            data, target = data.to(device), target.to(device)   
    #         print(data.shape, data)
            output = model(data)  
    #         print(output.shape, output)
            loss=criterion(output,target)
            Loss += loss.item()

            pred_y = torch.max(output, 1)[1].cpu().numpy()
            y_label = target.cpu().numpy()
            accu += (pred_y == y_label).sum()
    #         print(pred_y, y_label)
    #         print(accu1 / num1, accu2 / num2)

            num += len(y_label)
    accu /= num
    Loss /= num
    # print("loss:",Loss,"accuracy:",accu)
    # print("loss:",Loss,"accuracy:",accu)
    return accu, Loss
  
def main(args):
    in_dataset = args.in_dataset

    ckpt_save_dir = os.path.join("checkpoints", "network", args.name, args.in_dataset)
    if not os.path.exists(ckpt_save_dir):
        os.makedirs(ckpt_save_dir)

    loader_in_dict = get_dataloader_in(args, split=('train','val'))
    train_dataloader, test_dataloader, num_classes = loader_in_dict.train_loader, loader_in_dict.val_loader, loader_in_dict.num_classes
    train_set, test_set = loader_in_dict.train_dataset, loader_in_dict.val_dataset
    args.num_classes = num_classes

    kwargs = {'num_workers': 2, 'pin_memory': True}
    if args.validation:
        train_set, val_set = validation_split(train_set, val_share=0.1)
        val_dataloader = torch.utils.data.DataLoader(
            val_set, batch_size=args.batch, shuffle=True, **kwargs)
        
    train_dataloader = torch.utils.data.DataLoader(train_set, batch_size=args.batch, shuffle=True, **kwargs)
    
    print(f"Using an train set with {len(train_set)} images.")
    print(f"Using an val set with {len(test_set)} images.")
    print(len(train_dataloader), len(test_dataloader))

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True
    teacher_model = get_model(args, num_classes, load_ckpt=False)
    checkpoint = torch.load(args.model_path)
    teacher_model.load_state_dict(checkpoint['state_dict'])
    teacher_model.eval()
    for param in teacher_model.parameters():
        param.requires_grad = False

    student_model = get_model(args, num_classes, load_ckpt=False)
    # checkpoint = torch.load(f'checkpoints/network/{args.name}/{args.in_dataset}/{args.model}_parameter.pth')
    # student_model.load_state_dict(checkpoint['state_dict'])
    if args.name == "KD_teacher_init":
        student_model.load_state_dict(checkpoint['state_dict'])
    

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
    # trainable_list = nn.ModuleList([])
    # trainable_list.append(net_s)
    # conector = transfer_conv(net_s.module.fea_dim, net_t.module.fea_dim).cuda()
    # trainable_list.append(conector)
    # optimizer = torch.optim.SGD(trainable_list.parameters(),  lr = args.lr, momentum = args.momentum,weight_decay = args.weight_decay)
    t_accu = 0
    num=0
    for data, target in tqdm(test_dataloader):  
        data, target = data.to(device), target.to(device)    
        with torch.no_grad():
            feat_t = teacher_model.forward_features(data)
            feat_t = torch.where(feat_t<(feature_std*lam+feature_mean),feat_t,feature_std*lam+feature_mean)
            feat_t = torch.where(feat_t>(-feature_std*lam+feature_mean),feat_t,-feature_std*lam+feature_mean)
            # feat_t, pred_t = net_t(data, is_adain=True)
        pred_t = teacher_model.forward_head(feat_t)
        pred_y = torch.max(pred_t, 1)[1].cpu().numpy()
        y_label = target.cpu().numpy()
        t_accu += (pred_y == y_label).sum()
#         print(pred_y, y_label)
#         print(accu1 / num1, accu2 / num2)

        num += len(y_label)
    t_accu /= num
    print(":\teacher-- accuracy =",t_accu)


    Train_accuracy = list()      #训练集正确率列表
    Val_accuracy = list()        #验证集正确率列表
    Train_loss = list()
    Val_loss=list()
    best_val_acc = 0.0
    
    criterion = nn.CrossEntropyLoss()                     #交叉熵损失                 
    optimizer = optim.SGD(student_model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=5e-4)

    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,100,150],gamma = 0.1)
    if args.lr < 0.01:
        scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[30, 60 ,90],gamma = 0.1)
    
    # wandb.watch(student_model, log="all", log_freq=10)

    for epoch in range(n_epochs):
        train_acc, train_loss = train(train_dataloader, teacher_model, student_model, optimizer, epoch, lam, feature_std, feature_mean)
        scheduler.step()
        student_model.eval() 
        state_dict = {'state_dict':student_model.state_dict(), 
                      'optimizer':optimizer.state_dict(), 
                      'epoch':epoch,
                      'scheduler':scheduler}
        
        torch.save(state_dict , f'{ckpt_save_dir}/{args.model}_parameter.pth')

        print('testing the models......')
        test_acc, test_loss = test_model(student_model, test_dataloader)
        
        if best_val_acc <= test_acc:
            best_val_acc = test_acc
            torch.save(state_dict , f'{ckpt_save_dir}/{args.model}_best_parameter.pth')
            
        
        Train_accuracy.append(train_acc)
        Val_accuracy.append(test_acc)
        Train_loss.append(train_loss)
        Val_loss.append(test_loss)
        print("epoch =",epoch,":\ntrain-- accuracy =",train_acc,",loss =",train_loss)
        print("epoch =",epoch,":\nteacher-- accuracy =",t_accu,",val-- accuracy =",test_acc,",loss =",test_loss)
        
        # log metrics to wandb
        if args.wandb != None:
            wandb.log({
                "Train Accuracy": 100. * train_acc,
                "Val Accuracy": 100. * test_acc,
                "Loss": train_loss})


        fp = open(f'{draw_result_save_dir}/{args.model}_{learning_rate}_accuracy.csv','a+')
        acc_result = []
        acc_result.append(epoch)
        acc_result.append(train_acc)
        acc_result.append(test_acc)
        context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
        context.writerow(acc_result)
        fp.close()

        
        fp = open(f'{draw_result_save_dir}/{args.model}_{learning_rate}_loss.csv','a+')
        loss_result = []
        loss_result.append(epoch)
        loss_result.append(train_loss)
        loss_result.append(test_loss)
        context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
        context.writerow(loss_result)
        fp.close()

    wandb.finish()

def train(train_dataloader, net_t, net_s, optimizer, epoch, lam=None, feature_std=None, feature_mean=None, conector=None):
    net_s.train()
    num=0
    accu=0
    pred_accu = 0
    Loss=0
    # conector.train()
    for data, target in tqdm(train_dataloader):  
        data, target = data.to(device), target.to(device)    
        with torch.no_grad():
            feat_t = net_t.forward_features(data)
            feat_t = torch.where(feat_t<(feature_std*lam+feature_mean),feat_t,feature_std*lam+feature_mean)
            feat_t = torch.where(feat_t>(-feature_std*lam+feature_mean),feat_t,-feature_std*lam+feature_mean)
            # feat_t, pred_t = net_t(data, is_adain=True)
        feat_s = net_s.forward_features(data)
        pred_s = net_s.forward_head(feat_s)
        # feat_s = conector(feat_s)

        # statmloss = statm_loss()
        # loss_stat = statmloss(feat_s, feat_t.detach())

        pred_sc = net_t.forward_head(feat_s)
        pred_t = net_t.forward_head(feat_t)

        # loss_kd = loss_stat + F.mse_loss(pred_sc, pred_t)*args.weight
        loss_kd = F.mse_loss(feat_s, feat_t) + F.mse_loss(pred_sc, pred_t) * args.weight
        # loss_kd = F.mse_loss(pred_sc, pred_t) * args.weight
        loss_ce = F.cross_entropy(pred_s, target)

        loss = loss_ce + loss_kd
        Loss += loss.item() 

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        pred_y = torch.max(pred_s, 1)[1].cpu().numpy()
        y_label = target.cpu().numpy()
        pred_accu += (pred_y == y_label).sum()
#         print(pred_y, y_label)
        num += len(y_label)
    train_acc = pred_accu / num
    train_loss = Loss / num
    return train_acc, train_loss
        
        


parser = get_argparser()
parser.add_argument('--weight', type=float, default=1, help='weight for kd loss')

args = parser.parse_args()

train_on_gpu = torch.cuda.is_available() 

device = torch.device("cuda") 
if train_on_gpu:                                                   #部署到GPU上
    device = torch.device("cuda") 
else:
    device = torch.device("cpu")

n_epochs = args.epochs
batch_size = args.batch
learning_rate = args.lr

# if args.test:
#     args.wandb = False
# start a new wandb run to track this script
if args.wandb != None:
    wandb.init(
        # set the wandb project where this run will be logged
        project=f"{args.name}_{args.in_dataset}_{args.model}",
        name=f"{args.wandb}",
        
        # track hyperparameters and run metadata
        config={
        "method":args.name,
        "learning_rate": learning_rate,
        "model": args.model,
        "dataset": args.in_dataset,
        "epochs": args.epochs,
        }
    )

print(args)

print(f'{args.model}')

# model = nn.DataParallel(model)
# torch.save(cnn, "net_params.pkl")


draw_result_save_dir = os.path.join("draw_result", args.name, args.in_dataset)
if not os.path.isdir(draw_result_save_dir):
    os.makedirs(draw_result_save_dir)

if not os.path.exists(f'{draw_result_save_dir}/{args.model}_{learning_rate}_accuracy.csv'):
    fp = open(f'{draw_result_save_dir}/{args.model}_{learning_rate}_accuracy.csv','a+')
    loss_result = []
    loss_result.append('epoch')
    loss_result.append('train_acc')
    loss_result.append('test_acc')
    context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
    context.writerow(loss_result)
    fp.close()

    fp = open(f'{draw_result_save_dir}/{args.model}_{learning_rate}_loss.csv','a+')
    loss_result = []
    loss_result.append('epoch')
    loss_result.append('train_loss')
    loss_result.append('test_loss')
    context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
    context.writerow(loss_result)
    fp.close()



if __name__ == '__main__':
    main(args)
