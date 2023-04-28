# from draw import draw_box
# import imp
import csv
import heapq
import random
from utils import log
import time

import cv2
import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
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

# from sklearn.model_selection import KFold
 
# print(torch.__version__)



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


def train_model(args, model, train_dataloader, test_dataloader):

    criterion = nn.CrossEntropyLoss()                     #交叉熵损失                 
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9, weight_decay=0.0001)
    scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[50,75,90],gamma = 0.1)
    # if args.in_dataset == 'CIFAR-10':
    #     scheduler = optim.lr_scheduler.MultiStepLR(optimizer, milestones=[100,150],gamma = 0.1)


    Train_accuracy = list()      #训练集正确率列表
    Val_accuracy = list()        #验证集正确率列表
    Train_loss = list()
    Val_loss=list()
    
    best_val_acc = 0.0
    # wandb.watch(model, log="all")
    # wandb.watch(model, log="all", log_freq=10)
    
    for epoch in range(n_epochs):
        model.train()                                           ###启用 BatchNormalization 和 Dropout
        num=0
        accu=0
        pred_accu = 0
        Loss=0
        
        for data, target in tqdm(train_dataloader):       
            data, target = data.to(device), target.to(device)    

            output = model(data)
            loss = criterion(output, target)   
            optimizer.zero_grad()                           ##loss关于weight的导数置零
                                         
            Loss += loss.item()  
            loss.backward()
            optimizer.step()

            pred_y = torch.max(output, 1)[1].cpu().numpy()
            y_label = target.cpu().numpy()
            pred_accu += (pred_y == y_label).sum()
    #         print(pred_y, y_label)
            num += len(y_label)

            #train_loss += loss.item()*data.size(0)
        scheduler.step()
        model.eval()  
        train_acc = pred_accu / num
        train_loss = Loss / num
#         if epoch % 10 == 0:
        state_dict = {'state_dict':model.state_dict(), 
                      'optimizer':optimizer.state_dict(), 
                      'epoch':epoch,
                      'scheduler':scheduler}
        if args.arch:
            torch.save(state_dict , f'{ckpt_save_dir}/{args.model}_{args.arch}_parameter.pth')
        else:
            torch.save(state_dict , f'{ckpt_save_dir}/{args.model}_parameter.pth')
    
        test_acc, test_loss = test_model(model, test_dataloader)
        
        # if best_val_acc <= test_acc:
        #     best_val_acc = test_acc
        #     torch.save(state_dict , f'{ckpt_save_dir}/{args.model}_best_parameter.pth')
            
        
        Train_accuracy.append(train_acc)
        Val_accuracy.append(test_acc)
        Train_loss.append(train_loss)
        Val_loss.append(test_loss)
        print("epoch =",epoch,":\ntrain-- accuracy =",train_acc,"loss =",train_loss)
        print("epoch =",epoch,":\nval-- accuracy =",test_acc,"loss =",test_loss)
        
        if args.wandb != None:
            # log metrics to wandb
            wandb.log({
                "Train Accuracy": 100. * train_acc,
                "Val Accuracy": 100. * test_acc,
                "Loss": train_loss})


        fp = open(f'{draw_result_save_dir}/{args.model}_{args.arch}_{learning_rate}_accuracy.csv','a+')
        acc_result = []
        acc_result.append(epoch)
        acc_result.append(train_acc)
        acc_result.append(test_acc)
        context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
        context.writerow(acc_result)
        fp.close()

        
        # fp = open(f'{draw_result_save_dir}/{args.model}_{learning_rate}_loss.csv','a+')
        # loss_result = []
        # loss_result.append(epoch)
        # loss_result.append(train_loss)
        # loss_result.append(test_loss)
        # context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
        # context.writerow(loss_result)
        # fp.close()

    wandb.finish()


    return



parser = get_argparser()

args = parser.parse_args()
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

load_ckpt = False
if args.model_path != None:
    load_ckpt = True
model = get_model(args, num_classes, load_ckpt=load_ckpt)
# checkpoint = torch.load('checkpoints/network/BATS/CIFAR-10/resnet18_parameter.pth')
# model.load_state_dict(checkpoint['state_dict'])

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
        project=f"{args.in_dataset}_{args.model}_{args.lr}",
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

print(f"Using an train set with {len(train_set)} images.")
print(f"Using an val set with {len(test_set)} images.")
print(len(train_dataloader), len(test_dataloader))

draw_result_save_dir = os.path.join("draw_result", args.name, args.in_dataset)
if not os.path.isdir(draw_result_save_dir):
    os.makedirs(draw_result_save_dir)

if not os.path.exists(f'{draw_result_save_dir}/{args.model}_{args.arch}_{learning_rate}_accuracy.csv'):
    fp = open(f'{draw_result_save_dir}/{args.model}_{args.arch}_{learning_rate}_accuracy.csv','a+')
    loss_result = []
    loss_result.append('epoch')
    loss_result.append('train_acc')
    loss_result.append('test_acc')
    context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
    context.writerow(loss_result)
    fp.close()

    # fp = open(f'{draw_result_save_dir}/{args.model}_{learning_rate}_loss.csv','a+')
    # loss_result = []
    # loss_result.append('epoch')
    # loss_result.append('train_loss')
    # loss_result.append('test_loss')
    # context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
    # context.writerow(loss_result)
    # fp.close()



# opts = modify_command_options(opts)
if __name__ == '__main__':
    # train_k_fold_model(opts, cnn, model_name, 5)
    # train_model(opts, cnn, train_dataloader, val_dataloader, 'clean_model'))
    if args.test:
        model.eval()  
        test_acc, _ = test_model(model, test_dataloader)
        print("test-- accuracy =",test_acc)
    else:
        train_model(args, model, train_dataloader, test_dataloader)

    
    # trade_train_model(opts, cnn, train_dataloader, val_dataloader, 'trade_model')
#     print("train:")
#     test_model(cnn, train_dataloader)
#     print("val:")
#     test_model(cnn, val_dataloader)
    





