from utils import log
import torch
import torchvision as tv
import time

import numpy as np

from utils.test_utils import get_measures, get_measures2
from utils.cal_score import *
from argparser import *
import csv
import os
# os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from torch.autograd import Variable
import torch.nn as nn
import math

import copy
from torchvision import transforms, utils
from timm.models import create_model
from models.cifar_resnet import *
from utils.data_loader import get_dataloader_in, get_dataloader_out, cifar_out_datasets, imagenet_out_datasets
from utils.model_loader import get_model
        
from PIL import Image
class ImageListDataset(torch.utils.data.dataset.Dataset):

    def __init__(self, root_path, imglist, transform=None, target_transform=None):
        self.root_path = root_path
        self.transform = transform
        self.target_transform = target_transform
        with open(imglist) as f:
            self._indices = f.readlines()
    def __len__(self): 
        return len(self._indices)

    def __getitem__(self, index):
        img_path, label = self._indices[index].strip().split()
        img_path = os.path.join(self.root_path, img_path)
        img = Image.open(img_path).convert('RGB')             
        label = int(label)
        if self.transform is not None:
            img = self.transform(img)
        return img, label
        
        
def make_id_ood_ImageNet(args, logger):
    """Returns train and validation datasets."""

    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((384, 384)),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])

    in_set = tv.datasets.ImageFolder(args.in_datadir, val_tx)
    out_set = tv.datasets.ImageFolder(args.out_datadir, val_tx)

   
    in_loader = torch.utils.data.DataLoader(
        in_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

    return in_set, out_set, in_loader, out_loader

def make_id_ood_CIFAR(args, logger):
    """Returns train and validation datasets."""
    # crop = 480
    # crop = 32
 
    imagesize = 32
    val_tx = tv.transforms.Compose([
        tv.transforms.Resize((imagesize, imagesize)),
        tv.transforms.CenterCrop(imagesize),
        tv.transforms.ToTensor(),
        tv.transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
    ])        
        
    in_set = tv.datasets.CIFAR10("./ID_OOD_dataset/", 
                                   train=False, 
                                   transform=val_tx, 
                                   download=True)
    in_loader = torch.utils.data.DataLoader(in_set, batch_size=args.batch, shuffle=False, num_workers=4)
    if "SVHN" in args.out_datadir:
        out_set = tv.datasets.SVHN(
            root="./ID_OOD_dataset/", 
            split="test",
            download=True, 
            transform=val_tx)
    else:
        out_set = tv.datasets.ImageFolder(args.out_datadir, val_tx)

    logger.info(f"Using an in-distribution set with {len(in_set)} images.")
    logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

    out_loader = torch.utils.data.DataLoader(
        out_set, batch_size=args.batch, shuffle=False,
        num_workers=args.workers, pin_memory=True, drop_last=False)
    
    return in_set, out_set, in_loader, out_loader


def run_eval(model, in_loader, out_loader, logger, args, num_classes, out_dataset):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()
    print("In_Dataset:",args.in_dataset,"Method:",args.score,"Using BATS:", args.bats)
    
    
    bats = args.bats
    feature_std, feature_mean, lam = None, None, 0
    if bats:
        if args.in_dataset == 'imagenet':
            feature_std=torch.load(f"checkpoints/feature/vit_features_std.pt").cuda()
            feature_mean=torch.load(f"checkpoints/feature/vit_features_mean.pt").cuda()        
            lam = 1.05
        elif args.in_dataset == 'CIFAR-10':
            feature_std=torch.load(f"checkpoints/feature/{args.name}/{args.in_dataset}/{args.model}_features_std.pt").cuda()
            feature_mean=torch.load(f"checkpoints/feature/{args.name}/{args.in_dataset}/{args.model}_features_mean.pt").cuda()
            # lam = 3.25
            if args.model == 'wrn':
                lam = 2.25
            elif args.model == 'resnet18':
                lam = 3.3

        elif args.in_dataset == 'CIFAR-100':
            feature_std=torch.load(f"checkpoints/feature/{args.name}/{args.in_dataset}/{args.model}_features_std.pt").cuda()
            feature_mean=torch.load(f"checkpoints/feature/{args.name}/{args.in_dataset}/{args.model}_features_mean.pt").cuda()
            # lam = 3.25
            if args.model == 'wrn':
                lam = 1.5
            elif args.model == 'resnet18':
                lam = 1.35
            

    
    # print(feature_std.size(), feature_mean.size())

    if args.score == 'MSP':
        logger.info("Processing in-distribution data...")
        in_scores = bats_iterate_data_msp(in_loader, model,lam, feature_std, feature_mean, bats)
        logger.info("Processing out-of-distribution data...")
        out_scores = bats_iterate_data_msp(out_loader, model,lam, feature_std, feature_mean, bats)
    elif args.score == 'ODIN':
        logger.info("Processing in-distribution data...")
        in_scores = bats_iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, logger, lam, feature_std, feature_mean, bats)
        logger.info("Processing out-of-distribution data...")
        out_scores = bats_iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin, logger, lam, feature_std, feature_mean, bats)
    elif args.score == 'Energy':
        logger.info("Processing in-distribution data...")
        in_scores = bats_iterate_data_energy(in_loader, model, args.temperature_energy,lam, feature_std, feature_mean, bats)
        logger.info("Processing out-of-distribution data...")
        out_scores = bats_iterate_data_energy(out_loader, model, args.temperature_energy,lam, feature_std, feature_mean, bats)      
    elif args.score == 'GradNorm':
        logger.info("Processing in-distribution data...")
        in_scores = bats_iterate_data_gradnorm(in_loader, model, args.temperature_gradnorm, num_classes,lam, feature_std, feature_mean, bats)
        logger.info("Processing out-of-distribution data...")
        out_scores = bats_iterate_data_gradnorm(out_loader, model, args.temperature_gradnorm, num_classes,lam, feature_std, feature_mean, bats)     
    else:
        raise ValueError("Unknown score type {}".format(args.score))

    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    # fpr_list = get_measures2(in_examples, out_examples)
    # print("fpr_list:",fpr_list)
    if args.bats:
        result_path = os.path.join(args.logdir, "BATS", args.model, f"{args.in_dataset}_{args.score}.csv")
    else:
        result_path = os.path.join(args.logdir, args.name, args.model, f"{args.in_dataset}_{args.score}.csv")
    fp = open(result_path,'a+')
    result = []
    result.append(f'{args.name}_{args.bats}_{lam}')
    result.append(out_dataset)
    result.append("{:.4f}".format(auroc))
    result.append("{:.4f}".format(aupr_in))
    result.append("{:.4f}".format(aupr_out))
    result.append("{:.4f}".format(fpr95))
    context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
    context.writerow(result)
    fp.close()


    logger.info('============Results for {}============'.format(args.score))
    logger.info('=======in dataset: {}; ood dataset: {}============'.format(args.in_dataset, out_dataset))
    logger.info('AUROC: {}'.format(auroc))
    logger.info('AUPR (In): {}'.format(aupr_in))
    logger.info('AUPR (Out): {}'.format(aupr_out))
    logger.info('FPR95: {}'.format(fpr95))

    logger.flush()

    return auroc, aupr_in, aupr_out, fpr95
    

# def main(args):
#     logger = log.setup_logger(args)

#     torch.backends.cudnn.benchmark = True

#     if args.score == 'GradNorm':
#         args.batch = 1

#     if args.dataset == 'ImageNet':
#         model = create_model("vit_base_patch16_384",pretrained=True,num_classes=1000)
#         model = model.cuda()
#         in_set, out_set, in_loader, out_loader = make_id_ood_ImageNet(args, logger)
#         numc=1000
#     elif args.dataset == 'CIFAR':    
#         model = resnet18_cifar(num_classes=10)
#         model.load_state_dict(torch.load("./checkpoints/resnet18_cifar10.pth")['state_dict'])
#         model = model.cuda()
#         in_set, out_set, in_loader, out_loader = make_id_ood_CIFAR(args, logger)
#         numc=10

#     start_time = time.time()
#     run_eval(model, in_loader, out_loader, logger, args, num_classes=numc)
#     end_time = time.time()

#     logger.info("Total running time: {}".format(end_time - start_time))


def main(args):
    logger = log.setup_logger(args)
    result_save_dir = os.path.join(args.logdir, "BATS", args.model)
    if not os.path.isdir(result_save_dir):
        os.makedirs(result_save_dir)
    if args.bats:
        result_path = os.path.join(args.logdir, "BATS", args.model, f"{args.in_dataset}_{args.score}.csv")
    else:
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


    if args.score == 'GradNorm':
        args.batch = 1

    in_dataset = args.in_dataset

    in_save_dir = os.path.join(args.logdir, "BATS", args.model)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    loader_in_dict = get_dataloader_in(args, split=('val'))
    in_loader, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path is not None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    
    # model.eval()

    if args.out_dataset is not None:
        out_dataset = args.out_dataset
        loader_out_dict = get_dataloader_out(args, (None, out_dataset), split=('val'))
        out_loader = loader_out_dict.val_ood_loader

        in_set, out_set = loader_in_dict.val_dataset, loader_out_dict.val_dataset
        logger.info(f"Using an in-distribution set with {len(in_set)} images.")
        logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")


        start_time = time.time()
        run_eval(model, in_loader, out_loader, logger, args, num_classes=num_classes, out_dataset=out_dataset)
        end_time = time.time()

        logger.info("Total running time: {}".format(end_time - start_time))
    
    else:
        AUroc, AUPR_in, AUPR_out, Fpr95 = [], [], [], []
        out_datasets = []
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
            auroc, aupr_in, aupr_out, fpr95 = run_eval(model, in_loader, out_loader, logger, args, num_classes=num_classes, out_dataset=out_dataset)
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

        if args.bats:
            result_path = os.path.join(args.logdir, "BATS", args.model, f"{args.in_dataset}_{args.score}.csv")
        else:
            result_path = os.path.join(args.logdir, args.name, args.model, f"{args.in_dataset}_{args.score}.csv")
        fp = open(result_path,'a+')
        result = []
        result.append(f'{args.name}_{args.bats}')
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



if __name__ == "__main__":
    parser = get_argparser()

    main(parser.parse_args())

    