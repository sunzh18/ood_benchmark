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
from compute_gradcam import *

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
    # class_mean = np.squeeze(class_mean)

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

    elif args.score == 'my_score20':
        p = 0
        if args.p:
            p = args.p
        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_my20(in_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_my20(out_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)
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
       
        args.threshold = 1.0  #0.8
        p = 0
        args.p = 0

        info = np.load(f"cache/{args.name}/{args.in_dataset}_{args.model}_meanshap_class.npy")
        model = get_model(args, num_classes, load_ckpt=True, info=info, LU=True)
        model.eval()
        fc_w = extact_mean_std(args, model)
        mask, class_mean = get_class_mean2(args, fc_w)
        # mask, class_mean = get_class_mean(args)
        class_mean = class_mean.cuda()
        class_mean = class_mean.clip(max=args.threshold)
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
    
    elif args.score == 'gradcam':
        if args.model == 'resnet50':
            model.layer4[-1].register_forward_hook(farward_hook)  # 9
            model.layer4[-1].register_backward_hook(backward_hook)
            # model.avgpool.register_forward_hook(farward_hook)  # 9
            # model.avgpool.register_backward_hook(backward_hook)
            # print(model)
        elif args.model == 'densenet':
            model.relu.register_forward_hook(farward_hook)  # 9
            model.relu.register_backward_hook(backward_hook)
        p = 0
        if args.p:
            p = args.p
        if in_scores is None: 
            logger.info("Processing in-distribution data...")
            in_scores = iterate_data_camscore(in_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_camscore(out_loader, model, args.temperature_energy, mask, p, args.threshold, class_mean)
        analysis_score(args, in_scores, out_scores, out_dataset)
    

    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
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
    plt.savefig(save_pic_filename,dpi=600)
    
    plt.close()

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
    dataset = loader_in_dict.val_dataset
    args.num_classes = num_classes
    in_scores=None

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    model.eval()
    print(model)
    fc_w = extact_mean_std(args, model)
    mask, class_mean = get_class_mean2(args, fc_w)
    # mask, class_mean = get_class_mean(args)
    class_mean = class_mean.cuda()
    class_mean = class_mean.clip(max=args.threshold)
    if args.out_dataset is not None:
        out_dataset = args.out_dataset
        loader_out_dict = get_dataloader_out(args, (None, out_dataset), split=('val'))
        out_loader = loader_out_dict.val_ood_loader

        in_set, out_set = loader_in_dict.val_dataset, loader_out_dict.val_dataset
        logger.info(f"Using an in-distribution set with {len(in_set)} images.")
        logger.info(f"Using an out-of-distribution set with {len(out_set)} images.")

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

        result_path = os.path.join(args.logdir, args.name, args.model, f"{args.in_dataset}_{args.score}.csv")
        fp = open(result_path,'a+')
        result = []

        result.append(f'p: {args.p}/threshold{args.threshold}')
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

un_norm_cifar = transforms.Normalize(
    mean=[-0.4914/0.2023, -0.4822/0.1994, -0.4465/0.2010],
    std=[1/0.2023, 1/0.1994, 1/0.2010]
)

un_norm_largescale = transforms.Normalize(
    mean=[-0.485/0.229, -0.456/0.224, -0.406/0.225],
    std=[1/0.229, 1/0.224, 1/0.225]
)


unloader = transforms.ToPILImage()
def show_gradcam(args, model, data_loader, dataset, mask, fc_w):
    output_dir = f"gradcam/{args.name}/{args.in_dataset}_{args.model}"
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)
    
    if args.model == 'resnet50':
        model.layer4[-1].register_forward_hook(farward_hook)  # 9
        model.layer4[-1].register_backward_hook(backward_hook)
        un_norm = un_norm_largescale
        # model.avgpool.register_forward_hook(farward_hook)  # 9
        # model.avgpool.register_backward_hook(backward_hook)
        # print(model)
    elif args.model == 'densenet':
        model.relu.register_forward_hook(farward_hook)  # 9
        model.relu.register_backward_hook(backward_hook)
        un_norm = un_norm_cifar
        # print(model)
    print(fc_w.shape)
    for b, (x, y) in enumerate(data_loader):
        # imagepath = dataset.imgs[b][0]
        img = un_norm(x[0].detach().cpu())
        img = unloader(img)
        # print(type(image))
        img = cv2.cvtColor(np.asarray(img),cv2.COLOR_RGB2BGR)

        # img = cv2.imread(imagepath)
        x = Variable(x.cuda(), requires_grad=True)   
        output = model(x)
        # idx = np.argmax(output.cpu().data.numpy(), axis=1)
        class_loss, idx = torch.max(output, 1)[0], torch.max(output, 1)[1].cpu().numpy()
        # backward
        model.zero_grad()
        # class_loss = output[0,id]
        class_loss.backward()
        # 生成cam
        grads_val = grad_block[b].cpu().data.numpy().squeeze()
        fmap = feaure_block[b].cpu().data.numpy().squeeze()
        # print(f'fc_w{idx}: {fc_w[idx]}')
        print(f'{idx}')
        # 保存cam图片
        cam_show_img(img, fmap, grads_val, output_dir, f'{idx}+{b}', mask[idx], args.p, fc_w[idx])

def test_gradcam(args):
    if args.in_dataset == "CIFAR-10":
        data_path = '/data/Public/Datasets/cifar10'
        # Data loading code
        trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=transform_test)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, **kwargs)
        valset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=transform_test)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=True, **kwargs)

        num_classes = 10

    elif args.in_dataset == "CIFAR-100":
        data_path = '/data/Public/Datasets/cifar100'
        # Data loading code
        trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False, transform=transform_test)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, **kwargs)
        valset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=transform_test)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=False, **kwargs)

        num_classes = 100

    elif args.in_dataset == "imagenet100":
        root = '/data/Public/Datasets/ImageNet100'
        # Data loading code
        trainset = torchvision.datasets.ImageFolder(os.path.join(root, 'train'), transform_test_largescale)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, **kwargs)
        
        num_classes = 100

       
    elif args.in_dataset == "imagenet":
        root = '/data/Public/Datasets/ilsvrc2012'
        # Data loading code
        trainset = torchvision.datasets.ImageFolder(os.path.join(root, 'train'), transform_test_largescale)
        train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=args.batch, shuffle=False, **kwargs)
        valset = torchvision.datasets.ImageFolder(os.path.join(root, 'val'), transform_test_largescale)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=True, **kwargs)
        num_classes = 1000
    
    elif args.in_dataset == "Places":
        # Data loading code
        valset = torchvision.datasets.ImageFolder("/data/Public/Datasets/Places", transform=transform_test_largescale)
        val_dataloader = torch.utils.data.DataLoader(valset, batch_size=args.batch, shuffle=True, **kwargs)
        num_classes = 1000
        args.in_dataset = 'imagenet'

    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    # print(trainset.imgs[0][0])
    # checkpoint = torch.load(
    #         f'{args.model_path}/{args.name}/{args.in_dataset}/{args.model}_parameter.pth.tar')

    # model.load_state_dict(checkpoint['state_dict'])
    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    model.eval()
    
    fc_w = extact_mean_std(args, model)
    mask, class_mean = get_class_mean2(args, fc_w)
    # mask, class_mean = get_class_mean(args)
    class_mean = class_mean.cuda()
    class_mean = class_mean.clip(max=args.threshold)
    # show_gradcam(args, model, train_dataloader, trainset, mask, fc_w)
    # args.in_dataset = "Places"
    show_gradcam(args, model, val_dataloader, valset, mask, fc_w)

if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()
    if args.in_dataset == "CIFAR-10":
        args.threshold = 1.5
        args.p_a = 90
        args.p_w = 90

    elif args.in_dataset == "CIFAR-100":
        args.p_a = 10
        args.p_w = 90
        args.threshold = 1.5

    elif args.in_dataset == "imagenet":
        args.threshold = 1.0
        args.p_a = 10
        args.p_w = 10
    # args.threshold = 1e5
    # main(args)
    test_gradcam(args)
    # test_train(args)
    # test_mask(args)
