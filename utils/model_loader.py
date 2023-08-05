import os

from models.resnet import *
from models.cifar_resnet import resnet18_cifar, resnet50_cifar
from models.resnet_react import resnet18, resnet50
from models.mobilenet import mobilenet_v2
from models.wideresnet import WideResNet28
from models.densenet import DenseNet3
import torch
# from timm.models import create_model


train_on_gpu = torch.cuda.is_available() 


def get_model(args, num_classes, load_ckpt=True, info=None, LU=False):
    if args.in_dataset == 'imagenet':
        if args.model == 'resnet18':
            model = resnet18(num_classes=num_classes, pretrained=True, p=args.p, p_w=args.p_w, p_a=args.p_a, info=info, LU = LU, clip_threshold = args.threshold)
        elif args.model == 'resnet50':
            model = resnet50(num_classes=num_classes, pretrained=True, p=args.p, p_w=args.p_w, p_a=args.p_a, info=info, LU = LU, clip_threshold = args.threshold)
        elif args.model == 'mobilenet':
            model = mobilenet_v2(num_classes=num_classes, pretrained=False, p=args.p, p_w=args.p_w, p_a=args.p_a, info=info, LU = LU, clip_threshold = args.threshold)
            checkpoint = torch.load("/data/Public/PretrainedModels/mobilenet_v2-b0353104.pth")

        elif args.model == 'vit':
            # model = create_model("vit_base_patch16_384",pretrained=False,num_classes=num_classes)

    else:
        if args.model == 'resnet18':
            model = resnet18_cifar(num_classes=num_classes, p=args.p, p_w=args.p_w, p_a=args.p_a, info=info, LU = LU, clip_threshold = args.threshold)
        elif args.model == 'resnet50':
            model = resnet50_cifar(num_classes=num_classes, p=args.p, p_w=args.p_w, p_a=args.p_a, info=info, LU = LU, clip_threshold = args.threshold)
        elif args.model == 'wrn':
            model = WideResNet28(num_classes=num_classes, p=args.p, p_w=args.p_w, p_a=args.p_a, info=info, LU = LU, clip_threshold = args.threshold)
        elif args.model == 'mobilenet':
            model = mobilenet_v2(num_classes=num_classes)
        elif args.model == 'densenet':
            model = DenseNet3(100, num_classes, 12, reduction=0.5, bottleneck=True, dropRate=0.0, normalizer=None, p=args.p, p_w=args.p_w, p_a=args.p_a, info=info, LU = LU, clip_threshold = args.threshold)
        else:
            assert False, 'Not supported model arch: {}'.format(args.model)
    device = torch.device("cuda") 
    if train_on_gpu:                                                   #部署到GPU上
        device = torch.device("cuda") 
        # print('cuda')
    else:
        device = torch.device("cpu")
        # print('cpu')
    model = model.to(device) 

    # model = nn.DataParallel(model)

    if (args.in_dataset != 'imagenet') and load_ckpt:
        # checkpoint = torch.load("./checkpoints/{in_dataset}/{model}/checkpoint_{epochs}.pth.tar".format(in_dataset=args.in_dataset, model=args.model, epochs=args.epochs))
        # checkpoint = torch.load(args.model_path)
        if args.arch:
            checkpoint = torch.load(f'{args.model_path}/{args.name}/{args.in_dataset}/{args.model}_{args.arch}_parameter.pth')
        else:
            checkpoint = torch.load(f'{args.model_path}/{args.name}/{args.in_dataset}/{args.model}_parameter.pth')
        model.load_state_dict(checkpoint['state_dict'])
        # model.load_state_dict(torch.load(args.model_path))

    # get the number of model parameters
    print('Number of model parameters: {}'.format(
        sum([p.data.nelement() for p in model.parameters()])))
    return model