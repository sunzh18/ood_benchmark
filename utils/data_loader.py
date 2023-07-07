import os
import torch
import torchvision
from torchvision import transforms
from easydict import EasyDict
from datasets.svhn_loader import SVHN

cifar_out_datasets = ['SVHN', 'LSUN_C', 'LSUN_R', 'iSUN', 'Textures', 'Places']
# 'Tinyimagenet'
imagenet_out_datasets = ['Textures', 'Places', 'SUN', 'iNat']


imagesize = 32

transform_test = transforms.Compose([
    transforms.Resize((imagesize, imagesize)),
    transforms.CenterCrop(imagesize),
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train = transforms.Compose([
    transforms.RandomCrop(imagesize, padding=4),
    transforms.RandomHorizontalFlip(),
    # transforms.RandomResizedCrop(imagesize),
    
    transforms.ToTensor(),
    transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
])

transform_train_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.RandomResizedCrop(224),
    transforms.RandomHorizontalFlip(),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

transform_test_largescale = transforms.Compose([
    transforms.Resize(256),
    transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

val_tx = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    ])




kwargs = {'num_workers': 2, 'pin_memory': True}

def get_dataloader_in(args, config_type='default', split=('train', 'val')):
    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'batch_size': args.batch,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
        },
    })[config_type]

    train_loader, val_loader, trainset, valset, lr_schedule, num_classes, = None, None, None, None, [50, 75, 90], 0, 
    if args.in_dataset == "CIFAR-10":
        data_path = '/data/Public/Datasets/cifar10'
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR10(root=data_path, train=True, download=False, transform=config.transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR10(root=data_path, train=False, download=False, transform=config.transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 10

    elif args.in_dataset == "CIFAR-100":
        data_path = '/data/Public/Datasets/cifar100'
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.CIFAR100(root=data_path, train=True, download=False, transform=config.transform_train)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.CIFAR100(root=data_path, train=False, download=False, transform=config.transform_test)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 100

    elif args.in_dataset == "imagenet100":
        root = '/data/Public/Datasets/ImageNet100'
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.ImageFolder(os.path.join(root, 'train'), config.transform_train_largescale)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 1000

       
    elif args.in_dataset == "imagenet":
        root = '/data/Public/Datasets/ilsvrc2012'
        # Data loading code
        if 'train' in split:
            trainset = torchvision.datasets.ImageFolder(os.path.join(root, 'train'), config.transform_train_largescale)
            train_loader = torch.utils.data.DataLoader(trainset, batch_size=config.batch_size, shuffle=True, **kwargs)
        if 'val' in split:
            valset = torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale)
            val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 1000
    
    return EasyDict({
        "train_loader": train_loader,
        "val_loader": val_loader,
        "lr_schedule": lr_schedule,
        "num_classes": num_classes,
        "train_dataset": trainset,
        "val_dataset": valset
    })

def get_dataloader_out(args, dataset=(''), config_type='default', split=('val')):

    config = EasyDict({
        "default": {
            'transform_train': transform_train,
            'transform_test': transform_test,
            'transform_test_largescale': transform_test_largescale,
            'transform_train_largescale': transform_train_largescale,
            'batch_size': args.batch
        },
    })[config_type]
    train_ood_loader, val_ood_loader, trainset, valset, = None, None, None, None

    # if 'train' in split:
    #     if dataset[0].lower() == 'imagenet':
            
    #         train_ood_loader = torch.utils.data.DataLoader(
    #             ImageNet(transform=config.transform_train),
    #             batch_size=config.batch_size, shuffle=True, **kwargs)
    #     elif dataset[0].lower() == 'tim':
    #         train_ood_loader = torch.utils.data.DataLoader(
    #             TinyImages(transform=config.transform_train),
    #             batch_size=config.batch_size, shuffle=True, **kwargs)

    if 'val' in split:
        val_dataset = dataset[1]
        batch_size = args.batch
        if val_dataset == 'SVHN':       #cifar
            valset = SVHN('/data/Public/Datasets/SVHN/', split='test', transform=transform_test, download=False)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False,
                                                        num_workers=2)
            
        elif val_dataset == 'Textures':     #imagenet, cifar
            val_transform = config.transform_test_largescale if args.in_dataset in {'imagenet'} else config.transform_test
            valset = torchvision.datasets.ImageFolder(root="/data/Public/Datasets/dtd/images", transform=val_transform)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=True, num_workers=2)
        
        
        elif val_dataset == 'Places':   # imagenet, cifar
            val_transform = config.transform_test_largescale if args.in_dataset in {'imagenet'} else config.transform_test
            if args.in_dataset == 'imagenet':
                valset = torchvision.datasets.ImageFolder("/data/Public/Datasets/Places",
                                                        transform=val_transform)
            elif args.in_dataset in {'CIFAR-10', 'CIFAR-100'}:
                valset = torchvision.datasets.ImageFolder("/data/Public/Datasets/places365/test_subset",
                                                        transform=val_transform)
                # l = len(valset)
                # valset, _ = torch.utils.data.random_split(valset, [10000, l-10000])
            
            
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        elif val_dataset == 'SUN':      # imagenet
            valset = torchvision.datasets.ImageFolder("/data/Public/Datasets/SUN",
                                                        transform=config.transform_test_largescale)
            val_ood_loader = torch.utils.data.DataLoader(valset , batch_size=batch_size, shuffle=False, num_workers=2)
            
        elif val_dataset == 'iNat':     # imagenet
            valset = torchvision.datasets.ImageFolder("/data/Public/Datasets/iNaturalist",
                                                        transform=config.transform_test_largescale)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
            
        elif val_dataset == 'Tinyimagenet':     # cifar
            valset = torchvision.datasets.ImageFolder("/data/Public/Datasets/tiny-imagenet-200/val/",
                                                 transform=transform_test)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
            
        # elif val_dataset == 'imagenet':
        #     val_ood_loader = torch.utils.data.DataLoader(
        #         torchvision.datasets.ImageFolder(os.path.join('./datasets/id_data/imagenet', 'val'), config.transform_test_largescale),
        #         batch_size=config.batch_size, shuffle=True, **kwargs)

        # elif val_dataset == 'CIFAR-100':
        #     val_ood_loader = torch.utils.data.DataLoader(torchvision.datasets.CIFAR100(root='./data', train=False, download=True, transform=transform_test),
        #                                                batch_size=batch_size, shuffle=True, num_workers=2)
            
        else:       #cifar - LSUN-C, LSUN-R, iSUN
            valset = torchvision.datasets.ImageFolder("/data/Public/Datasets/{}".format(val_dataset),
                                                        transform=transform_test)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)

    return EasyDict({
        "train_ood_loader": train_ood_loader,
        "val_ood_loader": val_ood_loader,
        "train_dataset": trainset,
        "val_dataset": valset
    })