import torchvision.transforms as trn
import torchvision.datasets as dset
import datasets.svhn_loader as svhn


def build_dataset(dataset, mode="train"):

    mean = (0.492, 0.482, 0.446)
    std = (0.247, 0.244, 0.262)

    train_transform = trn.Compose([trn.RandomHorizontalFlip(), trn.RandomCrop(32, padding=4),
                                       trn.ToTensor(), trn.Normalize(mean, std)])
    test_transform = trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)])

    if dataset == 'cifar10':
        if mode == "train":
            data = dset.CIFAR10(root='/data/Public/Datasets/cifar10',
                                    download=True,
                                    train=True,
                                    transform=train_transform
                                    )
        else:
            data = dset.CIFAR10(root='/data/Public/Datasets/cifar10',
                                   download=True,
                                   train=False,
                                   transform=test_transform
                                   )
        num_classes = 10
    elif dataset == 'cifar100':
        if mode == "train":
            data = dset.CIFAR100(root='/data/Public/Datasets/cifar100',
                                     download=True,
                                     train=True,
                                     transform=train_transform
                                     )
        else:
            data = dset.CIFAR100(root='/data/Public/Datasets/cifar100',
                                    download=True,
                                    train=False,
                                    transform=test_transform
                                    )
        num_classes = 100

    elif dataset == 'imagenet100':
        mean = (0.485, 0.456, 0.406)
        std = (0.229, 0.224, 0.225)
        train_transform_imgnet = trn.Compose([trn.Resize((256, 256)), trn.RandomHorizontalFlip(), trn.RandomCrop((224, 224)),
                                       trn.ToTensor(), trn.Normalize(mean, std)])
        test_transform_imgnet = trn.Compose([trn.Resize((224, 224)), trn.ToTensor(), trn.Normalize(mean, std)])
        # test_transform_imgnet = trn.Compose([trn.CenterCrop(224), trn.ToTensor(), trn.Normalize(mean, std)])
        train_transform_imgnet = trn.Compose([
                                        trn.RandomResizedCrop((224, 224)),
                                        trn.RandomHorizontalFlip(),
                                        # trn.RandomCrop((224, 224)),
                                        trn.ToTensor(),
                                        trn.Normalize(mean, std)])
        test_transform_imgnet = trn.Compose([
                                        trn.Resize((256, 256)),
                                        trn.CenterCrop((224, 224)),
                                        trn.ToTensor(),
                                        trn.Normalize(mean, std)])
        if mode == "train":
            data = dset.ImageFolder(root='/data/Public/Datasets/ImageNet100/train',
                                     transform=train_transform_imgnet
                                     )
        else:
            data = dset.ImageFolder(root='/data/Public/Datasets/ImageNet100/val',
                                    transform=test_transform_imgnet
                                    )
        num_classes = 100
    elif dataset == "Textures":
        data = dset.ImageFolder(root="/data/Public/Datasets/dtd/images",
                                    transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                           trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "SVHN":
        if mode == "train":
            data = svhn.SVHN(root='/data/Public/Datasets/SVHN/', split="train",
                             transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]),
                             download=False)
        else:
            data = svhn.SVHN(root='/data/Public/Datasets/SVHN/', split="test",
                             transform=trn.Compose([trn.Resize(32), trn.ToTensor(), trn.Normalize(mean, std)]),
                             download=True)
        num_classes = 10

    elif dataset == "Places365":
        data = dset.ImageFolder(root="/data/Public/Datasets/Places",
                                transform=trn.Compose([trn.Resize(32), trn.CenterCrop(32),
                                                       trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "LSUN-C":
        data = dset.ImageFolder(root="/data/Public/Datasets/LSUN_C/",
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "LSUN-R":
        data = dset.ImageFolder(root="/data/Public/Datasets/LSUN_R/",
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    elif dataset == "iSUN":
        data = dset.ImageFolder(root="/data/Public/Datasets/iSUN/",
                                    transform=trn.Compose([trn.ToTensor(), trn.Normalize(mean, std)]))
        num_classes = 10
    return data, num_classes



