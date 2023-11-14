import os
import torch
import torchvision
from torchvision import transforms
from easydict import EasyDict
from datasets.svhn_loader import SVHN
from datasets.dataset_puzzle import *
from PIL import Image
cifar_out_datasets = ['SVHN', 'LSUN_C', 'LSUN_R', 'iSUN', 'Textures', 'Places']
# 'Tinyimagenet'
imagenet_out_datasets = ['iNat', 'SUN', 'Places', 'Textures']


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
            valset = torchvision.datasets.ImageFolder(root="/data/Public/Datasets/dtd/dtd/images", transform=val_transform)
            val_ood_loader = torch.utils.data.DataLoader(valset, batch_size=batch_size, shuffle=False, num_workers=2)
        
        
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




def generate_gaussian_noise_image(size, mean, std):
    noise = np.random.normal(0, 1, size)
    # noise = (noise * std) + mean
    noise = np.clip(noise, 0, 1).astype(np.float32)
    # noise = np.clip(noise, 0, 255).astype(np.uint8)
    noise_image = Image.fromarray(noise, mode='RGB')
    # noise_image = torch.tensor(noise).float()
    return noise_image

# 转换图像为 PyTorch 的 tensor 格式
def image_to_tensor(image):
    transform = torchvision.transforms.ToTensor()
    tensor_image = transform(image)
    return tensor_image

# 将图像添加到 torchvision.datasets.Dataset 类中
class GaussianNoiseDataset(torchvision.datasets.VisionDataset):
    def __init__(self, size, mean, std, num_images, transform=None, target_transform=None):
        super(GaussianNoiseDataset, self).__init__(None, transform=transform, target_transform=target_transform)
        self.size = size
        self.mean = mean
        self.std = std
        self.num_images = num_images

    def __getitem__(self, index):
        noise_image = generate_gaussian_noise_image(self.size, self.mean, self.std)
        if self.transform:
            noise_image = self.transform(noise_image)
        return noise_image, 0  # 第二个参数为目标标签，这里设置为0

    def __len__(self):
        return self.num_images

# 设置高斯噪声的参数，和 CIFAR-10 数据集一致
width, height = 32, 32  # CIFAR-10 图像大小为 32x32
mean, std = 128, 60  # CIFAR-10 数据集的均值和标准差
num_images = 10000  # 生成 10000 张高斯噪声图像


# 创建高斯噪声数据集

transform_puzzle_largescale = transforms.Compose([
    transforms.Resize((224, 224)),
    # transforms.CenterCrop(224),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                         std=[0.229, 0.224, 0.225]),
])

def get_dataloader_noise(args, config_type='default'):
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
        valset = GaussianNoiseDataset((3, width, height), mean, std, num_images, transform_test)

        val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)

        # valset = Puzzle_CIFAR10(root=data_path, train=False, download=False, transform=config.transform_test)
        # valset = torchvision.datasets.CIFAR100(root='/data/Public/Datasets/cifar100', train=False, download=False, transform=config.transform_train)
        # val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 10

    elif args.in_dataset == "CIFAR-100":
        data_path = '/data/Public/Datasets/cifar100'
        # Data loading code
        valset = GaussianNoiseDataset((3, width, height), mean, std, num_images, transform_test)

        val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)

        # valset = Puzzle_CIFAR100(root=data_path, train=False, download=False, transform=config.transform_test)
        # valset = torchvision.datasets.CIFAR10(root='/data/Public/Datasets/cifar10', train=False, download=False, transform=config.transform_train)
        # val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 100

    elif args.in_dataset == "imagenet100":
        root = '/data/Public/Datasets/ImageNet100'
        # Data loading code
        valset = GaussianNoiseDataset((3, 224, 224), mean, std, num_images)

        val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)

        # valset = torchvision.datasets.ImageFolder(os.path.join(root, 'val'), config.transform_test_largescale)
        # val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
        num_classes = 1000

       
    elif args.in_dataset == "imagenet":
        root = '/data/Public/Datasets/ilsvrc2012/'

        path = 'puzzle_data/ImageNet'

        # for file_dir in os.listdir(root):
        #     root_path = os.path.join(root, file_dir)
        #     save_path = os.path.join(path, file_dir)
        #     if not os.path.exists(save_path):
        #         os.mkdir(save_path)
        #     for file_name in os.listdir(root_path):
        #         save_file_name = os.path.join(save_path, file_name)
        #         file_name = os.path.join(root_path, file_name)
                
        #         img = Image.open(file_name) 
        #         tiles = split_image(img)
        #         shuffled_tiles = shuffle_tiles(tiles)
        #         new_img = recompose_image(shuffled_tiles)

        #         new_img.save(save_file_name)


        # Data loading code
        valset = GaussianNoiseDataset((3, 224, 224), mean, std, num_images, transform_test_largescale)

        val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
        # valset = torchvision.datasets.ImageFolder(path, config.transform_test_largescale)
        # l = len(valset)
        # valset, _ = torch.utils.data.random_split(valset, [10000, l-10000])
        # val_loader = torch.utils.data.DataLoader(valset, batch_size=config.batch_size, shuffle=True, **kwargs)
        # num_classes = 1000
    
    return EasyDict({
        "val_ood_loader": val_loader,
        "lr_schedule": lr_schedule,
        "num_classes": num_classes,
        "val_dataset": valset
    })