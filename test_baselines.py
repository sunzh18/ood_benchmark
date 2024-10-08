from utils import log
from models.resnetv2 import * 
import torch
import torch.nn as nn
import time
import csv
import numpy as np

from utils.test_utils import mk_id_ood, get_measures
import os

from sklearn.linear_model import LogisticRegressionCV
from torch.autograd import Variable
from utils.mahalanobis_lib import get_Mahalanobis_score
from utils.data_loader import get_dataloader_in, get_dataloader_out, cifar_out_datasets, imagenet_out_datasets
from utils.model_loader import get_model

from utils.cal_score import *
from argparser import *

def run_eval(model, in_loader, out_loader, logger, args, num_classes, out_dataset):
    # switch to evaluate mode
    model.eval()

    logger.info("Running test...")
    logger.flush()

    if args.score == 'MSP':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_msp(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_msp(out_loader, model)
    elif args.score == 'ODIN':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_odin(in_loader, model, args.epsilon_odin, args.temperature_odin, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_odin(out_loader, model, args.epsilon_odin, args.temperature_odin, logger)
    elif args.score == 'Energy':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy)
    elif args.score == 'dice':
        if args.in_dataset == "CIFAR-10":
            args.threshold = 1.0
            # args.p = 10
            args.p = 90

        elif args.in_dataset == "CIFAR-100":
            args.threshold = 1.0
            # args.p = 30
            args.p = 90

        info = np.load(f"checkpoints/feature/{args.name}/{args.in_dataset}/{args.model}_feat_stat.npy")
        model = get_model(args, num_classes, load_ckpt=True, info=info)
        model.eval()
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_energy(in_loader, model, args.temperature_energy)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_energy(out_loader, model, args.temperature_energy)
    elif args.score == 'dice_react':
        if args.in_dataset == "CIFAR-10":
            # args.threshold = 0.8
            # args.p = 10
            args.threshold = 1.0
            args.p = 90
        elif args.in_dataset == "CIFAR-100":
            # args.threshold = 1.0
            # args.p = 30
            args.threshold = 1.0
            args.p = 90

        # args.threshold = 1.0
        info = np.load(f"checkpoints/feature/{args.name}/{args.in_dataset}/{args.model}_feat_stat.npy")
        model = get_model(args, num_classes, load_ckpt=True, info=info)
        model.eval()
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_react(in_loader, model, args.temperature_energy, args.threshold)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_react(out_loader, model, args.temperature_energy, args.threshold)
    elif args.score == 'react':
        args.threshold = 1.0      
        # args.threshold = 0.7    #resnet18
        if args.in_dataset == "CIFAR-10":
            # args.threshold = 0.8
            # args.p = 10
            args.threshold = 1.5
        elif args.in_dataset == "CIFAR-100":
            # args.threshold = 1.0
            # args.p = 30
            args.threshold = 2.5
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_react(in_loader, model, args.temperature_energy, args.threshold)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_react(out_loader, model, args.temperature_energy, args.threshold)
    elif args.score == 'LINE':
        if args.in_dataset == "CIFAR-10":
            args.threshold = 1.0
            args.p_a = 90
            args.p_w = 90

        elif args.in_dataset == "CIFAR-100":
            args.threshold = 1.0
            args.p_a = 10
            args.p_w = 90
                
        elif args.in_dataset == "imagenet":
            args.threshold = 0.8
            args.p_a = 10
            args.p_w = 10
        # args.p_a = 90
        # args.p_w = 90
        # args.threshold = 1.0  #0.8
        info = np.load(f"cache/{args.name}/{args.in_dataset}_{args.model}_meanshap_class.npy")
        model = get_model(args, num_classes, load_ckpt=True, info=info, LU=True)
        model.eval()
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_LINE(in_loader, model, args.temperature_energy, args.threshold)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_LINE(out_loader, model, args.temperature_energy, args.threshold)

    elif args.score == 'bats':
        bats = 1
        feature_std=torch.load(f"checkpoints/feature/{args.name}/{args.in_dataset}/{args.model}_features_std.pt").cuda()
        feature_mean=torch.load(f"checkpoints/feature/{args.name}/{args.in_dataset}/{args.model}_features_mean.pt").cuda()
        if args.in_dataset == 'imagenet':       
            lam = 1.25

        elif args.in_dataset == 'CIFAR-10':
            if args.model == 'wrn':
                lam = 2.25
            elif args.model == 'resnet18':
                lam = 3.3
            elif args.model == 'densenet':
                lam = 1.0

        elif args.in_dataset == 'CIFAR-100':
            if args.model == 'wrn':
                lam = 1.5
            elif args.model == 'resnet18':
                lam = 1.35
            elif args.model == 'densenet':
                lam = 0.8
        # print(feature_std.shape)
        args.bats = lam
        logger.info("Processing in-distribution data...")
        in_scores = bats_iterate_data_energy(in_loader, model, args.temperature_energy, lam, feature_std, feature_mean, bats)
        logger.info("Processing out-of-distribution data...")
        out_scores = bats_iterate_data_energy(out_loader, model, args.temperature_energy, lam, feature_std, feature_mean, bats)

    elif args.score == 'GradNorm':
        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_gradnorm(in_loader, model, args.temperature_gradnorm, num_classes)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_gradnorm(out_loader, model, args.temperature_gradnorm, num_classes)
    elif args.score == 'Mahalanobis':
        save_dir = os.path.join('cache', 'mahalanobis', args.name)
        lr_weights, lr_bias, magnitude = np.load(
            os.path.join(save_dir, f'{args.in_dataset}_{args.model}_results.npy'), allow_pickle=True)
        

        # regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0, 0], [0, 0, 0, 0, 0], [1, 1, 1, 1, 1], [1, 1, 1, 1, 1]],
        #                                            [0, 0, 1, 1])
        regressor = LogisticRegressionCV(cv=2).fit([[0, 0, 0, 0], [0, 0, 0, 0], [1, 1, 1, 1], [1, 1, 1, 1]],
                                                   [0, 0, 1, 1])
        # regressor = LogisticRegressionCV(cv=2).fit([[0], [0], [1], [1]],
                                                #    [0, 0, 1, 1])

        regressor.coef_ = lr_weights
        regressor.intercept_ = lr_bias

        temp_x = torch.rand(2, 3, 32, 32)
        temp_x = Variable(temp_x).cuda()
        # temp_list = model(x=temp_x, layer_index='all')[1]
        temp_list = model.feature_list(temp_x)[1]
        num_output = len(temp_list)

        # file_folder = f'checkpoints/feature/{args.name}/{args.in_dataset}'
        file_folder = os.path.join('cache', 'mahalanobis', args.name)
        filename1 = os.path.join(file_folder, f'{args.in_dataset}_{args.model}_class_mean.npy')
        filename2 = os.path.join(file_folder, f'{args.in_dataset}_{args.model}_precision.npy')
        sample_mean = np.load(filename1, allow_pickle=True)
        precision = np.load(filename2, allow_pickle=True)
        # sample_mean = [torch.from_numpy(s).cuda() for s in sample_mean]
        sample_mean = [s.cuda() for s in sample_mean]

        precision = [torch.from_numpy(p).float().cuda() for p in precision]
        # class_mean = np.load(f"{file_folder}/{args.model}_class_mean.npy")
        # precision = np.load(f"{file_folder}/{args.model}_precision.npy")

        # class_mean = torch.from_numpy(class_mean).cuda().float()
        # precision = torch.from_numpy(precision).cuda().float()


        logger.info("Processing in-distribution data...")
        in_scores = iterate_data_mahalanobis(in_loader, model, num_classes, sample_mean, precision,
                                             num_output, magnitude, regressor, logger)
        logger.info("Processing out-of-distribution data...")
        out_scores = iterate_data_mahalanobis(out_loader, model, num_classes, sample_mean, precision,
                                              num_output, magnitude, regressor, logger)
    elif args.score == 'KL_Div':
        logger.info("Processing in-distribution data...")
        in_dist_logits, in_labels = iterate_data_kl_div(in_loader, model)
        logger.info("Processing out-of-distribution data...")
        out_dist_logits, _ = iterate_data_kl_div(out_loader, model)

        class_mean_logits = []
        for c in range(num_classes):
            selected_idx = (in_labels == c)
            selected_logits = in_dist_logits[selected_idx]
            class_mean_logits.append(np.mean(selected_logits, axis=0))
        class_mean_logits = np.array(class_mean_logits)

        logger.info("Compute distance for in-distribution data...")
        in_scores = []
        for i, logit in enumerate(in_dist_logits):
            if i % 100 == 0:
                logger.info('{} samples processed...'.format(i))
            min_div = float('inf')
            for c_mean in class_mean_logits:
                cur_div = kl(logit, c_mean)
                if cur_div < min_div:
                    min_div = cur_div
            in_scores.append(-min_div)
        in_scores = np.array(in_scores)

        logger.info("Compute distance for out-of-distribution data...")
        out_scores = []
        for i, logit in enumerate(out_dist_logits):
            if i % 100 == 0:
                logger.info('{} samples processed...'.format(i))
            min_div = float('inf')
            for c_mean in class_mean_logits:
                cur_div = kl(logit, c_mean)
                if cur_div < min_div:
                    min_div = cur_div
            out_scores.append(-min_div)
        out_scores = np.array(out_scores)
    else:
        raise ValueError("Unknown score type {}".format(args.score))

    in_examples = in_scores.reshape((-1, 1))
    out_examples = out_scores.reshape((-1, 1))

    auroc, aupr_in, aupr_out, fpr95 = get_measures(in_examples, out_examples)
    auroc, aupr_in, aupr_out, fpr95 = auroc*100, aupr_in*100, aupr_out*100, fpr95*100
    
    # result_path = os.path.join(args.logdir, args.name, args.model, f"{args.in_dataset}_{args.score}.csv")
    # fp = open(result_path,'a+')
    # result = []
    # if args.arch:
    #     result.append(f'{args.name}/{args.in_dataset}/{args.model}_{args.arch}')
    # else:
    #     # result.append(f'{args.name}/{args.in_dataset}/{args.model}')
    #     result.append(f'threshold{args.threshold}/p{args.p}/bat{args.bats}')
    # result.append(out_dataset)
    # result.append("{:.4f}".format(auroc))
    # result.append("{:.4f}".format(aupr_in))
    # result.append("{:.4f}".format(aupr_out))
    # result.append("{:.4f}".format(fpr95))
    # context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
    # context.writerow(result)
    # fp.close()

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

    # in_set, out_set, in_loader, out_loader = mk_id_ood(args, logger)

#     logger.info(f"Loading model from {args.model_path}")
#     model = KNOWN_MODELS[args.model](head_size=len(in_set.classes))

#     state_dict = torch.load(args.model_path)
#     model.load_state_dict_custom(state_dict['model'])

#     model = torch.nn.DataParallel(model)
#     model = model.cuda()

#     start_time = time.time()
#     run_eval(model, in_loader, out_loader, logger, args, num_classes=len(in_set.classes))
#     end_time = time.time()

#     logger.info("Total running time: {}".format(end_time - start_time))


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

    result_path = os.path.join(args.logdir, args.name, args.model, f"{args.in_dataset}.txt")
    if not os.path.exists(result_path):
        with open(result_path, 'a+', encoding='utf-8') as f:
            f.write('method  ')
            f.write('FPR95  ')
            f.write('AUROC\n')

    in_dataset = args.in_dataset

    in_save_dir = os.path.join(args.logdir, args.name, args.model)
    if not os.path.exists(in_save_dir):
        os.makedirs(in_save_dir)

    loader_in_dict = get_dataloader_in(args, split=('val'))
    in_loader, num_classes = loader_in_dict.val_loader, loader_in_dict.num_classes
    args.num_classes = num_classes

    load_ckpt = False
    if args.model_path != None:
        load_ckpt = True

    model = get_model(args, num_classes, load_ckpt=load_ckpt)
    
    model.eval()

    if args.score == 'react':
        find_threshold(args, model, in_loader)

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

        result_path = os.path.join(args.logdir, args.name, args.model, f"{args.in_dataset}_{args.score}.csv")
        fp = open(result_path,'a+')
        result = []
        if args.arch:
            result.append(f'{args.name}/{args.in_dataset}/{args.model}_{args.arch}')
        else:
            # result.append(f'{args.name}/{args.in_dataset}/{args.model}')
            result.append(f'threshold{args.threshold}/p{args.p}/lam{args.bats}')
        result.append('Average')
        result.append("{:.4f}".format(avg_auroc))
        result.append("{:.4f}".format(avg_aupr_in))
        result.append("{:.4f}".format(avg_aupr_out))
        result.append("{:.4f}".format(avg_fpr95))
        context = csv.writer(fp,dialect='excel')       # 定义一个变量进行写入，将刚才的文件变量传进来，dialect就是定义一下文件的类型，我们定义为excel类型
        context.writerow(result)
        fp.close()
        
        result_path = os.path.join(args.logdir, args.name, args.model, f"{args.in_dataset}.txt")
        with open(result_path, 'a+', encoding='utf-8') as f:
            f.write("{} & ".format(args.score))
            for i in range(len(AUroc)):
                fpr95 = Fpr95[i]
                auroc = AUroc[i]
                f.write("{:.2f} & ".format(fpr95))
                f.write("{:.2f} & ".format(auroc))
            f.write("{:.2f} & ".format(avg_fpr95))
            f.write("{:.2f}\n".format(avg_auroc))

        logger.info('============Results for {}============'.format(args.score))
        logger.info('=======in dataset: {}; ood dataset: Average============'.format(args.in_dataset))
        logger.info('Average AUROC: {}'.format(avg_auroc))
        logger.info('Average AUPR (In): {}'.format(avg_aupr_in))
        logger.info('Average AUPR (Out): {}'.format(avg_aupr_out))
        logger.info('Average FPR95: {}'.format(avg_fpr95))


def get_features(args, model, dataloader):
    features = []
    for b, (x, y) in enumerate(dataloader):
        with torch.no_grad():
            x = x.cuda()            
            # print(x.size())
            feature = model.forward_features(x)
            # print(feature.size())
            features.extend(feature.data.cpu().numpy())
            # x = feature[feature>=0]
            # print(x.size())

    features = np.array(features)
    # x = np.transpose(features)
    # print(features.shape, features)

    return features

def find_threshold(args, model, dataloader):
    features = get_features(args, model, dataloader)
    # print(features.flatten().shape)
    print(f"\nTHRESHOLD at percentile {90} is:")
    threshold = np.percentile(features.flatten(), 90)
    print(threshold)
    args.threshold = threshold
    return 






if __name__ == "__main__":
    parser = get_argparser()

    args = parser.parse_args()

    
    main(args)
