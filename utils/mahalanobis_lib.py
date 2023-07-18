from __future__ import print_function
import torch
import time
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from scipy.spatial.distance import pdist, cdist, squareform
from tqdm import tqdm
import os
from sklearn.linear_model import LogisticRegressionCV
from utils.test_utils import mk_id_ood, get_measures

def sample_estimator(model, num_classes, feature_list, train_loader):
    """
    compute sample mean and precision (inverse of covariance)
    return: sample_class_mean: list of class mean
             precision: list of precisions
    """
    import sklearn.covariance

    model.eval()
    group_lasso = sklearn.covariance.EmpiricalCovariance(assume_centered=False)
    correct, total = 0, 0
    num_output = len(feature_list)
    num_sample_per_class = np.empty(num_classes)
    num_sample_per_class.fill(0)
    list_features = []
    for i in range(num_output):
        temp_list = []
        for j in range(num_classes):
            temp_list.append(0)
        list_features.append(temp_list)

    for data, target in tqdm(train_loader):
        total += data.size(0)
        # print(total)
        # if total > 50000:
        #     break
        # data = data.cuda()
        data = Variable(data)
        data = data.cuda()
        output, out_features = model.feature_list(data)

        # get hidden features
        for i in range(num_output):
            out_features[i] = out_features[i].view(out_features[i].size(0), out_features[i].size(1), -1)
            # print(out_features[i].shape)
            out_features[i] = torch.mean(out_features[i].data, 2)
            # out_features[i] = out_features[i].view(out_features[i].size(0), -1)
            # print(out_features[i].shape)

        # compute the accuracy
        pred = output.data.max(1)[1]
        equal_flag = pred.eq(target.cuda()).cpu()
        correct += equal_flag.sum()

        # construct the sample matrix
        for i in range(data.size(0)):
            label = target[i]
            if num_sample_per_class[label] == 0:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] = out[i].view(1, -1)
                    out_count += 1
            else:
                out_count = 0
                for out in out_features:
                    list_features[out_count][label] \
                    = torch.cat((list_features[out_count][label], out[i].view(1, -1)), 0)
                    out_count += 1
            num_sample_per_class[label] += 1

    sample_class_mean = []
    out_count = 0
    for num_feature in feature_list:
        # temp_list = torch.Tensor(num_classes, int(num_feature)).cuda()
        temp_list = torch.Tensor(num_classes, int(num_feature)).cpu()

        for j in range(num_classes):
            # list_features[out_count][j] = list_features[out_count][j].clip(max=1.0)
            temp_list[j] = torch.mean(list_features[out_count][j], 0)
        sample_class_mean.append(temp_list)
        out_count += 1

    precision = []
    for k in range(num_output):
        X = 0
        for i in range(num_classes):
            if i == 0:
                X = list_features[k][i] - sample_class_mean[k][i].cuda()
            else:
                X = torch.cat((X, list_features[k][i] - sample_class_mean[k][i].cuda()), 0)

        # find inverse
        group_lasso.fit(X.cpu().numpy())
        temp_precision = group_lasso.precision_
        # temp_precision = torch.from_numpy(temp_precision).double().cuda()
        precision.append(temp_precision)

    # for i in range(len(sample_class_mean)):
    #     sample_class_mean[i] = sample_class_mean[i].cpu().numpy()

    print('\n Training Accuracy:({:.2f}%)\n'.format(100. * correct / total))

    return sample_class_mean, precision

def get_Mahalanobis_score(inputs, model, num_classes, sample_mean, precision, num_output, magnitude):

    for layer_index in range(num_output):
        data = Variable(inputs, requires_grad = True)
        data = data.cuda()

        out_features = model.intermediate_forward(data, layer_index)
        # output, out_features = model.feature_list(data)
        # out_features = out_features[layer_index]
        out_features = out_features.view(out_features.size(0), out_features.size(1), -1)
        out_features = torch.mean(out_features, 2)

        gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                gaussian_score = term_gau.view(-1,1)
            else:
                gaussian_score = torch.cat((gaussian_score, term_gau.view(-1,1)), 1)

        # Input_processing
        sample_pred = gaussian_score.max(1)[1]
        batch_sample_mean = sample_mean[layer_index].index_select(0, sample_pred)
        zero_f = out_features - Variable(batch_sample_mean)
        pure_gau = -0.5*torch.mm(torch.mm(zero_f, Variable(precision[layer_index])), zero_f.t()).diag()
        loss = torch.mean(-pure_gau)
        loss.backward()

        gradient =  torch.ge(data.grad.data, 0)
        gradient = (gradient.float() - 0.5) * 2

        tempInputs = torch.add(data.data, -magnitude, gradient)

        noise_out_features = model.intermediate_forward(Variable(tempInputs), layer_index)
        # noise_out, noise_out_features = model.feature_list(Variable(tempInputs))
        # noise_out_features = noise_out_features[layer_index]
        noise_out_features = noise_out_features.view(noise_out_features.size(0), noise_out_features.size(1), -1)
        noise_out_features = torch.mean(noise_out_features, 2)
        noise_gaussian_score = 0
        for i in range(num_classes):
            batch_sample_mean = sample_mean[layer_index][i]
            zero_f = noise_out_features.data - batch_sample_mean
            term_gau = -0.5*torch.mm(torch.mm(zero_f, precision[layer_index]), zero_f.t()).diag()
            if i == 0:
                noise_gaussian_score = term_gau.view(-1,1)
            else:
                noise_gaussian_score = torch.cat((noise_gaussian_score, term_gau.view(-1,1)), 1)

        noise_gaussian_score, _ = torch.max(noise_gaussian_score, dim=1)

        noise_gaussian_score = np.asarray(noise_gaussian_score.cpu().numpy(), dtype=np.float32)
        if layer_index == 0:
            Mahalanobis_scores = noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))
        else:
            Mahalanobis_scores = np.concatenate((Mahalanobis_scores, noise_gaussian_score.reshape((noise_gaussian_score.shape[0], -1))), axis=1)

    return Mahalanobis_scores

def tune_mahalanobis_hyperparams(args, model, num_classes, train_loader, val_loader):

    save_dir = os.path.join('cache', 'mahalanobis', args.name)

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    model.eval()

    # set information about feature extaction
    temp_x = torch.rand(2, 3, 32, 32)
    temp_x = Variable(temp_x).cuda()
    temp_list = model.feature_list(temp_x)[1]
    num_output = len(temp_list)
    feature_list = np.empty(num_output)
    count = 0
    for out in temp_list:
        feature_list[count] = out.size(1)
        count += 1

    print('get sample mean and covariance')
    filename1 = os.path.join(save_dir, f'{args.in_dataset}_{args.model}_class_mean.npy')
    filename2 = os.path.join(save_dir, f'{args.in_dataset}_{args.model}_precision.npy')

    # if not os.path.exists(filename1):
    sample_mean, precision = sample_estimator(model, num_classes, feature_list, train_loader)
    # for i in range(len(sample_mean)):
    #     # print(mean.shape)
    #     sample_mean[i] = np.array([item.cpu().detach().numpy() for item in sample_mean[i]])
    #     precision[i] = np.array([item.cpu().detach().numpy() for item in precision[i]])
    # sample_mean = np.array(sample_mean)
    # precision = np.array(precision)
    # sample_mean = [s.cpu().detach().numpy() for s in sample_mean]
    # precision = [p.cpu().detach().numpy()  for p in precision]
    # sample_mean = np.array(sample_mean)
    # precision = np.array(precision)
    # print(sample_mean.shape, precision.shape)
    # np.save(filename, np.array([sample_mean, precision]))
    np.save(filename1, sample_mean)
    np.save(filename2, precision)

    sample_mean = np.load(filename1, allow_pickle=True)
    precision = np.load(filename2, allow_pickle=True)
    # sample_mean = [torch.from_numpy(s).cuda() for s in sample_mean]
    sample_mean = [s.cuda() for s in sample_mean]
    precision = [torch.from_numpy(p).float().cuda() for p in precision]

    print('train logistic regression model')
    m = 500

    train_in = []
    train_in_label = []
    train_out = []

    val_in = []
    val_in_label = []
    val_out = []

    cnt = 0
    for data, target in val_loader:
        data = data.numpy()
        target = target.numpy()
        for x, y in zip(data, target):
            cnt += 1
            if cnt <= m:
                train_in.append(x)
                train_in_label.append(y)
            elif cnt <= 2*m:
                val_in.append(x)
                val_in_label.append(y)

            if cnt == 2*m:
                break
        if cnt == 2*m:
            break

    print('In {} {}'.format(len(train_in), len(val_in)))

    criterion = nn.CrossEntropyLoss().cuda()
    adv_noise = 0.05

    args.batch_size = args.batch
    for i in range(int(m/args.batch_size) + 1):
        if i*args.batch_size >= m:
            break
        data = torch.tensor(train_in[i*args.batch_size:min((i+1)*args.batch_size, m)])
        target = torch.tensor(train_in_label[i*args.batch_size:min((i+1)*args.batch_size, m)])
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        # output = model(data)

        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True).cuda()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2

        adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

        train_out.extend(adv_data.cpu().numpy())

    for i in range(int(m/args.batch_size) + 1):
        if i*args.batch_size >= m:
            break
        data = torch.tensor(val_in[i*args.batch_size:min((i+1)*args.batch_size, m)])
        target = torch.tensor(val_in_label[i*args.batch_size:min((i+1)*args.batch_size, m)])
        data = data.cuda()
        target = target.cuda()
        data, target = Variable(data, volatile=True), Variable(target)
        # output = model(data)

        model.zero_grad()
        inputs = Variable(data.data, requires_grad=True).cuda()
        output = model(inputs)
        loss = criterion(output, target)
        loss.backward()

        gradient = torch.ge(inputs.grad.data, 0)
        gradient = (gradient.float()-0.5)*2

        adv_data = torch.add(input=inputs.data, other=gradient, alpha=adv_noise)
        adv_data = torch.clamp(adv_data, 0.0, 1.0)

        val_out.extend(adv_data.cpu().numpy())

    print('Out {} {}'.format(len(train_out),len(val_out)))

    train_lr_data = []
    train_lr_label = []
    train_lr_data.extend(train_in)
    train_lr_label.extend(np.zeros(m))
    train_lr_data.extend(train_out)
    train_lr_label.extend(np.ones(m))
    train_lr_data = torch.tensor(train_lr_data)
    train_lr_label = torch.tensor(train_lr_label)

    best_fpr = 1.1
    best_magnitude = 0.0

    for magnitude in [0.0, 0.01, 0.005, 0.002, 0.0014, 0.001, 0.0005]:
        train_lr_Mahalanobis = []
        total = 0
        for data_index in range(int(np.floor(train_lr_data.size(0) / args.batch_size)) + 1):
            if total >= 2*m:
                break
            data = train_lr_data[total : total + args.batch_size].cuda()
            total += args.batch_size
            Mahalanobis_scores = get_Mahalanobis_score(data, model, num_classes, sample_mean, precision, num_output, magnitude)
            train_lr_Mahalanobis.extend(Mahalanobis_scores)

        train_lr_Mahalanobis = np.asarray(train_lr_Mahalanobis, dtype=np.float32)
        # print(train_lr_Mahalanobis, train_lr_label)
        regressor = LogisticRegressionCV(n_jobs=-1).fit(train_lr_Mahalanobis, train_lr_label)

        print('Logistic Regressor params: {} {}'.format(regressor.coef_, regressor.intercept_))

        t0 = time.time()
        f1 = open(os.path.join(save_dir, f"{args.in_dataset}_{args.model}_confidence_mahalanobis_In.txt"), 'w')
        f2 = open(os.path.join(save_dir, f"{args.in_dataset}_{args.model}_confidence_mahalanobis_Out.txt"), 'w')

    ########################################In-distribution###########################################
        print("Processing in-distribution images")

        count = 0
        all_confidence_scores_in, all_confidence_scores_out = [], []
        for i in range(int(m/args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            images = torch.tensor(val_in[i * args.batch_size : min((i+1) * args.batch_size, m)]).cuda()
            # if j<1000: continue
            batch_size = images.shape[0]
            Mahalanobis_scores = get_Mahalanobis_score(images, model, num_classes, sample_mean, precision, num_output, magnitude)
            confidence_scores_in = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
            all_confidence_scores_in.extend(confidence_scores_in)

            for k in range(batch_size):
                f1.write("{}\n".format(confidence_scores_in[k]))

            count += batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time()-t0))
            t0 = time.time()

    ###################################Out-of-Distributions#####################################
        t0 = time.time()
        print("Processing out-of-distribution images")
        count = 0

        for i in range(int(m/args.batch_size) + 1):
            if i * args.batch_size >= m:
                break
            images = torch.tensor(val_out[i * args.batch_size : min((i+1) * args.batch_size, m)]).cuda()
            # if j<1000: continue
            batch_size = images.shape[0]

            Mahalanobis_scores = get_Mahalanobis_score(images, model, num_classes, sample_mean, precision, num_output, magnitude)

            confidence_scores_out = -regressor.predict_proba(Mahalanobis_scores)[:, 1]
            all_confidence_scores_out.extend(confidence_scores_out)

            for k in range(batch_size):
                f2.write("{}\n".format(confidence_scores_out[k]))

            count += batch_size
            print("{:4}/{:4} images processed, {:.1f} seconds used.".format(count, m, time.time()-t0))
            t0 = time.time()

        f1.close()
        f2.close()

        # results = metric(save_dir, stypes)
        # print_results(results, stypes)
        # fpr = results['mahalanobis']['FPR']
        all_confidence_scores_in = np.array(all_confidence_scores_in).reshape(-1, 1)
        all_confidence_scores_out = np.array(all_confidence_scores_out).reshape(-1, 1)
        print(all_confidence_scores_in.shape)
        print(all_confidence_scores_out.shape)

        _, _, _, fpr = get_measures(all_confidence_scores_in, all_confidence_scores_out)

        if fpr < best_fpr:
            best_fpr = fpr
            best_magnitude = magnitude
            best_regressor = regressor

    # print('Best Logistic Regressor params: {} {}'.format(best_regressor.coef_, best_regressor.intercept_))
    # print('Best magnitude: {}'.format(best_magnitude))

    print('Best Logistic Regressor params: {} {}'.format(best_regressor.coef_, best_regressor.intercept_))
    print('Best magnitude: {}'.format(best_magnitude))

    # sample_mean = [torch.from_numpy(s).cuda() for s in sample_mean]
    # precision = [torch.from_numpy(p).float().cuda() for p in precision]
    return sample_mean, precision, best_regressor, best_magnitude