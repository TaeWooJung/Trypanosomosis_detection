import torch
import torch.nn.functional as F
import numpy as np
import torchvision.datasets as datasets
import matplotlib.pyplot as plt
import cv2
from sklearn.metrics import confusion_matrix, accuracy_score


'''
********************************* Updates 2020/05/03 *********************************
This py file consist of train and test functions that can be imported locally.
I have decided to separate these functions from experimental py files, in order to improve
readability of my code for your users. I will make use of this functions from experiment 4.
Also, I will correct test results obtained for previous experiments, especially experiment 3.

Main updates:
1.  For calculation of sensitivity, specificity, NPV and PPV, I have adjusted to make use
    of confusion matrix from sklearn. They have been verified by comparing accuracy calculated 
    from confusion matrix and accuracy obtained from sklearn.
2.  Previously, evaluation of validation and testing have been done using the same function
    called 'evaluate'. On this set up, evaluation of validation has been added to a function
    'train' and evaluation of testing can be performed by newly added function called 'test'.
    
********************************* Updates 2020/05/12 *********************************
Sampling of images are needed for qualitative analysis

Main updates:
1.  Introduce sampling functions for qualitative analysis

'''


class ImageFolderWithPaths(datasets.ImageFolder):
    """
    Jong, A. (2019, July 5). PyTorch Image File Paths With Dataset Dataloader. Retrieved from https://gist.github.com/andrewjong/6b02ff237533b3b2c554701fb53d5c4d.
    """

    # override the __getitem__ method. this is the method that dataloader calls
    def __getitem__(self, index):
        # this is what ImageFolder normally returns
        original_tuple = super(ImageFolderWithPaths, self).__getitem__(index)
        # the image file path
        path = self.imgs[index][0]
        # make a new tuple that includes original and the path
        tuple_with_path = (original_tuple + (path,))
        return tuple_with_path


def sampling(paths, y, fx, preds, samples):

    i = 0
    for path in paths:
        if y[i].item() == preds[i].item():
            if preds[i].item() == 1:
                samples.append((path, 'TP', fx[i][preds[i]].item()))
            else:
                samples.append((path, 'TN', fx[i][preds[i]].item()))
        else:
            if preds[i].item() == 1:
                samples.append((path, 'FP', fx[i][preds[i]].item()))
            else:
                samples.append((path, 'FN', fx[i][preds[i]].item()))
        i += 1

    return samples


def save_samples(samples, save_dir):
    fig = 0
    count = 1
    plt.figure(figsize=(10, 12))

    for path, result, probability in samples:

        if count == 17:
            fig += 1
            plt.savefig('{}/{}'.format(save_dir, fig))
            plt.close()
            plt.figure(figsize=(10, 12))
            count = 1

        img = cv2.imread(path, cv2.COLOR_BGR2RGB)
        name = '{}_{}'.format(path.split('\\')[-2][0], path.split('\\')[-1][:-4])
        xlabel = '{0}, {1:0.5f}'.format(result, probability)

        ax = plt.subplot(4, 4, count)
        ax.set_title(name)
        ax.imshow(img)
        ax.set_xlabel(xlabel)
        ax.set_xticks([]), ax.set_yticks([])  # remove unnecessary numbers on both x and y axes
        count += 1

    if count != 17:
        plt.savefig('{}/{}'.format(save_dir, fig+1))
        plt.close()


def test_metrics(TN, FP, FN, TP):

    acc = (TP + TN) / (TN + FP + FN + TP)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)

    return acc, sensitivity, specificity, PPV, NPV


def get_n_params(model):
    pp = 0
    for p in list(model.parameters()):
        n = 1
        for s in list(p.size()):
            n = n*s
        pp += n
    return pp


def train(model, device, train_iterator, valid_iterator, optimizer, criterion):
    train_loss = []
    train_acc = []
    model.train()

    for train_x, train_y in train_iterator:
        train_x = train_x.to(device)
        train_y = train_y.to(device)

        optimizer.zero_grad()

        train_fx = F.softmax(model(train_x), dim=1)
        train_preds = train_fx.argmax(dim=1)

        # calculate loss and accuracy
        loss = criterion(train_fx, train_y)
        acc = accuracy_score(train_y.detach().cpu(), train_preds.detach().cpu())

        train_loss.append(loss.item())
        train_acc.append(acc)

        loss.backward()
        optimizer.step()

    val_loss = []
    val_acc = []
    model.eval()

    with torch.no_grad():
        for val_x, val_y in valid_iterator:
            val_x = val_x.to(device)
            val_y = val_y.to(device)

            val_fx = F.softmax(model(val_x), dim=1)
            val_preds = val_fx.argmax(dim=1)

            # calculate loss and accuracy
            loss = criterion(val_fx, val_y).item()
            acc = accuracy_score(val_y.detach().cpu(), val_preds.detach().cpu())

            val_loss.append(loss)
            val_acc.append(acc)

    return np.mean(train_loss), np.mean(train_acc), np.mean(val_loss), np.mean(val_acc)


def test(model, device, test_iterator, criterion):

    samples = []
    epoch_TN, epoch_FP, epoch_FN, epoch_TP = 0, 0, 0, 0
    test_loss = []

    model.eval()

    with torch.no_grad():
        for test_x, test_y, paths in test_iterator:
            test_x = test_x.to(device)
            test_y = test_y.to(device)

            test_fx = F.softmax(model(test_x), dim=1)
            test_preds = test_fx.argmax(dim=1)
            TN, FP, FN, TP = confusion_matrix(test_y.detach().cpu(), test_preds.detach().cpu(), labels=[0,1]).ravel()
            loss = criterion(test_fx, test_y).item()

            test_loss.append(loss)

            epoch_TN += TN
            epoch_FP += FP
            epoch_FN += FN
            epoch_TP += TP

            samples = sampling(paths, test_y, test_fx, test_preds, samples)

    test_acc, sensitivity, specificity, PPV, NPV = test_metrics(epoch_TN, epoch_FP, epoch_FN, epoch_TP)

    return np.mean(test_loss), test_acc, sensitivity, specificity, PPV, NPV, samples

