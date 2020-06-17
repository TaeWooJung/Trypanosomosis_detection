import torch
import torch.nn as nn
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torchvision.models as models
import numpy as np
from sklearn.metrics import accuracy_score, confusion_matrix

# Source code for testing with a dataset that only consist of negative patches from negative videos

# T5
mean_5 = [0.556668, 0.5451229, 0.56774807]
std_5 = [0.15359154, 0.14803909, 0.15245321]

test_transforms = transforms.Compose([
    transforms.ToTensor(),
    transforms.Normalize(mean=mean_5, std=std_5)
])

# Controllers
isquick = True

device = torch.device('cuda')
print('Running on GPU')

TEST_DIR = 'TP_wo_pos'
model_name = 'set5_SGD_TP5_TL_epoch032.pt'
MODEL_SAVE_DIR = 'models/resnet18_TP5/best/'
test_data = datasets.ImageFolder(TEST_DIR, test_transforms)
print('Number of testing examples: {}\n'.format(len(test_data)))

test_iterator = torch.utils.data.DataLoader(test_data, batch_size=64)

model = models.resnet18(pretrained=False).to(device)
criterion = nn.CrossEntropyLoss()

for param in model.parameters():
    param.requires_grad = False

# out_features = number of classes in the dataset
model.fc = nn.Linear(in_features=512, out_features=2).to(device)


def test_metrics(TN, FP, FN, TP):

    acc = (TP + TN) / (TN + FP + FN + TP)
    sensitivity = TP/(TP+FN)
    specificity = TN/(TN+FP)
    PPV = TP/(TP+FP)
    NPV = TN/(TN+FN)

    return acc, sensitivity, specificity, PPV, NPV


def quick_test(model, device, test_iterator, criterion):
    test_loss = []
    test_acc = []
    epoch_TN, epoch_FP, epoch_FN, epoch_TP = 0, 0, 0, 0
    model.eval()

    with torch.no_grad():
        for test_x, test_y in test_iterator:
            test_x = test_x.to(device)
            test_y = test_y.to(device)

            test_fx = model(test_x)
            test_preds = test_fx.argmax(dim=1)

            # calculate loss and accuracy
            loss = criterion(test_fx, test_y).item()
            acc = accuracy_score(test_y.detach().cpu(), test_preds.detach().cpu())
            TN, FP, FN, TP = confusion_matrix(test_y.detach().cpu(), test_preds.detach().cpu(), labels=[0,1]).ravel()

            test_loss.append(loss)
            test_acc.append(acc)
            epoch_TN += TN
            epoch_FP += FP
            epoch_FN += FN
            epoch_TP += TP

    return np.mean(test_loss), np.mean(test_acc), epoch_TN, epoch_FP, epoch_FN, epoch_TP


def run_quick_test(model, model_name, MODEL_SAVE_DIR, isquick):

    model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, model_name)))

    if isquick:
        with open("test.log", "a") as f:
            test_loss, test_acc, TN, FP, FN, TP = quick_test(model, device, test_iterator, criterion)
            _, sensitivity, specificity, PPV, NPV = test_metrics(TN, FP, FN, TP)
            print('Val. Loss: {0:0.4f}, Val. Acc: {1:0.4f}, TN: {2}, FP: {3}, FN: {4}, TP: {5}, Sensitivity: {6}, '
                  'Specificity: {7}, PPV: {8}, NPV: {9}'.format(test_loss, test_acc, TN, FP, FN, TP, sensitivity, specificity, PPV, NPV))
            f.write('Val. Loss: {0:0.4f}, Val. Acc: {1:0.4f}, TN: {2}, FP: {3}, FN: {4}, TP: {5}, Sensitivity: {6}, '
                  'Specificity: {7}, PPV: {8}, NPV: {9}\n'.format(test_loss, test_acc, TN, FP, FN, TP, sensitivity, specificity, PPV, NPV))


run_quick_test(model, model_name, MODEL_SAVE_DIR, isquick)

