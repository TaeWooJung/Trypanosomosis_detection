import torch
import torch.nn as nn
import torch.optim as optim
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import os
import torchvision.models as models
import csv
from TP_resnet18 import train, test, ImageFolderWithPaths, save_samples

'''
Source code for experiment 6 (With Edge Sharpening)
********************************* Updates 2020/05/12 *********************************
1. Different mean and standard deviation for each set
'''

# T1
mean_1 = [0.55486816, 0.5435331, 0.56540835]
std_1 = [0.17937568, 0.17555736, 0.17809482]

# T2
mean_2 = [0.55610305, 0.5448368, 0.5668692]
std_2 = [0.17982134, 0.17594525, 0.17850631]

# T3
mean_3 = [0.55567974, 0.54473305, 0.56683266]
std_3 = [0.17947727, 0.17556751, 0.17813207]

# T4
mean_4 = [0.55535865, 0.544147, 0.5661491 ]
std_4 = [0.1806105, 0.17670362, 0.17932624]

# T5
mean_5 = [0.5554702,  0.5442094, 0.56618005]
std_5 = [0.17949283, 0.17575373, 0.17836432]


train_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=mean_5, std=std_5)
                       ])

test_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=mean_5, std=std_5)
                       ])

# Controllers
x = 5               # T5 = 5
BATCH_SIZE = 128
EPOCHS = 200
istrain = False
istest = False

Set = 'set{}'.format(x)
SAVE_DIR = 'models/resnet18_TP6/{}'.format(Set)
MODEL_SAVE_DIR = 'models/resnet18_TP6/{}'.format(Set)
TRAIN_LOG = 'resnet18_TP6_{}.csv'.format(Set)
TEST_LOG = 'resnet18_TP6_test_{}.csv'.format(Set)

# if directory 'models' does not exist, create a new directory called 'models'
if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

device = torch.device('cuda')
print('Running on GPU')

TRAIN_DIR = 'trypanosome_data/Sharpened/{}/train'.format(Set)
VALID_DIR = 'trypanosome_data/Sharpened/{}/valid'.format(Set)
TEST_DIR = 'trypanosome_data/Sharpened/{}/test'.format(Set)


train_data = datasets.ImageFolder(TRAIN_DIR, train_transforms)
valid_data = datasets.ImageFolder(VALID_DIR, test_transforms)
test_data = ImageFolderWithPaths(TEST_DIR, test_transforms)

print('Number of training examples: {}'.format(len(train_data)))
print('Number of validation examples: {}'.format(len(valid_data)))
print('Number of testing examples: {}\n'.format(len(test_data)))

train_iterator = torch.utils.data.DataLoader(train_data, shuffle=True, batch_size=BATCH_SIZE)
valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
test_iterator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

model = models.resnet18(pretrained=False).to(device)

for param in model.parameters():
    param.requires_grad = False

# out_features = number of classes in the dataset
model.fc = nn.Linear(in_features=512, out_features=2).to(device)
criterion = nn.CrossEntropyLoss()


def run_train(model, istrain, SAVE_DIR, TRAIN_LOG, Set):

    if istrain:

        best_valid_acc = 0

        # For SGD with adaptive lr
        # lr = 0.1
        # optimizer = optim.SGD(model.parameters(), lr=lr, momentum=0.9, weight_decay=1e-4)
        # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, 'min', verbose=True)

        # For Adam optimizer with lr=0.001, batch=128
        lr = 0.001
        optimizer = optim.Adam(model.parameters(), lr=lr)

        with open(TRAIN_LOG, "w", newline='') as csv_file:
            write = csv.writer(csv_file)
            write.writerow(['Epoch', 'Train_loss', 'Train_acc', 'Valid_loss', 'Valid_acc'])
            for epoch in range(EPOCHS):

                train_loss, train_acc, valid_loss, valid_acc = train(model, device, train_iterator, valid_iterator, optimizer, criterion)
                MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'resnet18_TP6_epoch{0:03d}.pt'.format(epoch+1))
                # scheduler.step(valid_loss)

                if valid_acc > best_valid_acc:
                    best_valid_acc = valid_acc
                    torch.save(model.state_dict(), MODEL_SAVE_PATH)

                print(
                    '{0} - Epoch: {1:0>2}, Train Loss: {2:0.4f}, Train Acc: {3:0.4f}, '
                    'Val. Loss: {4:0.4f}, Val. Acc: {5:0.4f}'.format(Set, epoch + 1, train_loss,
                                                                     train_acc, valid_loss, valid_acc))
                write.writerow([epoch+1, train_loss, train_acc, valid_loss, valid_acc])

            csv_file.close()


def run_test(model, istest, MODEL_SAVE_DIR, TEST_LOG, sample=False):
    if not istest:
        return

    model_names = sorted(os.listdir(MODEL_SAVE_DIR))

    with open(TEST_LOG, "w", newline='') as csv_file:
        write = csv.writer(csv_file)
        write.writerow(['Model', 'Test_loss', 'Test_Acc', 'Sensitivity', 'Specificity', 'PPV', 'NPV'])

        for model_name in model_names:

            model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, model_name)))
            test_loss, test_acc, sensitivity, specificity, PPV, NPV, samples = test(model, device, test_iterator, criterion)
            # print(TNFPFNTP)
            print('{0}, Test Loss: {1:0.4f}, Test Acc: {2:0.4f}, Sensitivity: {3:0.4f},'
                  ' Specificity: {4:0.4f}, PPV: {5:0.4f}, NPV: {6:0.4f}'
                  .format(model_name, test_loss, test_acc, sensitivity, specificity, PPV, NPV))
            write.writerow([model_name, test_loss, test_acc, sensitivity, specificity, PPV, NPV])

            if sample:
                sample_dir = os.path.join('sample', 'sharpened_{}_{}'.format(Set, model_name))
                os.mkdir(sample_dir)
                # samples = [(path of an image, 'TP' or 'TN' or 'FP' or 'FN', probability), ...]
                p = [0, 0.55]   # set lower and upper bound
                organized_samples = sorted([i for i in samples if i[-1] > p[0] and i[-1] < p[1] and i[1] == 'FP'], key=lambda x: x[2])
                # 'sample/model_name' is a directory to save samples
                save_samples(organized_samples, sample_dir)

        csv_file.close()

# Train and test a models
run_train(model, istrain, SAVE_DIR, TRAIN_LOG, Set)
run_test(model, istest, MODEL_SAVE_DIR, TEST_LOG, False)
