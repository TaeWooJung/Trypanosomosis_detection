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
Source code for experiment 5 (No edge sharpening)
********************************* Updates 2020/05/12 *********************************
1. Different mean and standard deviation for each set
'''
# T1
mean_1 = [0.55619025, 0.54442054, 0.5668326]
std_1 = [0.15338533, 0.14782225, 0.15213947]

# T2
mean_2 = [0.55744743, 0.5457371, 0.5683189]
std_2 = [0.15367888, 0.14802714, 0.15239693]

# T3
mean_3 = [0.55702317, 0.54563445, 0.5682874]
std_3 = [0.15336181, 0.14766185, 0.15203352]

# T4
mean_4 = [0.55669624, 0.5450452, 0.56758934]
std_4 = [0.1546661, 0.14899929, 0.15342303]

# T5
mean_5 = [0.55679774, 0.54509205, 0.56761116]
std_5 = [0.15356685, 0.14808999, 0.15250486]

train_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=mean_1, std=std_1)
                       ])

test_transforms = transforms.Compose([
                           transforms.ToTensor(),
                           transforms.Normalize(mean=mean_1, std=std_1)
                       ])

# Controllers
x = 1               # T1 = 1, T2 =2 , ..., T5 = 5
BATCH_SIZE = 128
EPOCHS = 200
istrain = False
istest = False

Set = 'set{}'.format(x)
SAVE_DIR = 'models/resnet18_TP5/{}'.format(Set)
MODEL_SAVE_DIR = 'models/resnet18_TP5/{}'.format(Set)
TRAIN_LOG = "resnet18_TP5.csv"
TEST_LOG = "resnet18_TP5_test.csv"

# if directory 'models' does not exist, create a new directory called 'models'
if not os.path.isdir(f'{SAVE_DIR}'):
    os.makedirs(f'{SAVE_DIR}')

device = torch.device('cuda')
print('Running on GPU')

TRAIN_DIR = 'trypanosome_data/experiment_5/{}/train'.format(Set)
VALID_DIR = 'trypanosome_data/experiment_5/{}/valid'.format(Set)
TEST_DIR = 'trypanosome_data/experiment_5/{}/test'.format(Set)

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
                MODEL_SAVE_PATH = os.path.join(SAVE_DIR, 'resnet18_TP5_epoch{0:03d}.pt'.format(epoch+1))
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

    with open(TEST_LOG, "a") as j:

        for model_name in model_names:

            model.load_state_dict(torch.load(os.path.join(MODEL_SAVE_DIR, model_name)))
            test_loss, test_acc, sensitivity, specificity, PPV, NPV, samples = test(model, device, test_iterator, criterion)
            print('{0}, Test Loss: {1:0.4f}, Test Acc: {2:0.4f}, Sensitivity: {3:0.4f},'
                  ' Specificity: {4:0.4f}, PPV: {5:0.4f}, NPV: {6:0.4f}'
                  .format(model_name, test_loss, test_acc, sensitivity, specificity, PPV, NPV))
            j.write('{0}, Test Loss: {1:0.4f}, Test Acc: {2:0.4f}, Sensitivity: {3:0.4f},'
                    ' Specificity: {4:0.4f}, PPV: {5:0.4f}, NPV: {6:0.4f}\n'
                    .format(model_name, test_loss, test_acc, sensitivity, specificity, PPV, NPV))

            if sample:
                sample_dir = os.path.join('sample', '{}_{}'.format(Set, model_name))
                os.mkdir(sample_dir)
                # samples = [(path of an image, 'TP' or 'TN' or 'FP' or 'FN', probability), ...]
                p = [0.7, 0.99]   # set lower and upper bound
                # sorted([i for i in samples if i[-1] > p[0] and i[-1] < p[1] and i[1] == 'FP'], key=lambda x: x[2])
                organized_samples = [i for i in samples if i[-1] > p[0] and i[-1] < p[1] and i[1] == 'FP']
                # 'sample/model_name' is a directory to save samples
                save_samples(organized_samples, sample_dir)


# Train and test a models
run_train(model, istrain, SAVE_DIR, TRAIN_LOG, Set)
run_test(model, istest, MODEL_SAVE_DIR, TEST_LOG, True)
