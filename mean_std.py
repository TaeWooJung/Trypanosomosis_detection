import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import numpy as np

BATCH_SIZE = 64

# DIR = 'trypanosome_data/experiment_5/set5/train'
# DIR2 = 'trypanosome_data/experiment_5/set5/valid'
# DIR3 = 'trypanosome_data/experiment_5/set5/train'

DIR = 'Sharpened/set5/train'
DIR2 = 'Sharpened/set5/valid'
DIR3 = 'Sharpened/set5/train'

# adjusted into a range [0, 1]
transforms = transforms.Compose([transforms.ToTensor()])

train_data = datasets.ImageFolder(DIR, transforms)
valid_data = datasets.ImageFolder(DIR2, transforms)
test_data = datasets.ImageFolder(DIR3, transforms)

train_iterator = torch.utils.data.DataLoader(train_data, batch_size=BATCH_SIZE)
valid_iterator = torch.utils.data.DataLoader(valid_data, batch_size=BATCH_SIZE)
test_iterator = torch.utils.data.DataLoader(test_data, batch_size=BATCH_SIZE)

imgs = []

for (x, _) in train_iterator:
    for i in range(x.shape[0]):
        imgs.append(x[i])

for (x, _) in valid_iterator:
    for i in range(x.shape[0]):
        imgs.append(x[i])

for (x, _) in test_iterator:
    for i in range(x.shape[0]):
        imgs.append(x[i])

imgs = torch.stack(imgs).numpy()
print(imgs.shape)

# Find mean and standard deviation for each color channel
print(np.mean(imgs, axis=(0, 2, 3)))
print(np.std(imgs, axis=(0, 2, 3)))

'''
Without sharpening
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
'''


'''
Sharpened
# T5
mean_5 = [0.5554702  0.5442094  0.56618005]
std_5 = [0.17949283 0.17575373 0.17836432]
'''
