import matplotlib
matplotlib.use('Agg')

import torch
import torchvision
from torchvision.transforms import v2


import pandas as pd
import tqdm


class Dataset(torch.utils.data.Dataset):
    def __init__(self, path, transform, labels_path = None):
        self.labels_df = None
        if labels_path is not None:
            self.labels_df = pd.read_csv(labels_path)

        self.img_dir = path
        self.transform = transform

    def __len__(self):
        if self.labels_df is None:
            return 10000
        return len(self.labels_df)

    def __getitem__(self, idx):
        label = 0
        if self.labels_df is not None:
            label = self.labels_df.iloc[idx, 1]

        img_name = '0' * (5 - len(str(idx))) + str(idx)

        if self.labels_df is not None:
            img_name = 'trainval_' + img_name
        else:
            img_name = 'test_' + img_name

        img_path_str = self.img_dir + '/' + img_name + '.jpg'

        # image = cv2.imread(img_path)
        # image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        image = torchvision.io.decode_image(img_path_str, mode=torchvision.io.ImageReadMode.RGB)
        # image = image.float() / 255.0

        image = self.transform(image)

        return (image, label)

def get_data(batch_size, transform_train):
    trainvalset = Dataset(path=f'{data_dir_path}/trainval', transform=transform_train, labels_path=f'{data_dir_path}/labels.csv')
    train_loader = torch.utils.data.DataLoader(trainvalset, batch_size=batch_size,
                                               shuffle=True, num_workers=2)
    return train_loader


data_dir_path = './bhw1'
saves_path = './saves'
checkpoint_dir = './previous_runs_best_saves'


print("=== Loading data... ===")

batch_size=128
transform = v2.Compose([
    v2.ToDtype(torch.float32, scale=True),
])
train_loader = get_data(batch_size=batch_size, transform_train=transform)

print("=== Data loaded successfully ===")


channel_sum = torch.zeros(3)
channel_N = 0

for data, target in tqdm.tqdm(train_loader, desc='Calculating mean', unit='batch', leave=True, dynamic_ncols=True):
    channel_sum += torch.sum(data, dim=[0, 2, 3])
    channel_N += data.shape[0] * data.shape[2] * data.shape[3]

channel_mean = channel_sum / channel_N

channel_quadro_sum = torch.zeros(3)
channel_mean_reshaped = channel_mean.view(3, 1, 1, 1)
for data, target in tqdm.tqdm(train_loader, desc='Calculating std', unit='batch', leave=True, dynamic_ncols=True):
    data = torch.transpose(data, 0, 1)
    channel_quadro_sum += torch.sum((data - channel_mean_reshaped)**2, dim=[1, 2, 3])

channel_std = (channel_quadro_sum / channel_N).sqrt()

print('mean: ', channel_mean)
print('std: ', channel_std)

# mean:  (0.5691, 0.5447, 0.4933)
# std:  (0.2386, 0.2335, 0.2516)