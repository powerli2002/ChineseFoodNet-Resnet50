import torch
import numpy as np

from torchvision import transforms
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from utility.cutout import Cutout

from utils.getNamePathLabel import getPathLabel

# 将数据集中的图片统一修改大小
image_size = 600

# mean[0.52011104 0.44459117 0.30962785]
# std [0.25595631 0.25862494 0.26925405]
data_transforms = {
    'train': transforms.Compose([
        # transforms.RandomResizedCrop(600),
        transforms.CenterCrop(500),
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52011104, 0.44459117, 0.30962785], std=[0.25595631, 0.25862494, 0.26925405]),
        Cutout()
    ]),
    'test': transforms.Compose([
        transforms.CenterCrop(500),
        transforms.Resize(128),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52011104, 0.44459117, 0.30962785], std=[0.25595631, 0.25862494, 0.26925405])
    ]),
    'valid': transforms.Compose([
        transforms.CenterCrop(500),
        transforms.Resize(128),
        transforms.RandomHorizontalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.52011104, 0.44459117, 0.30962785], std=[0.25595631, 0.25862494, 0.26925405])
    ])
}

class ChineseFoodNetTrainSet(Dataset):
    def __init__(self, transform=data_transforms['train'], root='./ChineseFoodNet/release_data/train/'):
        self.root = root
        self.transforms = transform
        paths_labels = getPathLabel('train_sets')
        self.paths = paths_labels['path']
        self.labels = paths_labels['label']

    def __getitem__(self, index):
        image = Image.open(self.root + self.paths[index]).convert('RGB')
        image = self.transforms(image)
        label = self.labels[index]

        return image, label

    def __len__(self):
        return len(self.labels)


class ChineseFoodNetTestSet(Dataset):
    def __init__(self, transform=data_transforms['test'], root='./ChineseFoodNet/release_data/test/'):
        self.root = root
        self.transforms = transform
        paths_labels = getPathLabel('test_sets')
        self.paths = paths_labels['path']
        self.labels = paths_labels['label']

    def __getitem__(self, index):
        image = Image.open(self.root + self.paths[index]).convert('RGB')
        # image = Image.open(self.root + self.paths[index])
        image = self.transforms(image)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)


class ChineseFoodNetValSet(Dataset):
    def __init__(self, transform=data_transforms['valid'], root='./ChineseFoodNet/release_data/val/'):
        self.root = root
        self.transforms = transform
        paths_labels = getPathLabel('val_sets')
        self.paths = paths_labels['path']
        self.labels = paths_labels['label']

    def __getitem__(self, index):
        image = Image.open(self.root + self.paths[index]).convert("RGB")
        image = self.transforms(image)
        label = self.labels[index]
        return image, label

    def __len__(self):
        return len(self.labels)


def getStatistics(dataset_class='train'):
    """
    函数用于计算某个数据集图片的期望值和标准差，用于后续对图片进行归一化
    但是好像不需要每个集合对自己的期望和标准差进行归一化，都可以使用训练集的统计特征
    由于数据集太大，如果每次训练都来调用这个函数算一次，会浪费很多时间和内存，如果在图片裁剪大小没有变化的情况下，可以提前算好
    :param dataset_class: 默认为'train'，枚举值
    :return: mean and std
    """
    print('Calculate the mean and std of {} set '.format(dataset_class))
    if dataset_class == 'train':
        dataset = ChineseFoodNetTrainSet(transform=transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor()
        ]))
    elif dataset_class == 'test':
        dataset = ChineseFoodNetTestSet(transform=transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor()
        ]))
    elif dataset_class == 'val':
        dataset = ChineseFoodNetValSet(transform=transforms.Compose([
            transforms.RandomResizedCrop(image_size),
            transforms.ToTensor()
        ]))
    else:
        raise ValueError('请输入正确的集合类别:训练集(train), 测试集(test), 验证集(val)')

    dataloader = DataLoader(dataset=dataset, batch_size=512)
    """
    下面这种算法太吃内存，16G的内存都会爆，还是批量处理比较好
    """

    # data = torch.cat([d[0] for d in dataloader])
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    # data.to(device)
    # return data.mean(dim=[0, 2, 3]), data.std(dim=[0, 2, 3])

    mean_list = []
    std_list = []
    length = len(dataloader)
    for i, data in enumerate(dataloader):
        m = data[0].mean(dim=[0, 2, 3]).to(device)
        s = data[0].std(dim=[0, 2, 3]).to(device)
        mean_list.append(m.tolist())
        std_list.append(s.tolist())
        print('[{}/{}]: mean={}, std={}'.format(i, length - 1, m.tolist(), s.tolist()))
    return mean_list, std_list


if __name__ == '__main__':
    m, s = getStatistics()
    f1 = open('mean.txt', 'w')
    for line in m:
        f1.write(str(line) + '\n')
    f1.close()

    f2 = open('std.txt', 'w')
    for line in s:
        f2.write(str(line) + '\n')

    a_m = np.array(m)
    a_s = np.array(s)
    print('final: mean={}'.format(a_m.mean(axis=0)))
    print('final: std={}'.format(a_s.mean(axis=0)))
