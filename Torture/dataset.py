import os
import torch
from torchvision import transforms
from torch.utils.data import Dataset
import numpy as np
import PIL.Image


class InfinityDataloaderWraper(object):
    def __init__(self, dataloader):
        self.dataloader = dataloader
        self._dataiterator = iter(self.dataloader)

    def __next__(self):
        try:
            batch = self._dataiterator.next()
        except StopIteration:
            self._dataiterator = iter(self.dataloader)
            batch = self._dataiterator.next()
        return batch

    def next(self):
        return self.__next__()


class NumpyImageLabelDataset(Dataset):
    def __init__(self, imgs, label, transform=None, require_onehot=False):
        self.imgs = imgs
        self.label = label
        self.transform = transform

        self.num_classes = np.unique(label).shape[0]
        self.require_onehot = require_onehot

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = np.array(self.imgs[index])
        label = self.label[index]
        if img.ndim == 2:
            img = np.expand_dims(img, 2)
            img = np.repeat(img, 3, axis=2)

        # label_onehot = torch.zeros([self.num_classes], dtype=torch.int64)
        # label_onehot[label] = 1
        # if self.require_onehot is True:
        #     label = label_onehot

        img = (img * 255.0).astype(np.uint8)
        img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img, label


class NumpyImageDataset(Dataset):
    def __init__(self, imgs, transform=None):
        self.imgs = imgs
        self.transform = transform

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        img = np.array(self.imgs[index])
        if img.ndim == 2:
            img = np.expand_dims(img, 2)
            img = np.repeat(img, 3, axis=2)

        img = (img * 255.0).astype(np.uint8)
        img = PIL.Image.fromarray(img)
        if self.transform is not None:
            img = self.transform(img)

        return img


class ImageFolderDataset(Dataset):
    def __init__(self, img_paths, label, root="/home/Data", transform=None):
        self.root = root
        self.img_paths = img_paths
        self.label = label
        self.transform = transform

        self.num_classes = np.unique(label).shape[0]

    def __len__(self):
        return len(self.img_paths)

    def __getitem__(self, index):
        img_p = self.img_paths[index]
        label = self.label[index]

        label_onehot = torch.zeros([self.num_classes], dtype=torch.int64)
        label_onehot[label] = 1

        img = PIL.Image.open(os.path.join(self.root, img_p)).convert("RGB")
        if self.transform is not None:
            img = self.transform(img)

        return img, label


# def get_nsamples(data_loader, N):
#     x = []
#     y = []
#     n = 0
#     while n < N:
#         x_next, y_next = next(iter(data_loader))
#         x.append(x_next)
#         y.append(y_next)
#         n += x_next.size(0)
#     x = torch.cat(x, dim=0)[:N]
#     y = torch.cat(y, dim=0)[:N]
#     return x, y

if __name__ == "__main__":
    # default setting for Caltech101
    tfs = transforms.Compose(
        transforms.Resize(256), transforms.RandomCrop(224, pad_if_needed=True),
        transforms.RandomHorizontalFlip(), transforms.ToTensor(),
        transforms.Normalize([0.5, 0.5, 0.5], [0.5, 0.5, 0.5]))
    print(tfs)
