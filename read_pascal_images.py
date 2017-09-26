
from torch.utils.data import Dataset
import torch
import os
import fnmatch
import scipy.misc as m
from skimage import io
from torchvision import transforms
import matplotlib.pyplot as plt

import pdb
import numpy as np
from progress_bar import InitBar

class Read_pascal_images(Dataset):

    def __init__(self, root, transform=None):
        self.transform = transform
        self.filenames = []
        for path, dirnames, filenames in os.walk(root):
            for filename in fnmatch.filter(filenames, '*.jpg'):
                self.filenames.append(os.path.join(root, filename))

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        image_name = self.filenames[idx]
        image = io.imread(image_name)
        #image = image.transpose(2, 0, 1)
        #image = image.astype(float) / 255.0
        image = transforms.ToPILImage()(image)
        if self.transform:
            image = self.transform(image)
        #print(type(image))
        #print(image.numpy())
        #print(image.max())
        return image

class Read_pascal_labeled_data(Dataset):

    def __init__(self, root, nCls, splits_path, split, img_size, apply_transform=True):
        self.apply_transform = apply_transform
        self.img_size = img_size
        self.nCls = nCls
        self.image_files = []
        self.label_files = []
        self.root = root
        with open(os.path.join(splits_path, '{}.txt'.format(split))) as f:
            lines = f.readlines()
        filenames = [l.strip() for l in lines]
        N = len(filenames)
        pbar = InitBar()
        print('Loading image and label filenames...\n')
        for i in range(N):
            pbar(100.0 * float(i) / float(N))
            image, label = filenames[i].split()
            self.image_files.append(root+image)
            self.label_files.append(root+label)

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        label_name = self.label_files[idx]
        image = io.imread(image_name)
        label = io.imread(label_name)
        #image = transforms.ToPILImage()(image)
        if self.apply_transform:
            image, label = self.transform(image, label)
        return image, label

    def transform(self, img, lbl):
        #img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img -= 255
        img = m.imresize(img, (self.img_size, self.img_size))
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)

        lbl[lbl==255] = 0
        lbl = lbl.astype(float)
        lbl = m.imresize(lbl, (self.img_size, self.img_size), 'nearest', mode='F')
        lbl = lbl.astype(int)

        img = torch.from_numpy(img).float()
        lbl = torch.from_numpy(lbl).long()
        return img, lbl

    def get_pascal_labels(self):
        return np.asarray([[0,0,0], [128,0,0], [0,128,0], [128,128,0], [0,0,128], [128,0,128],
                              [0,128,128], [128,128,128], [64,0,0], [192,0,0], [64,128,0], [192,128,0],
                              [64,0,128], [192,0,128], [64,128,128], [192,128,128], [0, 64,0], [128, 64, 0],
                              [0,192,0], [128,192,0], [0,64,128]])

    def encode_segmap(self, mask):
        mask = mask.astype(int)
        label_mask = np.zeros((mask.shape[0], mask.shape[1]), dtype=np.int16)
        for i, label in enumerate(self.get_pascal_labels()):
            label_mask[np.where(np.all(mask == label, axis=-1))[:2]] = i
        label_mask = label_mask.astype(int)
        return label_mask

    def decode_segmap(self, temp, plot=False):
        label_colours = self.get_pascal_labels()
        r = temp.copy()
        g = temp.copy()
        b = temp.copy()
        for l in range(0, self.nCls):
            r[temp == l] = label_colours[l, 0]
            g[temp == l] = label_colours[l, 1]
            b[temp == l] = label_colours[l, 2]

        rgb = np.zeros((temp.shape[0], temp.shape[1], 3))
        rgb[:, :, 0] = r
        rgb[:, :, 1] = g
        rgb[:, :, 2] = b
        if plot:
            plt.imshow(rgb)
            plt.show()
        else:
            return rgb

class Read_imagenet_data(Dataset):

    def __init__(self, root, img_size, apply_transform=True):
        self.apply_transform = apply_transform
        self.img_size = img_size
        self.image_files = []
        self.label_files = []
        self.root = root
        with open(os.path.join(root, 'ImageSets/CLS-LOC/train_cls.txt')) as f:
            lines = f.readlines()
        filenames = [l.strip() for l in lines]
        N = len(filenames)
        pbar = InitBar()
        print('Loading imagenet filenames...\n')
        for i in range(N):
            pbar(100.0 * float(i) / float(N))
            image = filenames[i].split()[0]
            self.image_files.append(root+'/'+image+'.JPEG')

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        image_name = self.image_files[idx]
        image = io.imread(image_name)
        #image = transforms.ToPILImage()(image)
        if self.apply_transform:
            image = self.transform(image)
        return image

    def transform(self, img):
        #img = img[:, :, ::-1]
        img = img.astype(np.float64)
        img = m.imresize(img, (self.img_size, self.img_size))
        img = img.astype(float) / 255.0
        img = img.transpose(2, 0, 1)

        img = torch.from_numpy(img).float()
        return img

class Rescale(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        self.output_size = output_size

    def __call__(self, input):
        h, w = input.shape[:2]
        if isinstance(self.output_size, int):
            if h > w:
                new_h, new_w = self.output_size * h / w, self.output_size
            else:
                new_h, new_w = self.output_size, self.output_size * w / h
        else:
            new_h, new_w = self.output_size

        new_h, new_w = int(new_h), int(new_w)
        img = m.imresize(input, (new_h, new_w))
        return img

class RandomCrop(object):

    def __init__(self, output_size):
        assert isinstance(output_size, (int, tuple))
        if isinstance(output_size, int):
            self.output_size = (output_size, output_size)
        else:
            assert len(output_size) == 2
            self.output_size = output_size

    def __call__(self, data):
        if isinstance(data, tuple):
            image, label = data
        else:
            image = data
        h, w = image.shape[:2]
        new_h, new_w = self.output_size
        top = np.random.randint(0, h - new_h)
        left = np.random.randint(0, w - new_w)
        image = image[top: top + new_h, left: left + new_w]
        if isinstance(data, tuple):
            label = label[top: top + new_h, left: left + new_w]
            return image, label
        else:
            return image