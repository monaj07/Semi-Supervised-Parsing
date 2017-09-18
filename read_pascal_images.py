
from torch.utils.data import Dataset
import os
import fnmatch
#import scipy.misc as m
from skimage import io
from torchvision import transforms

import pdb

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

#pascal_data = Read_pascal_images('/home/monaj/bin/VOCdevkit/VOC2012/JPEGImages')
