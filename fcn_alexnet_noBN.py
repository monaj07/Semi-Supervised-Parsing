from __future__ import print_function
import argparse
import os
import random

from read_pascal_images import *
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim as optim
import torch.utils.data
import torchvision.datasets as dset
import torchvision.models as models
import torchvision.transforms as transforms
import torchvision.utils as vutils
from torch.autograd import Variable

from lr_scheduling import *
from metric import scores

import pdb


parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=1, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate, default=0.0002')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--outf', default='./saved', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')
parser.add_argument('--splitPath', required=True, help='path to dataset splits')
parser.add_argument('--phase', required=True, help='train | val')
parser.add_argument('--net_features', default='', help="path to feature_net (to continue training)")
parser.add_argument('--net_segmenter', default='', help="path to segmenter_net (to continue training)")
parser.add_argument('--gpu', type=int, default=0, help='which GPU to use')

opt = parser.parse_args()
print(opt)

try:
    os.makedirs(opt.outf)
except OSError:
    pass

if opt.manualSeed is None:
    opt.manualSeed = random.randint(1, 10000)
print("Random Seed: ", opt.manualSeed)
random.seed(opt.manualSeed)
torch.manual_seed(opt.manualSeed)
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

if opt.dataset in ['imagenet', 'folder', 'lfw']:
    # folder dataset
    dataset = dset.ImageFolder(root=opt.dataroot,
                               transform=transforms.Compose([
                                   transforms.Scale(opt.imageSize),
                                   transforms.CenterCrop(opt.imageSize),
                                   transforms.ToTensor(),
                                   transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                               ]))
elif opt.dataset == 'pascal':
    NUM_CLASSES = 21
    dataset = Read_pascal_labeled_data_minus_imagenet(root=opt.dataroot, nCls=NUM_CLASSES, splits_path=opt.splitPath,
                                       split=opt.phase, img_size=opt.imageSize, apply_transform=True)
    dataset_val = Read_pascal_labeled_data_minus_imagenet(root=opt.dataroot, nCls=NUM_CLASSES, splits_path=opt.splitPath,
                                       split='val', img_size=opt.imageSize, apply_transform=True)
elif opt.dataset == 'lsun':
    dataset = dset.LSUN(db_path=opt.dataroot, classes=['bedroom_train'],
                        transform=transforms.Compose([
                            transforms.Scale(opt.imageSize),
                            transforms.CenterCrop(opt.imageSize),
                            transforms.ToTensor(),
                            transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                        ]))
elif opt.dataset == 'cifar10':
    dataset = dset.CIFAR10(root=opt.dataroot, download=True,
                           transform=transforms.Compose([
                               transforms.Scale(opt.imageSize),
                               transforms.ToTensor(),
                               transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
                           ]))
elif opt.dataset == 'fake':
    dataset = dset.FakeData(image_size=(3, opt.imageSize, opt.imageSize),
                            transform=transforms.ToTensor())
assert (dataset and dataset_val)
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))
dataloader_val = torch.utils.data.DataLoader(dataset_val, batch_size=opt.batchSize,
                                         shuffle=False, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)


# custom weights initialization called on netG and netD
def weights_init(m):
    classname = m.__class__.__name__
    if classname.find('Conv') != -1:
        m.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        m.weight.data.normal_(1.0, 0.02)
        m.bias.data.fill_(0)


def cross_entropy2d(input, target, weight=None, size_average=True):
    # https://github.com/meetshah1995/pytorch-semseg/blob/master/ptsemseg/loss.py
    n, c, h, w = input.size()
    log_p = F.log_softmax(input)
    log_p = log_p.transpose(1, 2).transpose(2, 3).contiguous().view(-1, c)
    log_p = log_p[target.view(n, h, w, 1).repeat(1, 1, 1, c) >= 0]
    log_p = log_p.view(-1, c)

    mask = target >= 0
    target = target[mask]
    loss = F.nll_loss(log_p, target, weight=weight, size_average=False)
    if size_average:
        loss /= mask.data.sum()
    return loss

class alexnet_features(nn.Module):

    def __init__(self):
        super(alexnet_features, self).__init__()

        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=11, stride=4, padding=0),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(64, 192, kernel_size=5, stride=1, padding=2),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(192, 384, kernel_size=3, stride=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(384, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),

            nn.Conv2d(256, 256, kernel_size=3, stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, ceil_mode=True),
        )

        self.classifier = nn.Sequential(
            nn.Dropout2d(),
            nn.Conv2d(256, 4096, kernel_size=6, stride=1),
            nn.ReLU(inplace=True),

            nn.Dropout2d(),
            nn.Conv2d(4096, 4096, kernel_size=1, stride=1),
            nn.ReLU(inplace=True),
        )

    def forward(self, x):
        output = self.features(x)
        output = self.classifier(output)
        return output

    def init_pretrained_params(self):
        network = models.alexnet(pretrained=True)
        features = list(network.features.children())
        for l1, l2 in zip(features, self.features):
            if isinstance(l1, nn.Conv2d) and isinstance(l2, nn.Conv2d):
                # print idx, l1, l2
                assert l1.weight.size() == l2.weight.size()
                assert l1.bias.size() == l2.bias.size()
                l2.weight.data = l1.weight.data
                l2.bias.data = l1.bias.data
        for i1, i2 in zip([1, 4], [1, 4]):
            l1 = network.classifier[i1]
            l2 = self.classifier[i2]
            # print type(l1), dir(l1),
            l2.weight.data = l1.weight.data.view(l2.weight.size())
            l2.bias.data = l1.bias.data.view(l2.bias.size())

class alexnet_segmenter(nn.Module):
    """
    feature map to segmentation map:
    """
    def __init__(self, orig_size, nCls=21, learned_bilinear=False):
        super(alexnet_segmenter, self).__init__()
        assert (orig_size is not None)
        self.orig_size = orig_size
        self.nCls = nCls
        self.learned_bilinear = learned_bilinear

        self.classifier= nn.Sequential(
            nn.Conv2d(4096, self.nCls, kernel_size=1),
        )

        # TODO: Add support for learned upsampling
        if self.learned_bilinear:
            raise NotImplementedError
            # upscore = nn.ConvTranspose2d(self.n_classes, self.n_classes, 64, stride=32, bias=False)
            # upscore.scale_factor = None

    def forward(self, x):
        y = self.classifier(x)
        if isinstance(self.orig_size, int):
            self.orig_size = (self.orig_size, self.orig_size)
        output = F.upsample_bilinear(y, self.orig_size)
        return output

    def init_pretrained_params(self): # Copying imagenet 1000-way classification weights
        network = models.alexnet(pretrained=True)
        l1 = network.classifier[6]
        l2 = self.classifier[0]
        l2.weight.data = l1.weight.data[:self.nCls, :].view(l2.weight.size())
        l2.bias.data = l1.bias.data[:self.nCls]

net_features = alexnet_features()
#net_features.apply(weights_init)
net_features.init_pretrained_params()
if opt.net_features != '':
    net_features.load_state_dict(torch.load(os.path.join(opt.outf, opt.net_features)))
print(net_features)

net_segmenter = alexnet_segmenter(opt.imageSize)
net_segmenter.apply(weights_init)
#net_segmenter.init_pretrained_params()
if opt.net_segmenter != '':
    net_segmenter.load_state_dict(torch.load(os.path.join(opt.outf, opt.net_segmenter)))
print(net_segmenter)


class padder_layer(nn.Module):
    def __init__(self, pad_size):
        super(padder_layer, self).__init__()
        self.apply_padding = nn.ConstantPad2d(pad_size, 0) # Pad with zero values
    def forward(self, x):
        output = self.apply_padding(x)
        return output
padder = padder_layer(100)

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
label = torch.FloatTensor(opt.batchSize)

gpu = opt.gpu

if opt.cuda:
    net_features.cuda(gpu)
    net_segmenter.cuda(gpu)
    padder.cuda(gpu)
    input, label = input.cuda(gpu), label.cuda(gpu)

# setup optimizer
optimizerS = optim.Adam(list(net_features.parameters())+list(net_segmenter.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
#optimizerS = optim.SGD(list(net_features.parameters())+list(net_segmenter.parameters()), lr=opt.lr, momentum=0.99, weight_decay=5e-4)


for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):

        real_cpu, label_cpu = data
        if opt.cuda:
            real_cpu = real_cpu.cuda(gpu)
            label_cpu = label_cpu.cuda(gpu)
        input.resize_as_(real_cpu).copy_(real_cpu)
        inputv = Variable(input)
        labelv_semantic = Variable(label_cpu)

        iter = len(dataloader) * epoch + i

        poly_lr_scheduler(optimizerS, opt.lr, iter)

        ############################
        # Update semantic labeling network
        ###########################
        net_features.zero_grad()
        net_segmenter.zero_grad()

        images = padder(inputv)
        feature_maps = net_features(images)
        outputs = net_segmenter(feature_maps)
        loss = cross_entropy2d(outputs, labelv_semantic)
        loss.backward()

        optimizerS.step()

        ######################################################
        ### Loss Report:

        if (i + 1) % 10 == 0:
            print("Iter [%d/%d] Epoch [%d/%d] Loss: %.4f" % (i + 1, len(dataloader), epoch + 1, opt.niter, loss.data[0]))

    ######################################################
    ### Validation of the semantic segmentation module:
    ######################################################

    gts, preds = [], []
    for j, data_val in enumerate(dataloader_val, 0):
        real_cpu, label_cpu = data_val
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda(gpu)
            label_cpu = label_cpu.cuda(gpu)
        input.resize_as_(real_cpu).copy_(real_cpu)
        inputv = Variable(input)
        labelv_semantic = Variable(label_cpu)

        images = padder(inputv)
        feature_maps = net_features(images)
        outputs = net_segmenter(feature_maps)

        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=1)
        gt = labelv_semantic.data.cpu().numpy()
        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)

    print('\n' + '-' * 40)
    score, class_iou = scores(gts, preds, n_class=NUM_CLASSES)
    for k, v in score.items():
        print(k, v)
    print('-' * 20)
    for i in range(NUM_CLASSES):
        print(i, class_iou[i])
    print('-' * 40 + '\n')

    # do checkpointing
    if (epoch%10==0) and (epoch>0):
        torch.save(net_features.state_dict(), '%s/net_features_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(net_segmenter.state_dict(), '%s/net_segmenter_epoch_%d.pth' % (opt.outf, epoch))