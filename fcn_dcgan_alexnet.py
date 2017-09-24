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
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--niter', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.0002, help='learning rate, default=0.0002')
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
    dataset = Read_pascal_labeled_data(root=opt.dataroot, nCls=NUM_CLASSES, splits_path=opt.splitPath,
                                       split=opt.phase, img_size=opt.imageSize, apply_transform=True)
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
assert dataset
dataloader = torch.utils.data.DataLoader(dataset, batch_size=opt.batchSize,
                                         shuffle=True, num_workers=int(opt.workers))

ngpu = int(opt.ngpu)
nz = int(opt.nz)
ngf = int(opt.ngf)
ndf = int(opt.ndf)
nc = 3


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

    def __init__(self, supervised_pretrained_weights=False):
        super(alexnet_features, self).__init__()

        self.main_body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(num_features=96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #nn.MaxPool2d(kernel_size=3, stride=2),

            nn.Conv2d(96, 256, kernel_size=5, stride=2, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.MaxPool2d(3, 2),

            nn.Conv2d(256, 384, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.MaxPool2d(3, 2),

            nn.Conv2d(256, 4096, kernel_size=7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
        )
        self.main_body2 = nn.Sequential(

            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        output = self.main_body(x)
        output = self.main_body2(output)
        return output

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

net_features = alexnet_features()
net_features.apply(weights_init)
if opt.net_features != '':
    net_features.load_state_dict(torch.load(os.path.join(opt.outf, opt.net_features)))
print(net_features)

net_segmenter = alexnet_segmenter(opt.imageSize)
net_segmenter.apply(weights_init)
if opt.net_segmenter != '':
    net_segmenter.load_state_dict(torch.load(os.path.join(opt.outf, opt.net_segmenter)))
print(net_segmenter)


class _netG(nn.Module):
    def __init__(self, ngpu):
        super(_netG, self).__init__()
        self.ngpu = ngpu
        self.main = nn.Sequential(
            # input is Z, going into a convolution
            nn.ConvTranspose2d(     nz, ngf * 8, 7, 1, 0, bias=False),
            nn.BatchNorm2d(ngf * 8),
            nn.ReLU(True),
            # state size. (ngf*8) x 7 x 7
            nn.ConvTranspose2d(ngf * 8, ngf * 4, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 4),
            nn.ReLU(True),
            # state size. (ngf*4) x 14 x 14
            nn.ConvTranspose2d(ngf * 4, ngf * 2, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf * 2),
            nn.ReLU(True),
            # state size. (ngf*2) x 28 x 28
            nn.ConvTranspose2d(ngf * 2,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 56 x 56
            nn.ConvTranspose2d(ngf,     ngf, 4, 2, 1, bias=False),
            nn.BatchNorm2d(ngf),
            nn.ReLU(True),
            # state size. (ngf*2) x 112 x 112
        )
        self.main2 = nn.Sequential(
            nn.ConvTranspose2d(ngf, nc, 4, 2, 1, bias=False),
            nn.Tanh()
            # state size. (nc) x 224 x 224
        )

    def forward(self, input):
        if isinstance(input.data, torch.cuda.FloatTensor) and self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            output = self.main2(output)
        return output


netG = _netG(ngpu)
netG.apply(weights_init)
if opt.netG != '':
    netG.load_state_dict(torch.load(os.path.join(opt.outf, opt.netG)))
print(netG)



class _netD(nn.Module):
    def __init__(self, ngpu=1):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.classifier= nn.Sequential(
            nn.Conv2d(4096, 1, 1), # Is this the right way, or we'd better use a linear layer???
            # The data shape is now (,1)
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.classifier(x)
        return y.view(-1, 1).squeeze(1)

netD = _netD(ngpu)
netD.apply(weights_init)
if opt.netD != '':
    netD.load_state_dict(torch.load(os.path.join(opt.outf, opt.netD)))
print(netD)

class padder_layer(nn.Module):
    def __init__(self, pad_size):
        super(padder_layer, self).__init__()
        self.apply_padding = nn.ConstantPad2d(pad_size, 0) # Pad with zero values
    def forward(self, x):
        output = self.apply_padding(x)
        return output
padder = padder_layer(100)

criterion = nn.BCELoss()

input = torch.FloatTensor(opt.batchSize, 3, opt.imageSize, opt.imageSize)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
label = torch.FloatTensor(opt.batchSize)

real_label = .9
fake_label = .1

if opt.cuda:
    net_features.cuda()
    net_segmenter.cuda()
    padder.cuda()
    netD.cuda()
    netG.cuda()
    criterion.cuda()
    input, label = input.cuda(), label.cuda()
    noise, fixed_noise = noise.cuda(), fixed_noise.cuda()

fixed_noise = Variable(fixed_noise)

# setup optimizer
optimizerS = optim.Adam(list(net_features.parameters())+list(net_segmenter.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerD = optim.Adam(list(net_features.parameters())+list(netD.parameters()), lr=opt.lr, betas=(opt.beta1, 0.999))
optimizerG = optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

SS = 0
GAN = 1

for epoch in range(opt.niter):
    for i, data in enumerate(dataloader, 0):

        real_cpu, label_cpu = data
        batch_size = real_cpu.size(0)
        if opt.cuda:
            real_cpu = real_cpu.cuda()
            label_cpu = label_cpu.cuda()
        input.resize_as_(real_cpu).copy_(real_cpu)
        inputv = Variable(input)
        labelv_semantic = Variable(label_cpu)

        if GAN:
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            netD.zero_grad()
            net_features.zero_grad()

            # train with real
            label.resize_(batch_size).fill_(real_label)
            labelv = Variable(label)
            inputv_feats = net_features(inputv)
            output = netD(inputv_feats)
            errD_real = criterion(output, labelv)
            errD_real.backward()
            D_x = output.data.mean()

            # train with fake
            noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            noisev = Variable(noise)
            fake = netG(noisev)
            labelv = Variable(label.fill_(fake_label))
            fake_feats = net_features(fake.detach())
            output = netD(fake_feats)
            errD_fake = criterion(output, labelv)
            errD_fake.backward()
            D_G_z1 = output.data.mean()

            errD = errD_real + errD_fake
            optimizerD.step()

            ############################
            # (2) Update G network: maximize log(D(G(z)))
            ###########################
            netG.zero_grad()

            label.resize_(batch_size).fill_(real_label)
            labelv = Variable(label.fill_(real_label))  # fake labels are real for generator cost
            noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            noise2v = Variable(noise)
            fake2 = netG(noise2v)
            fake_feats2 = net_features(fake2)
            output = netD(fake_feats2)
            errG = criterion(output, labelv)
            errG.backward()
            D_G_z2 = output.data.mean()
            optimizerG.step()

        if SS:
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
        ######################################################

        if GAN:
            if (i + 1) % 10 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, opt.niter, i+1, len(dataloader),
                         errD.data[0], errG.data[0], D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(real_cpu,
                        '%s/real_samples.png' % opt.outf,
                        normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.data,
                        '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch),
                        normalize=True)
        if (SS):
            if (i + 1) % 10 == 0:
                print("Iter [%d/%d] Epoch [%d/%d] Loss: %.4f" % (i + 1, len(dataloader), epoch + 1, opt.niter, loss.data[0]))

    ######################################################
    ### Evaluation of the semantic segmentation:
    ######################################################
    if SS:
        ### Validate only over the last processed minibatch:
        gts, preds = [], []
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
        torch.save(netG.state_dict(), '%s/netG_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(netD.state_dict(), '%s/netD_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(net_features.state_dict(), '%s/net_features_epoch_%d.pth' % (opt.outf, epoch))
        torch.save(net_segmenter.state_dict(), '%s/net_segmenter_epoch_%d.pth' % (opt.outf, epoch))