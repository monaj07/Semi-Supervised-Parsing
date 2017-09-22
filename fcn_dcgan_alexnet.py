import sys
import torch
import visdom
import argparse
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
import torch.backends.cudnn as cudnn
import os
from torch.autograd import Variable
from torch.utils import data
import random
# import pdb
from read_pascal_images import *
from lr_scheduling import *
from metric import scores
import torchvision.utils as vutils

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='pascal | cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--splitPath', required=True, help='path to dataset splits')
parser.add_argument('--phase', required=True, help='train | val')
parser.add_argument('--workers', type=int, help='number of data loading workers', default=2)
parser.add_argument('--batchSize', type=int, default=64, help='input batch size')
parser.add_argument('--imageSize', type=int, default=64, help='the height / width of the input image to network')
parser.add_argument('--nz', type=int, default=100, help='size of the latent z vector')
parser.add_argument('--ngf', type=int, default=64)
parser.add_argument('--ndf', type=int, default=64)
parser.add_argument('--epochs', type=int, default=25, help='number of epochs to train for')
parser.add_argument('--lr', type=float, default=0.00001, help='learning rate, default=0.00001')
parser.add_argument('--beta1', type=float, default=0.5, help='beta1 for adam. default=0.5')
parser.add_argument('--ngpu', type=int, default=1, help='number of GPUs to use')
parser.add_argument('--netG', default='', help="path to netG (to continue training)")
parser.add_argument('--netD', default='', help="path to netD (to continue training)")
parser.add_argument('--netSaved', default='', help="path to saved net (to continue training)")
parser.add_argument('--outf', default='./saved', help='folder to output images and model checkpoints')
parser.add_argument('--manualSeed', type=int, help='manual seed')

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
if torch.cuda.is_available():
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

NUM_CLASSES = 21
if opt.dataset == 'pascal':
    dataset = Read_pascal_labeled_data(root=opt.dataroot, nCls=21, splits_path=opt.splitPath,
                                       split=opt.phase, img_size=opt.imageSize, apply_transform=True)
else:
    raise NotImplementedError
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
if opt.netSaved != '':
    net_features.load_state_dict(torch.load(opt.netSaved))
print(net_features)

net_segmenter = alexnet_segmenter(opt.imageSize)
net_segmenter.apply(weights_init)
if opt.netSaved != '':
    net_segmenter.load_state_dict(torch.load(opt.netSaved))
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
    netG.load_state_dict(torch.load(opt.netG))
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
    netD.load_state_dict(torch.load(opt.netD))
print(netD)

criterion_GAN = nn.BCELoss()
fixed_noise = torch.FloatTensor(opt.batchSize, nz, 1, 1).normal_(0, 1)
noise = torch.FloatTensor(opt.batchSize, nz, 1, 1)
labels_GAN = torch.FloatTensor(opt.batchSize)
real_label = .7
fake_label = 0.3

class padder_layer(nn.Module):
    def __init__(self, pad_size):
        super(padder_layer, self).__init__()
        self.apply_padding = nn.ConstantPad2d(pad_size, 0) # Pad with zero values
    def forward(self, x):
        output = self.apply_padding(x)
        return output

padder = padder_layer(100)
if torch.cuda.is_available():
    net_features.cuda(0)
    net_segmenter.cuda(0)
    netD.cuda(0)
    netG.cuda(0)
    padder.cuda(0)
    noise, fixed_noise = noise.cuda(0), fixed_noise.cuda(0)
fixed_noise = Variable(fixed_noise)

optimizer_segmentation = torch.optim.SGD(list(net_features.parameters())+list(net_segmenter.parameters()), lr=opt.lr, momentum=0.99, weight_decay=5e-4)
optimizerD             = torch.optim.SGD(list(net_features.parameters())+list(netD.parameters()), lr=opt.lr, momentum=0.99, weight_decay=5e-4)
optimizerG             = torch.optim.Adam(netG.parameters(), lr=opt.lr, betas=(opt.beta1, 0.999))

SS = 0
GAN = 1

for epoch in range(opt.epochs):
    for i, (images, labels) in enumerate(dataloader, 0):
        if torch.cuda.is_available():
            images = images.cuda(0)
            labels = labels.cuda(0)
            labels_GAN = labels_GAN.cuda(0)
            noise = noise.cuda(0)

        images = Variable(images)
        labels = Variable(labels)

        iter = len(dataloader) * epoch + i

        poly_lr_scheduler(optimizer_segmentation, opt.lr, iter)

        if (GAN):
            ############################
            # (1) Update D network: maximize log(D(x)) + log(1 - D(G(z)))
            ###########################
            # train the GAN discriminator with real images
            batch_size = images.size()[0]
            optimizerD.zero_grad()
            feature_maps = net_features(images)
            outputD_real = netD(feature_maps)
            labels_GAN.resize_(batch_size).fill_(real_label)
            labels_GAN_var = Variable(labels_GAN)
            lossD_real = criterion_GAN(outputD_real, labels_GAN_var)
            lossD_real.backward()
            D_x = outputD_real.data.mean()

            # train the GAN discriminator with fake images
            noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
            noise_var = Variable(noise)
            fake_images = netG(noise_var)
            feature_maps = net_features(fake_images)
            outputD_fake = netD(feature_maps.detach())
            labels_GAN_var = Variable(labels_GAN.fill_(fake_label))
            lossD_fake = criterion_GAN(outputD_fake, labels_GAN_var)
            lossD_fake.backward()
            D_G_z1 = outputD_fake.data.mean()
            lossD = lossD_real + lossD_fake
            optimizerD.step()

            if i % 5 == 0:
                ############################
                # (2) Update G network: maximize log(D(G(z)))
                ###########################
                optimizerG.zero_grad()
                labels_GAN.resize_(batch_size).fill_(real_label)
                labels_GAN_var = Variable(labels_GAN.fill_(real_label))  # fake labels are real for generator cost
                noise.resize_(batch_size, nz, 1, 1).normal_(0, 1)
                noise_var = Variable(noise)
                fake_images2 = netG(noise_var)
                feature_maps = net_features(fake_images2)
                outputD_fake_real_label = netD(feature_maps)
                lossG = criterion_GAN(outputD_fake_real_label, labels_GAN_var)
                lossG.backward()
                D_G_z2 = outputD_fake_real_label.data.mean()
                optimizerG.step()


        if (SS):
            ############################
            # Update semantic labeling network
            ###########################
            optimizer_segmentation.zero_grad()

            images = padder(images)
            feature_maps = net_features(images)
            outputs = net_segmenter(feature_maps)

            loss = cross_entropy2d(outputs, labels)
            loss.backward()

            optimizer_segmentation.step()



            """
            vis.line(
                X=torch.ones((1, 1)).cpu() * i,
                Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
                win=loss_window,
                update='append')
            """

        if (GAN):
            if i % 10 == 0:
                print('[%d/%d][%d/%d] Loss_D: %.4f Loss_G: %.4f D(x): %.4f D(G(z)): %.4f / %.4f'
                      % (epoch, opt.epochs, i, len(dataloader),
                         lossD.data[0], lossG.data[0], D_x, D_G_z1, D_G_z2))
            if i % 100 == 0:
                vutils.save_image(images.data.cpu(), '%s/real_samples.png' % opt.outf, normalize=True)
                fake = netG(fixed_noise)
                vutils.save_image(fake.data, '%s/fake_samples_epoch_%03d.png' % (opt.outf, epoch), normalize=True)

        if (SS):
            if (i + 1) % 10 == 0:
                print("Iter [%d/%d] Epoch [%d/%d] Loss: %.4f" % (i + 1, len(dataloader), epoch + 1, opt.epochs, loss.data[0]))


        # vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
        # vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
        # vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))

    if (SS):
        ### Validate over the last minibatch:
        gts, preds = [], []
        pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=1)
        gt = labels.data.cpu().numpy()
        for gt_, pred_ in zip(gt, pred):
            gts.append(gt_)
            preds.append(pred_)

        print('\n' + '-' * 40)
        score, class_iou = scores(gts, preds, n_class=NUM_CLASSES)
        for k, v in score.items():
            print k, v
        print('-' * 20)
        for i in range(NUM_CLASSES):
            print i, class_iou[i]
        print('-' * 40 + '\n')

    """
    test_output = outputs[0, ...].cpu().data.numpy()
    predicted = dataset.decode_segmap(test_output.argmax(0))
    target = dataset.decode_segmap(labels[0, ...].cpu().data.numpy())
    pdb.set_trace()
    """

    torch.save([net_features, net_segmenter], "{}_{}.pkl".format(opt.dataset, epoch))
