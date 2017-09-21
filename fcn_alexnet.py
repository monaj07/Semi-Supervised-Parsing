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

parser = argparse.ArgumentParser()
parser.add_argument('--dataset', required=True, help='pascal | cifar10 | lsun | imagenet | folder | lfw | fake')
parser.add_argument('--dataroot', required=True, help='path to dataset')
parser.add_argument('--splitPath', required=True, help='path to dataset splits')
parser.add_argument('--phase', required=True, help='train | val')
parser.add_argument('--cuda', action='store_true', help='enables cuda')
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
parser.add_argument('--netSaved', default='', help="path to saved net (to continue training)")
parser.add_argument('--outf', default='.', help='folder to output images and model checkpoints')
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
if opt.cuda:
    torch.cuda.manual_seed_all(opt.manualSeed)

cudnn.benchmark = True

if torch.cuda.is_available() and not opt.cuda:
    print("WARNING: You have a CUDA device, so you should probably run with --cuda")

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

class alexnet(nn.Module):

    def __init__(self, nCls=21, init_padding=100, learned_bilinear=False):
        super(alexnet, self).__init__()
        self.nCls = nCls
        self.init_padding = init_padding
        self.learned_bilinear = learned_bilinear

        self.main_body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=11, stride=4, padding=0),
            nn.BatchNorm2d(num_features=96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            nn.MaxPool2d(kernel_size=3, stride=2),  # Maxpooling is prohibited in GAN

            nn.Conv2d(96, 256, kernel_size=5, padding=2),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(3, 2),   # Maxpooling is prohibited in GAN

            nn.Conv2d(256, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(384, 384, kernel_size=3, padding=1),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(384, 256, kernel_size=3, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            nn.MaxPool2d(3, 2),   # Maxpooling is prohibited in GAN
        )

        self.classifier = nn.Sequential(
            nn.Conv2d(256, 4096, kernel_size=7),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(4096, self.nCls, kernel_size=1),
        )

        # TODO: Add support for learned upsampling
        if self.learned_bilinear:
            raise NotImplementedError
            # upscore = nn.ConvTranspose2d(self.n_classes, self.n_classes, 64, stride=32, bias=False)
            # upscore.scale_factor = None

    def forward(self, x):
        y = self.main_body(x)
        y = self.classifier(y)
        orig_size = (x.size()[2]-2*self.init_padding, x.size()[3]-2*self.init_padding)
        output = F.upsample_bilinear(y, orig_size)
        return output

net = alexnet()
net.apply(weights_init)
if opt.netSaved != '':
    net.load_state_dict(torch.load(opt.netSaved))
print(net)

if torch.cuda.is_available():
    net.cuda(0)


class padder_layer(nn.Module):
    def __init__(self, pad_size):
        super(padder_layer, self).__init__()
        self.apply_padding = nn.ConstantPad2d(pad_size, 0) # Pad with zero values
    def forward(self, x):
        output = self.apply_padding(x)
        return output
padder = padder_layer(100)
if torch.cuda.is_available():
    padder.cuda(0)

optimizer = torch.optim.SGD(net.parameters(), lr=opt.lr, momentum=0.99, weight_decay=5e-4)

for epoch in range(opt.epochs):
    for i, (images, labels) in enumerate(dataloader, 0):
        if torch.cuda.is_available():
            images = Variable(images.cuda(0))
            labels = Variable(labels.cuda(0))
        else:
            images = Variable(images)
            labels = Variable(labels)

        iter = len(dataloader) * epoch + i
        poly_lr_scheduler(optimizer, opt.lr, iter)

        optimizer.zero_grad()
        images = padder(images)
        outputs = net(images)

        loss = cross_entropy2d(outputs, labels)
        loss.backward()
        optimizer.step()

        """
        vis.line(
            X=torch.ones((1, 1)).cpu() * i,
            Y=torch.Tensor([loss.data[0]]).unsqueeze(0).cpu(),
            win=loss_window,
            update='append')
        """
        if (i + 1) % 10 == 0:
            print("Iter [%d/%d] Epoch [%d/%d] Loss: %.4f" % (i + 1, len(dataloader), epoch + 1, opt.epochs, loss.data[0]))


        # vis.image(test_image[0].cpu().data.numpy(), opts=dict(title='Input' + str(epoch)))
        # vis.image(np.transpose(target, [2,0,1]), opts=dict(title='GT' + str(epoch)))
        # vis.image(np.transpose(predicted, [2,0,1]), opts=dict(title='Predicted' + str(epoch)))

    ### Validate over the last minibatch:
    gts, preds = [], []
    pred = np.squeeze(outputs.data.max(1)[1].cpu().numpy(), axis=1)
    gt = labels.data.cpu().numpy()
    for gt_, pred_ in zip(gt, pred):
        gts.append(gt_)
        preds.append(pred_)

    score, class_iou = scores(gts, preds, n_class=NUM_CLASSES)
    for k, v in score.items():
        print k, v

    for i in range(NUM_CLASSES):
        print i, class_iou[i]
    """
    test_output = outputs[0, ...].cpu().data.numpy()
    predicted = dataset.decode_segmap(test_output.argmax(0))
    target = dataset.decode_segmap(labels[0, ...].cpu().data.numpy())
    pdb.set_trace()
    """

    torch.save(net, "{}_{}.pkl".format(opt.dataset, epoch))
