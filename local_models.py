import torch
import torch.nn as nn
import torch.functional as F

class alexnet_features(nn.Module):

    def __init__(self, supervised_pretrained_weights=False):
        super(alexnet_features, self).__init__()

        self.main_body = nn.Sequential(
            nn.Conv2d(in_channels=3, out_channels=96, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(num_features=96),
            nn.LeakyReLU(negative_slope=0.2, inplace=True),

            #nn.MaxPool2d(kernel_size=3, stride=2),                      # Maxpooling is prohibited in GAN
            nn.Conv2d(96, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),

            #nn.MaxPool2d(3, 2),                                         # Maxpooling is prohibited in GAN
            nn.Conv2d(256, 384, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True),

            nn.Conv2d(384, 384, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(384),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Conv2d(384, 256, kernel_size=4, stride=2, padding=1),
            nn.BatchNorm2d(256),
            nn.LeakyReLU(0.2, inplace=True),
            #nn.MaxPool2d(3, 2),                                         # Maxpooling is prohibited in GAN

            nn.Conv2d(256, 4096, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),

            nn.Conv2d(4096, 4096, kernel_size=1),
            nn.LeakyReLU(0.2, inplace=True),
            nn.Dropout(0.5),
        )

    def forward(self, x):
        output = self.main_body(x)
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


class _netG(nn.Module):
    def __init__(self, ngpu, ngf, nz, nc):
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


class _netD(nn.Module):
    def __init__(self, ngpu=1):
        super(_netD, self).__init__()
        self.ngpu = ngpu
        self.classifier= nn.Sequential(
            nn.Conv2d(4096, 1, 7), # Is this the right way, or we'd better use a linear layer???
            # The data shape is now (,1)
            nn.Sigmoid()
        )

    def forward(self, x):
        y = self.classifier(x)
        return y.view(-1, 1).squeeze(1)
