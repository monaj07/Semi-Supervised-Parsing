# Semi-Supervised Parsing #

- Implementing a semantic segmentation system that is trained using an image dataset where only a small fraction of data is annotated.

- How to run the data generator for Pascal images:
`python dcgan_pascal.py --dataset pascal --dataroot /home/monaj/bin/VOCdevkit/VOC2012/JPEGImages --cuda --niter 200 --imageSize 128 --outf ./saved`
(Note that for images of other sizes, the Discriminator and Generator networks need to be modified, (adding or removing strided convolutional layers))

- FCN training using Alexnet (no pre-trained weights used):
`python fcn_alexnet.py --dataset pascal --dataroot /home/monaj/bin/VOCdevkit/VOC2012/ --cuda --imageSize 227 --phase train --splitPath ./splits --batchSize 32`

- Training the GAN in fcn_dcgan_alexnet.py for Pascal Images:
`python fcn_dcgan_alexnet.py --dataset pascal --dataroot /home/monaj/bin/VOCdevkit/VOC2012/ --imageSize 224 --batchSize 64 --manualSeed 5069 --cuda --niter 50  --phase train --splitPath ./splits`
- If wanted to start from a recorded epoch:
`python fcn_dcgan_alexnet.py --dataset pascal --dataroot /home/monaj/bin/VOCdevkit/VOC2012/ --imageSize 224 --batchSize 64 --manualSeed 5069 --cuda --niter 50  --phase train --splitPath ./splits --netG netG_epoch_160.pth --netD netD_epoch_160.pth --net_features net_features_epoch_160.pth`
