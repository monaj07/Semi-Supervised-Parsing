# Semi-Supervised Parsing #

- Implementing a semantic segmentation system that is trained using an image dataset where only a small fraction of data is annotated.

- How to run the data generator for Pascal images:
`python dcgan_alexnet.py --dataset pascal --dataroot /home/monaj/bin/VOCdevkit/VOC2012/JPEGImages --cuda --niter 50`

- Or for images with higher resolution (requirs a modification in the Discriminator and Generator definitions):
`python dcgan_alexnet.py --dataset pascal --dataroot /home/monaj/bin/VOCdevkit/VOC2012/JPEGImages --cuda --niter 200 --imageSize 128 --outf ./saved`