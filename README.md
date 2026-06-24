# Image Classification at the Edge

This repository contains the code used to train and deploy the models on the following platforms:
- STM32U5
- MAX78000
- Coral Micro (TPU)

Note that you may find some files which refer to cifar400k. These files actually refer to the 300 kB model, the same one as in the presentation. Unfortunately, the first iteration of this model contained
a 3 x 3 convolution which resulted in a model of around 380 kB, but which could not be deployed on the MAX78000. I later changed that layer to be a 1 x 1 convolution resulting in a model of around 310 kB.
This is the reason for the name mismatch.
