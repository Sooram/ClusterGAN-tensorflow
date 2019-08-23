# ClusterGAN-tensorflow
Tensorflow implementation of ClusterGAN(MNIST example)

## Reference 
Slightly modified codes from: https://github.com/sudiptodip15/ClusterGAN

## Original paper
[ClusterGAN : Latent Space Clustering in Generative Adversarial Networks](https://arxiv.org/pdf/1809.03627.pdf)

## Loss function
1. Adversarial loss\
Implemented in WGAN-GP

2. Clustering specific loss\
(to be added)

3. Training details\
(to be added)

## Results
(to be modified)
Successful case(mode is given by 0):\
![Overview](https://github.com/Sooram/ClusterGAN-tensorflow/blob/master/res_mode0_label6.PNG)
When the mode is given by 2, which needs to generate label 0 images, the results were not good.\
![Overview](https://github.com/Sooram/ClusterGAN-tensorflow/blob/master/res_mode2_label0.PNG)

## Comments
(to be modified)
- Very similar concept with InfoGAN
- Training needed quite a lot of time(did not converged well)
- From my experiment, the results were not that impressive.(Maybe my code...? or not sufficiently trained...? I need to check again.)
