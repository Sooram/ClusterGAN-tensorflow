# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 10:01:30 2019

@author: Sooram Kang

"""
import os
import argparse
import tensorflow as tf
from clustergan import ClusterGAN, train, inference

parser = argparse.ArgumentParser(description='ClusterGAN')
parser.add_argument('--model_dir', type=str, 
                      default='./exp2',
                      help='Directory in which the model is stored')
#parser.add_argument('--data_dir', type=str,
#                      default='../data',
#                      help='Directory in which the data is stored')
parser.add_argument('--is_training', type=bool, default=True, help='whether it is training or inferecing')
parser.add_argument('--n_cat', type=int, default=1, help='number of categorical variables')
parser.add_argument('--num_classes', type=int, default=10, help='dimension of categorical variables')
parser.add_argument('--dim_gen', type=int, default=30, help='continuous dim of latent variable')
parser.add_argument('--z_dim', type=int, default=40, help='random noise dim of latent variable')
parser.add_argument('--sampler', type=str, default='one_hot')
parser.add_argument('--batch_size', type=int, default=64, help='batch size')
parser.add_argument('--epoch', type=int, default=5000, help='epochs')
parser.add_argument('--saving_cycle', type=int, default=1, help='how often the model will be saved')
parser.add_argument('--d_lr', type=float, default=1e-4, help='learning rate for discriminator')
parser.add_argument('--g_lr', type=float, default=1e-4, help='learning rate for generator')
parser.add_argument('--gpu_num', type=str, default="1", help='gpu to be used')

        
#%%
def main(args):
    # build model 
    model = ClusterGAN(args)
    
    # open session 
#     c = tf.ConfigProto()
#     c.gpu_options.visible_device_list = args.gpu_num
#     sess = tf.Session(config=c)

    sess = tf.Session(config=tf.ConfigProto(allow_soft_placement=True))#, log_device_placement=True))
    sess.run(tf.global_variables_initializer())

    train(args, model, sess) if args.is_training else inference(args, model, sess)


    
if __name__ == '__main__':
    args, unparsed = parser.parse_known_args()
    os.environ["CUDA_DEVICE_ORDER"]="PCI_BUS_ID"   
    os.environ["CUDA_VISIBLE_DEVICES"]=args.gpu_num
    main(args)







