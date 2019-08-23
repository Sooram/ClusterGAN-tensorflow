# -*- coding: utf-8 -*-
"""
Created on Mon Jun 10 09:34:55 2019

@author: Sooram Kang

"""
import os
#import cv2
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from general_model import Model
from utils import get_batch, sample_Z
from tensorflow.examples.tutorials.mnist import input_data


img = {
    "w": 28,
    "h": 28,
    "c": 1
}


class ClusterGAN(Model):
    def __init__(self, args):

        #########################
        #                       #
        #    General Setting    #
        #                       #
        #########################

        self.args = args

        self.model_dir = args.model_dir

        if not self.model_dir:
            raise ValueError('Need to provide model directory')

        self.log_dir = os.path.join(self.model_dir, 'log')
        self.test_dir = os.path.join(self.model_dir, 'test')

        if not os.path.exists(self.model_dir):
            os.makedirs(self.model_dir)

        if not os.path.exists(self.log_dir):
            os.makedirs(self.log_dir)

        if not os.path.exists(self.test_dir):
            os.makedirs(self.test_dir)

        self.global_step = tf.train.get_or_create_global_step()
        
        #########################
        #                       #
        #     Model Building    #
        #                       #
        #########################
                
        net_G = Generator()
        net_D = Discriminator()
        net_E = Encoder()

        # 1. Build Generator
        # Create latent variable
        with tf.name_scope('noise_sample'):
            self.z = tf.placeholder(tf.float32, [None, args.z_dim], name='z')

            self.z_gen = self.z[:,0:args.dim_gen]
            self.z_hot = self.z[:,args.dim_gen:]


        self.x_ = net_G(self.z, args.is_training)
        self.z_enc_gen, self.z_enc_label, self.z_enc_logits = net_E(self.x_, args, reuse=False)
        
        # 2. Build Discriminator
        # Real Data
        with tf.name_scope('data_and_target'):
            self.x = tf.placeholder(tf.float32, [None, img['w'], img['h'], img['c']])


        self.d = net_D(self.x, args.is_training, reuse=False)
        self.d_ = net_D(self.x_, args.is_training)
        self.z_infer_gen, self.z_infer_label, self.z_infer_logits = net_E(self.x, args)


        # 3. Calculate loss
        with tf.name_scope('loss'):
            self.beta_cycle_gen = 10.0
            self.beta_cycle_label = 10.0
            self.g_loss = tf.reduce_mean(self.d_) + \
                  self.beta_cycle_gen * tf.reduce_mean(tf.square(self.z_gen - self.z_enc_gen)) +\
                  self.beta_cycle_label * tf.reduce_mean(
                      tf.nn.softmax_cross_entropy_with_logits(logits=self.z_enc_logits,labels=self.z_hot))
                  
            self.d_loss = tf.reduce_mean(self.d) - tf.reduce_mean(self.d_)
            
            # WGAN-GP
            epsilon = tf.random_uniform([], 0.0, 1.0)
            x_hat = epsilon * self.x + (1 - epsilon) * self.x_
            d_hat = net_D(x_hat, args.is_training)
    
            ddx = tf.gradients(d_hat, x_hat)[0]
            ddx = tf.sqrt(tf.reduce_sum(tf.square(ddx), axis=1))
            scale = 10.0
            ddx = tf.reduce_mean(tf.square(ddx - 1.0) * scale)
    
            self.d_loss = self.d_loss + ddx
                  

        # 4. Update weights
        g_param = tf.trainable_variables(scope='generator')
        d_param = tf.trainable_variables(scope='discriminator')
        e_param = tf.trainable_variables(scope='encoder')


        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            g_optim = tf.train.AdamOptimizer(learning_rate=args.g_lr, beta1=0.5, beta2=0.9)
            self.g_train_op = g_optim.minimize(self.g_loss, var_list=[g_param,e_param], global_step=self.global_step)
            d_optim = tf.train.AdamOptimizer(learning_rate=args.d_lr, beta1=0.5, beta2=0.9)
            self.d_train_op = d_optim.minimize(self.d_loss, var_list=d_param)

            
        # 5. Visualize
        tf.summary.image('Real', self.x)
        tf.summary.image('Fake', self.x_)

        with tf.name_scope('All_Loss'):
            tf.summary.scalar('g_loss', self.g_loss)
            tf.summary.scalar('d_loss', self.d_loss)
            tf.summary.scalar('zn_distance', self.beta_cycle_gen * tf.reduce_mean(tf.square(self.z_gen - self.z_enc_gen)))
            tf.summary.scalar('cat_loss', self.beta_cycle_label * tf.reduce_mean(
                       tf.nn.softmax_cross_entropy_with_logits(logits=self.z_enc_logits,labels=self.z_hot)))
            tf.summary.scalar('ddx', ddx)
            tf.summary.scalar('dx', tf.reduce_mean(self.d))
            tf.summary.scalar('dx_', -tf.reduce_mean(self.d_))

        self.summary_op = tf.summary.merge_all()
        
        super(ClusterGAN, self).__init__(5)


class Generator(object):
    def __call__(self, noise, is_training):
        with tf.variable_scope('generator'):
            output = tf.layers.dense(noise, 1024, kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5))
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
            output = tf.layers.dense(output, 7*7*128, kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5))
            output = tf.reshape(output, [-1, 7, 7, 128])
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
            output = tf.layers.conv2d_transpose(output, 64, [4, 4], strides=(2, 2), padding='SAME',
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5))
            output = tf.nn.relu(tf.layers.batch_normalization(output, training=is_training))
            output = tf.layers.conv2d_transpose(output, 1, [4, 4], strides=(2,2), padding='SAME',
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5))
            output = tf.sigmoid(output)

        return output

def leaky_relu(x, leak=0.2):
    return tf.maximum(tf.minimum(0.0, leak * x), x)
                
class Discriminator(object):
    def __call__(self, inputs, is_training, reuse=True):
        with tf.variable_scope('discriminator') as scope:
            if reuse:
                scope.reuse_variables()
            output = tf.layers.conv2d(inputs, filters=64, kernel_size=[4, 4], strides=(2, 2), padding='SAME',
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            output = leaky_relu(output)
            output = tf.layers.conv2d(output, filters=128, kernel_size=[4, 4], strides=(2, 2), padding='SAME',
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            output = leaky_relu(output)

            flat = tf.contrib.layers.flatten(output)
            d_out = tf.layers.dense(flat, 1024, kernel_initializer=tf.random_normal_initializer(stddev=0.02))
            d_out = leaky_relu(d_out)
            d_out = tf.layers.dense(d_out, 1, activation=tf.identity)

        return d_out

class Encoder(object):
    def __call__(self, inputs, args, reuse=True):
        with tf.variable_scope('encoder') as scope:
            if reuse:
                scope.reuse_variables()
            output = tf.layers.conv2d(inputs, filters=64, kernel_size=[4, 4], strides=(2, 2), padding='SAME',
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5))
            output = leaky_relu(output)
            output = tf.layers.conv2d(output, filters=128, kernel_size=[4, 4], strides=(2, 2), padding='SAME',
                                    kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5))
            output = leaky_relu(output)

            flat = tf.contrib.layers.flatten(output)
            out = tf.layers.dense(flat, 1024, kernel_initializer=tf.random_normal_initializer(stddev=0.02),
                                    kernel_regularizer=tf.contrib.layers.l2_regularizer(2.5e-5))
            out = leaky_relu(out)
            out = tf.layers.dense(out, args.z_dim, activation=tf.identity)

            logits = out[:, args.dim_gen:]
            y = tf.nn.softmax(logits)
            
        return out[:, 0:args.dim_gen], y, logits

#%%
    
def train(args, model, sess):
    summary_writer = tf.summary.FileWriter(model.log_dir, sess.graph)
    
    # load data
    mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)

    # load previous model
    model.load(sess, args.model_dir)  

    steps_per_epoch = mnist.train.labels.shape[0] // args.batch_size
    
    for epoch in range(args.epoch):

        epoch_loss_d = []
        epoch_loss_g = []
        
        for step in range(steps_per_epoch):
            
            # train discriminator for d_iters times first
            d_iters = 5
            for _ in range(0, d_iters):
                bx, _ = mnist.train.next_batch(args.batch_size)
                bx = np.reshape(bx, [-1, img['w'], img['h'], img['c']])                
                bz = sample_Z(args.batch_size, args.z_dim, args.sampler, args.num_classes, args.n_cat)
                sess.run(model.d_train_op, feed_dict={model.x: bx, model.z: bz})
            
            # train generator
            bz = sample_Z(args.batch_size, args.z_dim, args.sampler, args.num_classes, args.n_cat)
            sess.run(model.g_train_op, feed_dict={model.x: bx, model.z: bz})
            
            # for tensorboard
            summary, global_step = sess.run([model.summary_op, model.global_step],
                                    feed_dict={model.x: bx,
                                               model.z: bz})

            if step % 100 == 0:
                bx, _ = mnist.train.next_batch(args.batch_size)
                bx = np.reshape(bx, [-1, img['w'], img['h'], img['c']])               
                bz = sample_Z(args.batch_size, args.z_dim, args.sampler, args.num_classes, args.n_cat)
      
                d_loss = sess.run(
                    model.d_loss, feed_dict={model.x: bx, model.z: bz}
                )
                g_loss = sess.run(
                    model.g_loss, feed_dict={model.z: bz}
                )
                
                print('Epoch[{}/{}] Step[{}/{}] g_loss:{:.4f}, d_loss:{:.4f}'.format(epoch, args.epoch, step,
                                                                                     steps_per_epoch, g_loss,
                                                                                     d_loss))
            summary_writer.add_summary(summary, global_step)
            epoch_loss_d.append(d_loss)
            epoch_loss_g.append(g_loss)

        mean_loss_d = sum(epoch_loss_d)/len(epoch_loss_d)
        mean_loss_g = sum(epoch_loss_g)/len(epoch_loss_g)

        print('Epoch:', '%04d' % epoch,
            'G loss: {:.4}'.format(mean_loss_g),
            'D loss: {:.4}'.format(mean_loss_d))

        with open(model.log_dir + "/training_loss.txt", "a+") as file:
                file.write("Epoch: %d\t LossD: %f\t LossG: %f\n" % (epoch, mean_loss_d, mean_loss_g))

        if(epoch % args.saving_cycle == 0):
            model.save(sess, args.model_dir, global_step)
        
        

def inference(args, model, sess):

        if args.model_dir is None:
            raise ValueError('Need to provide model directory')

        # load model
        model.load(sess, args.model_dir)
        
        """ give real imgs and check how they get clustered(cluster accuracy) """
        mnist = input_data.read_data_sets("MNIST_data/", one_hot=True)
        bx, bl = mnist.test.next_batch(args.batch_size)
        bx = np.reshape(bx, [-1, 28, 28, 1])

        zhats_gen, zhats_label = sess.run([model.z_infer_gen, model.z_infer_label], feed_dict={model.x : bx})

        mode2label = [6,5,0,3,2,9,7,4,8,1]

        acc = 0
        for i in range(args.batch_size):
            if(np.argmax(bl[i]) == mode2label[np.argmax(zhats_label[i])]):
                acc += 1
        acc = acc / args.batch_size * 100
        print(acc)
