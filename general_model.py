# -*- coding: utf-8 -*-
"""
Created on Thu May 16 09:39:21 2019

@author: Sooram Kang

"""
import tensorflow as tf 
import os
import sys

class Model(object):

    def __init__(self, num_to_keep):
        self.saver = tf.train.Saver(max_to_keep=num_to_keep)

    def session(self, sess, gpu_num):
        if sess is not None:
            self.sess = sess
        else:
            config_proto = tf.ConfigProto()
            config_proto.gpu_options.visible_device_list = gpu_num
            self.sess = tf.Session(config=config_proto)

    def initialize(self):
        self.sess.run(tf.global_variables_initializer())

    def save(self, sess, logdir, step):        
        model_name = 'model.ckpt'
        checkpoint_path = os.path.join(logdir, model_name)
        print('Storing checkpoint to {} ...'.format(logdir), end="")
        sys.stdout.flush()

        if not os.path.exists(logdir):
            os.makedirs(logdir)

        self.saver.save(sess, checkpoint_path, global_step=step)
        print(' Done.')

    def load(self, sess, logdir):
        print("Trying to restore saved checkpoints from {} ...".format(logdir),
              end="")

        ckpt = tf.train.get_checkpoint_state(logdir)
        if ckpt:
            print("  Checkpoint found: {}".format(ckpt.model_checkpoint_path))
            global_step = int(ckpt.model_checkpoint_path
                              .split('/')[-1]
                              .split('-')[-1])
            print("  Global step was: {}".format(global_step))
            print("  Restoring...", end="")
            self.saver.restore(sess, ckpt.model_checkpoint_path)
            print(" Done.")
            return global_step, sess
        else:
            print(" No checkpoint found.")
            return None, sess

    def close(self):
        self.sess.close()