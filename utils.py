# -*- coding: utf-8 -*-
"""
Created on Wed Aug 21 16:28:34 2019

@author: Sooram Kang
"""
import random
import numpy as np

def get_batch(data, config, num_batch):
    """
    Randomly get 'num_batch' number of cubes
    :param data: a three dimensional numpy array
    :param num_batch: batch size(number of cubes for each batch)
    :return: cubes(num_batch, n_set, x_size, y_size, z_size) (batch_size,4,3,3,11)
    """

    batch_cubes = []
    for _ in range(num_batch):
        # starting indices
        x = random.randint(0, config['x']-config['batchX'])
        y = random.randint(0, config['y']-config['batchY'])
        z = random.randint(0, config['z']-config['batchZ'])

        curr_data = []
        for i in range(config['n_set']):
            x_idx = x + config['x']*i
            curr_data.append(data[x_idx:x_idx+config['batchX'],y:y+config['batchY'],z:z+config['batchZ']])

        batch_cubes.append(curr_data)

    return np.array(batch_cubes)

def sample_Z(batch, z_dim , sampler = 'one_hot', num_class = 10, n_cat = 1, label_index = None):
    if sampler == 'mul_cat':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        return np.hstack((0.10 * np.random.randn(batch, z_dim-num_class*n_cat),
                          np.tile(np.eye(num_class)[label_index], (1, n_cat))))
    elif sampler == 'one_hot':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        return np.hstack((0.10 * np.random.randn(batch, z_dim-num_class), np.eye(num_class)[label_index]))
    elif sampler == 'uniform':
        return np.random.uniform(-1., 1., size=[batch, z_dim])
    elif sampler == 'normal':
        return 0.15*np.random.randn(batch, z_dim)
    elif sampler == 'mix_gauss':
        if label_index is None:
            label_index = np.random.randint(low = 0 , high = num_class, size = batch)
        return (0.1 * np.random.randn(batch, z_dim) + np.eye(num_class, z_dim)[label_index])