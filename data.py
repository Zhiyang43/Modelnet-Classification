#!/usr/bin/env python
# -*- coding: utf-8 -*-
# 2021/12/01~
# Zhiyang Wang, zhiyangw@seas.upenn.edu

"""
@Author: Yue Wang
@Contact: yuewangx@mit.edu
@File: data.py
@Time: 2018/10/13 6:21 PM
"""


import os
import glob
import h5py
import numpy as np
from torch.utils.data import Dataset

zeroTolerance = 1e-5




def load_data(partition, num_points, perturb, relative):
    BASE_DIR = os.path.dirname(os.path.abspath(__file__))
    DATA_DIR = os.path.join(BASE_DIR, 'datasets')
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR,'modelnet10_hdf5_2048', '%s*.h5'%partition), recursive=True):
        f = h5py.File(h5_name)
        data = f['data'][:].astype('float32')
        # Samples are randomly drawn for training dataset so as to generate different realizations
        # if partition == 'train':
        #     samples = np.random.randint(data.shape[1], size = num_points)
        # else: # For testing, the stability is verified based on the same test dataset.
        samples = np.arange(num_points)

        data = data[:, samples, :]
        if perturb != 0 and relative == 0:
            data = jitter_pointcloud(data, sigma=perturb, clip=perturb *2) # Perturb the test dataset

        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_Lap_mat = computeLaplacian(computeAdj(all_data), True) # Calculate the corresponding Laplacian matrix
    if perturb != 0 and relative == 1:
        all_Lap_mat = rela_pointcloud(all_Lap_mat, sigma=perturb)

    all_label = np.concatenate(all_label, axis=0)


    all_label = np.where(all_label == 1, 0, all_label) # Convert the objects with label 2 as 1 and the rest as 0.
    all_label = np.where(all_label == 2, 1, all_label)
    all_label = np.where(all_label != 1, 0, all_label)


    return all_data, all_Lap_mat, all_label


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])
       
    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud

def computeAdj(x):
        # x: (batch_size, num_points, num_features)
        x_transpose = np.transpose(x, axes = [0,2,1])
        x_inner = np.matmul(x, x_transpose)
        x_inner = -2 * x_inner
        x_square = np.sum(np.square(x), axis = -1, keepdims = True)
        x_square_transpose = np.transpose(x_square, axes = [0,2,1])
        adj_mat = x_square + x_inner + x_square_transpose
        adj_mat = np.exp(-adj_mat)

        return adj_mat

def computeLaplacian(adj_mat, normalize):
        if normalize:
            D = np.sum(adj_mat, axis = 1)  # (batch_size,num_points)
            eye = np.ones_like(D[0,:])

            eye = np.diag(eye)
            D = 1 / np.sqrt(D)

            D_diag = np.diag(D[0,:])
            D_diag = np.expand_dims(D_diag, axis = 0)

            for i in range(1, D.shape[0], 1):
                Dtemp = np.diag(D[i,:])
                Dtemp = np.expand_dims(Dtemp, axis = 0)
                D_diag = np.concatenate((D_diag, Dtemp), axis=0) 
            
            L = eye - np.matmul(np.matmul(D_diag, adj_mat), D_diag)
            L = np.where(abs(L) < zeroTolerance, 0, L)

        else:
            D = np.sum(adj_mat, axis=1)  # (batch_size,num_points)
           
            D_diag = np.diag(D[0,:])
            D_diag = np.expand_dims(D_diag, axis = 0)

            for i in range(1, D.shape[0], 1):
                Dtemp = np.diag(D[i,:])
                Dtemp = np.expand_dims(Dtemp, axis = 0)
                D_diag = np.concatenate((D_diag, Dtemp), axis=0) 
      
            L = D_diag - adj_mat
            L = np.where(abs(L) < zeroTolerance, 0, L)
            
        return L


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    B, N, C = pointcloud.shape
    # Add a Gaussian random variable to the original coordinates
    pointcloud += np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    return pointcloud

def rela_pointcloud(Lap_mat, sigma=0.01):
    B, N, _ = Lap_mat.shape
    for i in range(B):

        E = np.random.uniform(-sigma, sigma, N)
        E = np.diag(E)
    # Add a Gaussian random variable to the original coordinates
        Lap_mat[i, :, :] += E.dot(Lap_mat[i,:, :])
    return Lap_mat


class ModelNet40(Dataset):
    def __init__(self, num_points, partition='train', perturb =0, relative = 0):
        self.data, self.lap_mat,  self.label = load_data(partition, num_points, perturb, relative)
        self.num_points = num_points
        self.partition = partition        

    def __getitem__(self, item):
        pointcloud = self.data[item, :, :]
        lap_matrix = self.lap_mat[item, :, :]
        label = self.label[item]

        return pointcloud, lap_matrix, label

    def __len__(self):
        return self.data.shape[0]


# if __name__ == '__main__':
#     train = ModelNet40(20)
# #     # test = ModelNet40(2048, 'test')
# #     # i = 0
# #     # for data, label in train:
# #     #     i = i + 1
# #     #     print(i)
# #     #     print(data.shape)
# #     #     print(label)
#     x, y, z = train.__getitem__([1,2])
#     print(len(z))
    # adj = computeAdj(x)
    # lap = computeLaplacian(adj, True)
    # print( lap )


