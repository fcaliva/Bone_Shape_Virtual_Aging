import numpy as np
import sys
import os
from os.path import join
import h5py
import pickle
import random
from matplotlib import pyplot as plt
import time
import pandas as pd
from scipy.io import loadmat
import pdb


def noisy(image, var= 0.1, noise_typ="gauss"):
    if noise_typ == "gauss":
      row,col= image.shape
      mean = 0
      var = var
      sigma = var**0.5
      gauss = np.random.normal(mean,sigma,(row,col))
      gauss = gauss.reshape(row,col)
      noisy = image + gauss
      return noisy
    elif noise_typ == "s&p":
      row,col = image.shape
      s_vs_p = 0.5
      amount = 0.004
      out = np.copy(image)
      # Salt mode
      num_salt = np.ceil(amount * image.size * s_vs_p)
      coords = [np.random.randint(0, i - 1, int(num_salt)) for i in image.shape]
      out[coords] = 1
      # Pepper mode
      num_pepper = np.ceil(amount* image.size * (1. - s_vs_p))
      coords = [np.random.randint(0, i - 1, int(num_pepper)) for i in image.shape]
      out[coords] = 0
      return out
    elif noise_typ == "poisson":
      vals = len(np.unique(image))
      vals = 2 ** np.ceil(np.log2(vals))
      noisy = np.random.poisson(image * vals) / float(vals)
      return noisy
    elif noise_typ =="speckle":
      row,col = image.shape
      gauss = np.random.randn(row,col)
      gauss = gauss.reshape(row,col)        
      noisy = image + image * gauss
      return noisy

def load_sample(path, predict_remodeling=False, add_noise=False):
    # import tar this is 3 channeled
    tar = loadmat(path[-1])['bone_sphere']
    if predict_remodeling == True:
        # t1, t2, t3, t4
        for_remodeling = loadmat(path[-2])['bone_sphere'][..., 0]  
        for_remodeling/=50
 
    # initialize img as seg's size 3channeled
    img = np.zeros(tar.shape)

    # since bone_sphere has replicated channels keep only the first
    tar = tar[..., 0]
    # normalize 50mm
    tar /= 50.0
    # prepare the input:
    for xx in range(3):
        # since bone_sphere has replicated channels keep only the first
        inp = loadmat(path[xx])['bone_sphere'][..., 0]               
        inp /= 50.0
        if add_noise == True:            
            noisy_inp = noisy(inp, var=0.03)
            # for var in [x / 10000.0 for x in range(10, 100, 10)]: noisy_inp = noisy(inp, var=var); plt.subplot(1,2,1);plt.imshow(inp); plt.colorbar(); plt.subplot(1,2,2); plt.imshow(noisy_inp);plt.colorbar();plt.show(); plt.savefig(f'/data/knee_mri8/Francesco/deep_bone_aging/code/python/DL_code/images_checks/noise{str(var).replace(".","")}.png',dpi=400); plt.close()            
            img[..., xx] = noisy_inp
        else:    
            img[..., xx] = inp
    if predict_remodeling == True and add_noise == False:
        target = tar - for_remodeling
        target *= 100.0
        mask = (tar != 0) * (img[...,-1]!=0)
        target *= mask
    if predict_remodeling == True and add_noise == True:
        target = tar - for_remodeling
        target *= 100.0
        mask = (tar != 0) * (img[..., -1] != 0)
        target *= mask
        target = np.stack((target, tar), axis=2)
    else:
        target = tar
    return img, target

show_example = False
if show_example:
    path_example = ['all_mat_save_spherical_3class/9271853_LEFT_0_KL2_z0z0z0.mat', 
                    'all_mat_save_spherical_3class/9271853_LEFT_1_KL2_z0z0z0.mat',
                    'all_mat_save_spherical_3class/9271853_LEFT_3_KL2_z0z0z0.mat', 
                    'all_mat_save_spherical_3class/9271853_LEFT_8_KL2_z0z0z0.mat']
    img, tar = load_sample(path_example)
    plt.figure(figsize=(10,10))
    plt.subplot(2,2,1)
    plt.imshow(img[..., 0]), plt.title('TP 0')
    plt.show()
    plt.subplot(2,2,2)
    plt.imshow(img[..., 1]), plt.title('TP 1')
    plt.show()
    plt.subplot(2,2,3)
    plt.imshow(img[...,2]),plt.title('TP 2')
    plt.show()
    plt.subplot(2,2,4)
    plt.imshow(tar), plt.title('Target')
    plt.show()    
    plt.savefig('images_checks/example/datasample.png')
 
def crop_volume(volume,crop):
    if crop[1] == 0:
        volume = volume[crop[0]:,...]
    else:
        volume = volume[crop[0]:-crop[1],...]

    if crop[3] == 0:
        volume = volume[:,crop[2]:,...]
    else:
        volume = volume[:,crop[2]:-crop[3],...]
    if len(crop)>4:
        if crop[5] == 0:
            volume = volume[:,:,crop[4]:,...]
        else:
            volume = volume[:,:,crop[4]:-crop[5],...]
    return volume

def get_normalization_values(normalization_file):
    list_files = pd.read_csv(normalization_file,header=None).values.tolist()
    minmax_vals = dict()
    for file in list_files:
        id_pat = str(file[0])
        minmax_vals[id_pat] = dict()
        minmax_vals[id_pat]['min']=file[1]
        minmax_vals[id_pat]['max']=file[2]
    return minmax_vals

class data_loader:
    def __init__(self, data_root, batch_size, im_dims, crop, num_classes, idx_classes, num_channels, normalization_file=' ', evaluate_mode=False, input_type='float32', target_type='float32', predict_remodeling = False, add_noise = False):
        # modify these lines to increase data loader flexibility
        if '.pkl' in data_root or '.pickle' in data_root:
            with open( data_root, "rb" ) as lf:
                list_files = pickle.load( lf )
        elif '.csv' in data_root:
            list_files = pd.read_csv(data_root,header=None).values.tolist()
        elif '.lsx' in data_root or '.xls' in data_root:
            list_files = pd.read_excel(data_root).values.tolist()
        else:
            print('Our dataLoader is not prepared to accept such input data, see its __init__')
            exit()
            return
        self.all_files       = list_files
        self.evaluate_mode  = evaluate_mode
        if self.evaluate_mode:
            self.order_idx       = list(range(len(self.all_files)))
        else:
            self.order_idx       = np.random.permutation(len(self.all_files))
        self.data_size       = len(self.order_idx)
        self.batch_size      = batch_size
        self.batch_cnt       = 0
        self.batch_max       = np.ceil(self.data_size/self.batch_size)
        self.im_dims         = im_dims
        self.im_batch = np.zeros(
            [self.batch_size]+[x for x in self.im_dims]+[num_channels], dtype=input_type)
        self.seg_batch = np.zeros(
            [self.batch_size]+[x for x in self.im_dims]+[num_classes], dtype=input_type)
        self.crop            = crop
        self.idx_classes     = idx_classes
        self.name_batch      = []
        if ' ' not in normalization_file:
            self.norm_values = get_normalization_values(normalization_file)
            self.normalize_input = True
        else:
            self.norm_values = normalization_file
            self.normalize_input = False
        self.predict_remodeling = predict_remodeling
        self.add_noise = add_noise

    def __len__( self ):
        return len(self.order_idx)

    def __shuffle__( self ):
        random.shuffle(self.order_idx)
        return self

    def __getitem__( self, key ):
        idx = self.order_idx[key]
        if len(self.all_files[0]) == 4:
            fname_img = self.all_files[idx][0]
            fname_seg= self.all_files[idx][-1]
        else:
            print('Our dataLoader is not prepared to accept such input data, see its __getitem__')
            exit()
            return
        img, seg = load_sample(
            path=self.all_files[idx], predict_remodeling=self.predict_remodeling, add_noise=self.add_noise)

        try:
            if self.normalize_input == True:
                idpat = fname_img.split('/')[-2];
                img = img- self.norm_values[idpat]['min']
                img = img/ (self.norm_values[idpat]['max']-self.norm_values[idpat]['min'])
        except:
            print(
                f'Error in the normalization file. File {idpat} is missing its normalization value')
            pdb.set_trace()

        img = crop_volume(img, self.crop)

        if seg.ndim == len(self.im_dims):
            seg = seg[...,np.newaxis]

        seg = seg[...,self.idx_classes]

        seg = crop_volume(seg, self.crop)

        return img, seg, fname_seg

    def fetch_batch(self):
        self.name_batch = []
        for i in range(self.batch_size):
            if (i + self.batch_size*self.batch_cnt) < self.data_size:
                idx = self.order_idx[i + self.batch_size*self.batch_cnt]
            else:
                idx = self.order_idx[random.randint(0, self.data_size)]

            img, seg, name = self.__getitem__( idx )
            self.name_batch.append(name)
            if img.ndim == len(self.im_dims):
                self.im_batch[i,...,0] = img
            else:
                self.im_batch[i:] = img
            self.seg_batch[i:] = seg

        self.batch_cnt += 1

        if self.batch_cnt >= self.batch_max:
            self.batch_cnt = 0
            if self.evaluate_mode == False:
                self.__shuffle__()

        return self.im_batch, self.seg_batch, self.name_batch
