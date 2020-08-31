import numpy as np
import os
import config
import torch
# from spline_Utils_jiahui import *
import h5py
from spline_Utils import plot_Spline_Img, plot_Spline_Curve_Control_Point


class H5Reader:
    def __init__(self, file_name, variable=False, multiple=False):
        print('==> Reading from ' + file_name)
        self.f = h5py.File(file_name, 'r')
        self.imgs = self.f['imgs']
        self.labels = self.f['labels']
        self.variable = variable
        self.multiple = multiple
        if variable:
            self.n_points = self.f['num_points']
        if multiple:
            self.n_lines = self.f['num_lines']
            self.n_points_list = self.f['num_points_list']

    def read_One(self, idx_patch, idx):
        img = self.imgs[str(idx_patch)][idx]
        label = self.labels[str(idx_patch)][idx]
        assert img.shape == (128, 128)

        if self.variable:
            assert label.shape == (config.max_point, 2)
            n_point = self.n_points[str(idx_patch)][idx]
            return img, label, n_point
        if self.multiple:
            # print(label.shape)
            assert label.shape == (
                config.max_line, config.max_point, config.n_dim)
            n_line = self.n_lines[str(idx_patch)][idx]
            n_point_matrix = self.n_points_list[str(idx_patch)][idx]
            # print(n_point_matrix)
            # print(n_point_matrix.shape)
            assert n_point_matrix.shape == (config.max_line,)
            return img, label, n_line, n_point_matrix
        return img, label


class Multi_Vary_Spline_Dataset:

    def __init__(self, path, n_all, verbose=0, train=True, train_part=0.7, transform=None):
        self.transform = transform
        self.train = train
        self.train_part = train_part
        self.path = path
        self.n_all = n_all
        self.n_train = int(self.n_all*self.train_part)
        self.n_test = self.n_all - self.n_train
        self.reader = H5Reader(os.path.join(
            self.path, 'dataset_cc_multiple_variable_spline_500000'+'.hdf5'), multiple=True)
        np.random.seed(10)
        self.img_list = np.random.permutation(self.n_all)
        print(self.img_list[:20])
        print('==> Please Check Whethere train and test are the same')

    def __getitem__(self, idx):
        if self.train:
            idx = self.img_list[idx]
        else:
            idx = self.img_list[idx+self.n_train]

        i_patch = int(idx/config.n_per_patch)
        idx_in_patch = idx % config.n_per_patch

        img, label, n_line, n_point = self.reader.read_One(
            i_patch, idx_in_patch)
        if config.reader_verbose:
            print(label)
            print(f"N_line = {n_line}")
            print(f"n_point = {n_point}")
            plot_Spline_Img(img)
            point_list = []
            for i_line in range(n_line):
                point_list.append(
                    label[i_line, :int(n_point[i_line]), :].squeeze())
            plot_Spline_Curve_Control_Point(point_list)
        img = img.reshape(128, 128, 1).astype(np.float)
        img = img*255
        if self.transform != None:
            img = self.transform(img)
        img = img.expand(3, 128, 128)
        label = torch.from_numpy(label).view(
            config.max_line, config.max_point, config.n_dim)
        label = label / 128
        # get line mask array
        mask_line_array = torch.zeros(config.max_line)
        line_clf_array = torch.zeros(config.max_line)
        line_clf_array[n_line - 1] = 1
        for idx in range(n_line):
            mask_line_array[idx] = 1
        mask_point_array = torch.zeros(config.max_line, config.max_point)
        point_clf_array = torch.zeros(config.max_line, config.max_point)
        for idx in range(n_line):
            n_point_line = int(n_point[idx])
            point_clf_array[idx, n_point_line - 1] = 1
            for i in range(n_point_line):
                mask_point_array[idx, i] = 1
        return img, label, mask_line_array, line_clf_array, mask_point_array, point_clf_array

    def __len__(self):
        if self.train:
            return self.n_train
        else:
            return self.n_test


if __name__ == '__main__':
    data = Multi_Vary_Spline_Dataset(
        config.dataset_path, config.n_data, verbose=1)
    for idx in range(10):
        # data[idx]
        img, target, mask, clf, mask_point, clf_point = data[idx]
        print(target, mask, clf, mask_point, clf_point)
