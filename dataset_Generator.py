import numpy as np
import os
import random
from utils import progress_bar
from spline_Utils import get_Spline_Matrix_Control_Point, get_Spline_Matrix_Point_Cloud, get_Spline_Matrix_Point_Cloud
from spline_Utils import  plot_Spline_Curve_Control_Point, plot_Spline_Curve_Point_Cloud, plot_Spline_Img, get_Spline_Weight_Matrix
import torch
import config
import h5py

random.seed(config.manual_seed)

class Spline_Generator():
	'''
	In this generator, I will generate only one line with variable number of control point
	'''

	def __init__(self, x_size=128, y_size=128):
		self.x_size = x_size
		self.y_size = y_size
		self.w_list = []
		for i in range(config.max_point - config.min_point + 1):
			self.w_list.append(get_Spline_Weight_Matrix(i + config.min_point, config.n_degree, config.N))

	def gen_one_img(self, n_point, point_pos=None):
		# img_matrix = np.zeros((self.x_size, self.y_size), dtype = np.bool)
		# target_matrix = np.zeros((config.n_discret_bins + 1,self.x_size, self.y_size), dtype = np.bool)
		w = self.w_list[n_point - config.min_point]

		if point_pos is None:

			point_pos = [[random.random() * self.x_size, random.random() * self.y_size]
				             for i in range(n_point)]
			point_pos = np.asarray(point_pos)
			if point_pos[0][0] > point_pos[-1,0]:
				point_pos = point_pos[::-1, :]
				point_pos = np.ascontiguousarray(point_pos)

		assert point_pos[0, 0] <= point_pos[-1, 0]
		assert point_pos.shape == (n_point, 2)
		point_cloud = np.dot(w, point_pos)
		img = get_Spline_Matrix_Point_Cloud(point_cloud)
		assert img.shape == (self.x_size, self.y_size)
		return img, point_pos

class Multi_Spline_Generator():
	'''
	In this generator, I will generate multiple spline curves with variable number of control points
	'''

	def __init__(self, x_size=128, y_size=128):
		self.x_size = x_size
		self.y_size = y_size
		self.w_list = []
		for i in range(config.max_point - config.min_point + 1):
			self.w_list.append(get_Spline_Weight_Matrix(i + config.min_point, config.n_degree, config.N))

	def gen_one_img(self, n_line):
		img_matrix = np.zeros((self.x_size, self.y_size), dtype = np.bool)
		# target_matrix = np.zeros((config.n_discret_bins + 1,self.x_size, self.y_size), dtype = np.bool)
		point_pos_matrix = np.zeros((n_line, config.max_point, config.n_dim))
		n_point_list = np.zeros((config.max_line))
		for i in range(n_line):
			n_point = random.randint(config.min_point, config.max_point)
			point_pos = [[random.random()*self.x_size, random.random()*self.y_size]
			             for j in range(n_point)]
			point_pos = np.asarray(point_pos)
			if point_pos[0][0] > point_pos[-1,0]:
				point_pos = point_pos[::-1, :]
				point_pos = np.ascontiguousarray(point_pos)
			assert point_pos[0,0] <= point_pos[-1, 0]
			assert point_pos.shape == (n_point, 2)
			point_pos_matrix[i, :n_point, :] = point_pos
			n_point_list[i] = n_point
			w = self.w_list[n_point - config.min_point]
			point_cloud = np.dot(w, point_pos)
			img = get_Spline_Matrix_Point_Cloud(point_cloud)
			img_matrix = np.bitwise_or(img_matrix, img)
		return img_matrix, point_pos_matrix, n_point_list

def generate_variable_spline(verbose = 0):
	random.seed(config.manual_seed+1)
	gen = Spline_Generator()
	img_list = []
	label_list = []
	n_point_list = []

	n_patch = 0
	if not os.path.exists(config.dataset_path):
		os.makedirs(config.dataset_path)
	f = h5py.File(config.dataset_path + '/dataset_cc_variable_number_tmp' + '.hdf5', 'w')
	img_grp = f.create_group('imgs')
	label_grp = f.create_group('labels')
	n_point_grp = f.create_group('num_points')

	for idx in range(1, config.n_data + 1):
		n_point = random.randint(config.min_point, config.max_point)
		img, point_pos = gen.gen_one_img(n_point)
		if config.gen_verbose:
			plot_Spline_Img(img)
			plot_Spline_Curve_Control_Point([point_pos], get_Spline_Weight_Matrix(n_point, config.n_degree, config.N))
		n_point_list.append(n_point)
		img_list.append(img)
		tmp_point_pos = np.zeros((config.max_point, config.n_dim))
		tmp_point_pos[:n_point] = point_pos
		label_list.append(tmp_point_pos)
		if idx % config.n_per_patch == 0:
			imgs = np.asarray(img_list)
			labels = np.asarray(label_list)
			n_points = np.asarray(n_point_list)
			assert np.sum(imgs > 1) == 0
			assert np.sum(labels > 128) == 0
			assert np.sum(imgs < 0) == 0
			assert np.sum(labels < 0) == 0
			assert imgs.shape == (config.n_per_patch, 128, 128)
			assert labels.shape == (config.n_per_patch, config.max_point, config.n_dim)
			# print(n_points.shape)
			# print(labels.shape)
			assert n_points.shape == (config.n_per_patch,)
			n_point_grp.create_dataset(str(n_patch), data = n_points)
			img_grp.create_dataset(str(n_patch), data = imgs)
			label_grp.create_dataset(str(n_patch), data = labels)
			img_list = []
			label_list = []
			n_point_list = []
			n_patch += 1
		progress_bar(idx, config.n_data)

def generate_multiple_variable_spline(verbose = 0):
	random.seed(config.manual_seed+1)
	gen = Multi_Spline_Generator()
	img_list = []
	label_list = []
	n_line_list = []
	n_point_list = []
	n_patch = 0
	if not os.path.exists(config.dataset_path):
		os.makedirs(config.dataset_path)
	f = h5py.File(config.dataset_path + '/dataset_cc_multiple_variable_spline_tmp' + '.hdf5', 'w')
	img_grp = f.create_group('imgs')
	label_grp = f.create_group('labels')
	n_point_grp = f.create_group('num_points_list')
	n_line_grp = f.create_group('num_lines')
	for idx in range(1, config.n_data + 1):
		n_line = random.randint(config.min_line, config.max_line)
		img, point_pos_matrix, n_point_matrix= gen.gen_one_img(n_line)
		if config.gen_verbose:
			plot_Spline_Img(img)
			plot_Spline_Curve_Control_Point([point_pos_matrix[i, :int(n_point_matrix[i])] for i in range(n_line)])

		n_line_list.append(n_line)
		n_point_list.append(n_point_matrix)
		img_list.append(img)
		tmp_pos = np.zeros((config.max_line, config.max_point, config.n_dim))
		tmp_pos[:n_line] = point_pos_matrix
		label_list.append(tmp_pos)
		if idx % config.n_per_patch == 0:
			imgs = np.asarray(img_list)
			labels = np.asarray(label_list)
			n_points = np.asarray(n_point_list)
			n_lines = np.asarray(n_line_list)
			assert np.sum(imgs > 1) == 0
			assert np.sum(labels > 128) == 0
			assert np.sum(imgs < 0) == 0
			assert np.sum(labels < 0) == 0
			assert imgs.shape == (config.n_per_patch, 128, 128)
			assert labels.shape == (config.n_per_patch, config.max_line, config.max_point, config.n_dim)
			# print(n_points.shape)
			# print(labels.shape)
			assert n_points.shape == (config.n_per_patch, config.max_line)
			assert n_lines.shape == (config.n_per_patch,)
			n_point_grp.create_dataset(str(n_patch), data = n_points)
			img_grp.create_dataset(str(n_patch), data = imgs)
			label_grp.create_dataset(str(n_patch), data = labels)
			n_line_grp.create_dataset(str(n_patch), data = n_lines)
			img_list = []
			label_list = []
			n_point_list = []
			n_line_list = []
			n_patch += 1
		progress_bar(idx, config.n_data)



if __name__ == '__main__':
	generate_multiple_variable_spline()

