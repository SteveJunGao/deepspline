import os
import ctypes as c
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import numpy as np
import config
import random
from cubic_spline.OldCubicSpline import CubicSpline
from matplotlib.backends.backend_agg import FigureCanvasAgg as FigureCanvas
from matplotlib.figure import Figure

def sample_Point_Cloud(img):
	''' Given an image, sample the point cloud of the image.
	The number of points is defined in the config file.
	Image should be binary images.
	Input: img: [x,y,1] np.array
	Return: np.array [config.N, 2]
	'''
	assert img.ndim == 3
	assert img.shape[2] == 1
	pos = img.nonzero()
	point_cloud_pos = []
	for i in range(config.N):
		idx = random.randint(0, len(pos[0]) - 1)
		point_cloud_pos.append([random.random() + pos[1][idx], random.random() + pos[0][idx]])
	point_cloud_pos = np.asarray(point_cloud_pos)
	return point_cloud_pos


def get_Img_Matrix_Antialiasing(point_pos, w, x_size=128, y_size=128):
	point_pos = np.dot(w, point_pos)
	mydpi = 96
	fig = Figure(figsize=(x_size / mydpi, y_size / mydpi), dpi=mydpi)
	fig.subplots_adjust(left=0., bottom=0., right=1., top=1., wspace=0, hspace=0)
	fig.set_facecolor('black')
	a = fig.add_subplot(111, axisbg='r')
	canvas = FigureCanvas(fig)
	# ax = fig.gca()

	a.axis([0, x_size, y_size, 0])
	a.axis('off')
	a.plot(point_pos[:, 0], point_pos[:, 1], c='w')
	# fig.tight_layout()
	# plt.tight_layout()
	canvas.draw()  # draw the canvas, cache the renderer
	width, height = fig.get_size_inches() * fig.get_dpi()
	# print(width, height)
	image = np.fromstring(canvas.tostring_rgb(), dtype='uint8').reshape(int(height), int(width), 3)[:, :, 0]
	return image
def get_Spline_Matrix_Control_Point(point_pos, w, x_size=128, y_size=128):
	'''Input: control point position, spline weights, image_size
		Output: binary image matrix (x_size, y_size)
	'''
	line_pos = np.dot(w, point_pos)
	return get_Spline_Matrix_Point_Cloud(line_pos)


def get_Spline_Matrix_Point_Cloud(point_cloud, x_size=128, y_size=128, up_int=False, down_int=True):
	''' Input: the Point Cloud of a image,
		Output: binary image, matrix (x_size, y_size)
	'''
	spline_pic = np.zeros((x_size, y_size), dtype=np.bool_)

	if down_int:
		floor = np.floor(point_cloud)
		floor = np.int32(floor)
		floor[floor >= x_size] = x_size - 1
		spline_pic[floor[:, 1], floor[:, 0]] = True
	if up_int:
		ceil = np.ceil(point_cloud)
		ceil = np.int32(ceil)
		ceil[ceil >= x_size] = x_size - 1
		spline_pic[ceil[:, 1], ceil[:, 0]] = True
	return spline_pic

def get_img_mask(point_cloud, x_size=128, y_size=128, up_int=False, down_int=True):
	img = np.zeros((x_size, y_size), dtype = np.bool)
	target = np.zeros((config.n_discret_bins+1, x_size, y_size), dtype = np.bool)
	if down_int:
		floor = np.floor(point_cloud)
		point_cloud = np.int32(floor)
		point_cloud[point_cloud >= x_size] = x_size - 1
	if up_int:
		ceil = np.ceil(point_cloud)
		point_cloud = np.int32(ceil)
		point_cloud[point_cloud >= x_size] = x_size - 1
	img[point_cloud[:, 1], point_cloud[:, 0]] = True
	for idx in range(config.n_discret_bins):
		tmp_pos = np.concatenate((point_cloud[idx*config.point_per_bins: (idx+1)*config.point_per_bins],
								  point_cloud[-((idx+1)*config.point_per_bins+1) : -(idx*config.point_per_bins+1)]))
		target[idx, tmp_pos[:,1], tmp_pos[:, 0]] = True
	target[config.n_discret_bins] = np.bitwise_not(img)
	return img, target





def get_Spline_Weight_Matrix(n_point, N):
	# s = Spline(n_point, n_degree)
	# m = s.get_Nodes_Eval_Matrix(N)
	# print(n_point)
	s = CubicSpline(np.ndarray((n_point, 2)))
	m = s.get_bfunc_matrix(np.linspace(0, 1, N))
	return m


def plot_Spline_Img(img, x_size=128, y_size=128, save=False, path=None):
	# img = img[::-1, :]
	plt.imshow(img, origin='lower')
	if save:
		assert not path is None
		plt.axis([0, x_size, 0, y_size])
		plt.savefig(path)
		plt.clf()
		plt.close()
		return
	plt.show()
	plt.clf()
	plt.close()


def plot_Spline_Curve_Control_Point(control_point_list, give_w = None, label_list=None, x_size=128, y_size=128,
									save=False, path=None, caption=None):
	plt.axis([0, x_size, 0, y_size])
	line_list = []
	for control_points in control_point_list:
		if give_w is None:
			w = get_Spline_Weight_Matrix(len(control_points), config.N)
		else: w = give_w
		line_point = np.dot(w, control_points)
		l, = plt.plot(line_point[:, 0], line_point[:, 1])
		plt.plot(control_points[:, 0], control_points[:, 1], 'o')
		for idx, xy in enumerate(zip(control_points[:, 0], control_points[:, 1])):
			plt.annotate(str(idx), xy)
		line_list.append(l)
	if not label_list is None:
		assert len(label_list) == len(line_list)
		plt.legend(line_list, label_list)
	if caption:
		plt.title(caption)
	if save:
		assert not path is None
		plt.savefig(path)
		plt.clf()
		plt.close()
		return
	plt.show()
	plt.clf()
	plt.close()


def plot_Spline_Curve_Point_Cloud(point_cloud_list, label_list=None, x_size=128, y_size=128,
								  save=False, path=None):
	plt.axis([0, x_size, 0, y_size])
	line_list = []
	for idx, points in enumerate(point_cloud_list):
		l, = plt.plot(points[:, 0], points[:, 1], 'o', label='Line_' + str(idx))
		line_list.append(l)
	if not label_list is None:
		assert len(line_list) == len(label_list)
		plt.legend(line_list, label_list)
	if save:
		assert not path is None
		plt.savefig(path)
	plt.show()
	plt.clf()
	plt.close()


if __name__ == '__main__':
	n_point = 5
	n_degree = 3
	N = 1024
	control_points = [[random.random(), random.random()] for i in range(n_point)]
	control_points = np.asarray(control_points)
	control_points = control_points * 128
	s = CubicSpline(np.ndarray((n_point, 2)))
	w = s.get_bfunc_matrix(np.linspace(0, 1, N))
	plot_Spline_Curve_Control_Point([control_points], w)
	plot_Spline_Curve_Point_Cloud([np.dot(w, control_points)])
	img = get_Spline_Matrix_Control_Point(control_points, w)
	plot_Spline_Img(img)
	img = get_Img_Matrix_Antialiasing(control_points, w)
	plot_Spline_Img(img)
	point_cloud = sample_Point_Cloud(img.reshape(128, 128, 1))
	plot_Spline_Curve_Point_Cloud([point_cloud])
