import os
import ctypes as c
import matplotlib
# matplotlib.use('Agg')
import matplotlib.pyplot as plt
# import matplotlib.pyplot as plt
import numpy as np
import config
import random
from spline_interface import Spline

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
		idx = random.randint(0, len(pos[0])-1)
		point_cloud_pos.append([random.random()+pos[1][idx], random.random()+pos[0][idx]])
	point_cloud_pos = np.asarray(point_cloud_pos)
	return point_cloud_pos

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
		floor[floor>=x_size] = x_size - 1
		spline_pic[floor[:,1], floor[:,0]] = True
	if up_int:
		ceil = np.ceil(point_cloud)
		ceil = np.int32(ceil)
		ceil[ceil>=x_size] = x_size - 1
		spline_pic[ceil[:,1], ceil[:,0]] = True
	return spline_pic

def get_Spline_Weight_Matrix(n_point, n_degree, N):
	s = Spline(n_point, n_degree)
	m = s.get_Nodes_Eval_Matrix(N)
	return m

def get_Spline_Point_Cloud(point_pos, n_degree, N):
	s = Spline(point_pos.shape[0], n_degree)
	m = s.get_Nodes_Eval_Matrix(N)
	return np.dot(m, point_pos)

def subplot_spline_control_point(subplt, control_point_list, give_w = None, label_list=None, x_size=128, y_size=128):
	subplt.axis([0, x_size, 0, y_size])
	line_list = []
	for control_points in control_point_list:
		if give_w is None:
			w = get_Spline_Weight_Matrix(len(control_points), config.n_degree, config.N)
		else:
			w = give_w
		# print(w.shape)
		line_point = np.dot(w, control_points)
		l, = subplt.plot(line_point[:, 0], line_point[:, 1])
		subplt.plot(control_points[:, 0], control_points[:, 1], 'o')
		for idx, xy in enumerate(zip(control_points[:, 0], control_points[:, 1])):
			subplt.annotate(str(idx), xy)
		line_list.append(l)
	if not label_list is None:
		assert len(label_list) == len(line_list)
		subplt.legend(line_list, label_list)

def plot_attention(input, attention_map, pos_pred, pos_target, n_pred_line, save=False, path = None):
	'''
	Plot attention map here,
	[[Input, att_0, att_1, att_2],
	 [Output, crv_0, crv_1, crv_2]]

	:return:
	'''
	input = input.squeeze()
	# print(pos_pred)
	pos_pred = pos_pred * 128
	pos_target = pos_target * 128
	plt.figure(figsize = (6 * 4, 6*2))
	# print(n_pred_line)
	input_img = plt.subplot(241)
	input_img.imshow(input, origin='lower')
	input_img.set_title('Input Image')
	subplt_t = plt.subplot(245)
	subplot_spline_control_point(subplt_t, [pos_target[i].reshape(config.max_point, config.n_dim) for i in range(n_pred_line)])
	subplt_t.set_title('Target Image')
	# print(attention_map.shape)
	att_0 = attention_map[0]
	att_0_img = plt.subplot(242)
	att_0_img.imshow(att_0, origin='lower')

	subplt_0 = plt.subplot(246)
	subplot_spline_control_point(subplt_0, [pos_pred[0].reshape(config.max_point, config.n_dim)], label_list=['Predict'])
	att_0_img.set_title('Attention Map 1')
	subplt_0.set_title('Predict Spline 1')
	if n_pred_line > 1:
		att_1 = attention_map[1]
		att_1_img = plt.subplot(243)
		att_1_img.imshow(att_1, origin='lower')
		subplt_1 = plt.subplot(247)
		subplot_spline_control_point(subplt_1, [pos_pred[1].reshape(config.max_point, config.n_dim)], label_list=['Predict'])
		att_1_img.set_title('Attention Map 2')
		subplt_1.set_title('Predict Spline 2')
	if n_pred_line > 2:
		att_2 = attention_map[2]
		att_2_img = plt.subplot(244)
		att_2_img.imshow(att_2, origin='lower')
		subplt_2 = plt.subplot(248)
		subplot_spline_control_point(subplt_2, [pos_pred[2].reshape(config.max_point, config.n_dim)], label_list=['Predict'])
		att_2_img.set_title('Attention Map 3')
		subplt_2.set_title('Predict Spline 3')


	if save:
		assert not path is None
		plt.savefig(path)
		plt.clf()
		plt.close()
		return
	plt.show()
	plt.clf()
	plt.close()

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
									save=False, path=None):
	plt.axis([0, x_size, 0, y_size])
	line_list = []
	for control_points in control_point_list:
		if give_w is None:
			w = get_Spline_Weight_Matrix(len(control_points), config.n_degree, config.N)
		else: w = give_w
		# print(w.shape)
		line_point = np.dot(w, control_points)
		l, = plt.plot(line_point[:,0], line_point[:,1])
		plt.plot(control_points[:,0], control_points[:,1], 'o')
		for idx, xy in enumerate(zip(control_points[:,0], control_points[:,1])):
			plt.annotate(str(idx), xy)
		line_list.append(l)
	if not label_list is None:
		assert len(label_list) == len(line_list)
		plt.legend(line_list, label_list)
	if save:
		assert not path is None
		plt.savefig(path)
		plt.clf()
		plt.close()
		return
	plt.show()
	plt.clf()
	plt.close()

def plot_Spline_Curve_Control_Point_Multi_Set(control_points_target, control_points_pred, give_w = None,
										target_label_list=None, pred_label_list = None, x_size=128, y_size=128,
									save=False, path=None):
	plt.figure(figsize = (6 * 2, 6))
	plt.axis([0, x_size, 0, y_size])
	line_list = []
	fig_t = plt.subplot(121)
	fig_t.axis([0, x_size, 0, y_size])
	for control_points in control_points_target:

		if give_w is None:
			w = get_Spline_Weight_Matrix(len(control_points), config.n_degree, config.N)
		else: w = give_w
			# print(w.shape)
		line_point = np.dot(w, control_points)
		l, = fig_t.plot(line_point[:,0], line_point[:,1])
		fig_t.plot(control_points[:,0], control_points[:,1], 'o')
		for idx, xy in enumerate(zip(control_points[:,0], control_points[:,1])):
			fig_t.annotate(str(idx), xy)
		line_list.append(l)
	if not target_label_list is None:
		assert len(target_label_list) == len(line_list)
		fig_t.legend(line_list, target_label_list)

	line_list = []
	fig_p = plt.subplot(122)
	fig_p.axis([0, x_size, 0, y_size])
	for control_points in control_points_pred:

		if give_w is None:
			w = get_Spline_Weight_Matrix(len(control_points), config.n_degree, config.N)
		else:
			w = give_w
		# print(w.shape)
		line_point = np.dot(w, control_points)
		l, = fig_p.plot(line_point[:, 0], line_point[:, 1])
		fig_p.plot(control_points[:, 0], control_points[:, 1], 'o')
		for idx, xy in enumerate(zip(control_points[:, 0], control_points[:, 1])):
			fig_p.annotate(str(idx), xy)
		line_list.append(l)
	if not pred_label_list is None:
		assert len(pred_label_list) == len(line_list)
		fig_p.legend(line_list, pred_label_list)
	if save:
		assert not path is None
		plt.savefig(path)
		plt.clf()
		plt.close()
		return
	plt.show()
	plt.clf()
	plt.close()


# def plot_Spline_Curve_Control_Point(control_point_list, label_list=None, x_size=128, y_size=128,
# 									save=False, path=None):
# 	plt.axis([0, x_size, 0, y_size])
# 	line_list = []
# 	for control_points in control_point_list:
# 		line_point = np.dot(w, control_points)
# 		l, = plt.plot(line_point[:,0], line_point[:,1])
# 		plt.plot(control_points[:,0], control_points[:,1], 'o')
# 		for idx, xy in enumerate(zip(control_points[:,0], control_points[:,1])):
# 			plt.annotate(str(idx), xy)
# 		line_list.append(l)
# 	if not label_list is None:
# 		assert len(label_list) == len(line_list)
# 		plt.legend(line_list, label_list)
# 	if save:
# 		assert not path is None
# 		plt.savefig(path)
# 		plt.clf()
# 		plt.close()
# 		return
# 	plt.show()
# 	plt.clf()
# 	plt.close()

def plot_Spline_Curve_Point_Cloud(point_cloud_list, label_list=None, x_size=128, y_size=128,
									save=False, path=None):
	plt.axis([0, x_size, 0, y_size])
	line_list = []
	for idx, points in enumerate(point_cloud_list):
		l, = plt.plot(points[:,0], points[:,1], 'o', label='Line_'+str(idx))
		line_list.append(l)
	if not label_list is None:
		assert len(line_list) == len(label_list)
		plt.legend(line_list, label_list)
	if save:
		assert not path is None
		plt.savefig(path)
	else:
		plt.show()
	plt.clf()
	plt.close()


def plot_Spline_Curve_Control_Point_Point_Cloud(control_point_list, w, point_cloud_list,
                                            cp_label_list=None, pc_label_list=None, x_size=128, y_size=128,
											save=False, path=None):
	plt.axis([0, x_size, 0, y_size])
	line_list = []
	for control_points in control_point_list:
		line_point = np.dot(w, control_points)
		l, = plt.plot(line_point[:,0], line_point[:,1])
		plt.plot(control_points[:,0], control_points[:,1], 'o')
		for idx, xy in enumerate(zip(control_points[:,0], control_points[:,1])):
			plt.annotate(str(idx), xy)
		line_list.append(l)
	if not cp_label_list is None:
		assert len(cp_label_list) == len(line_list)
		plt.legend(line_list, cp_label_list)

	line_list = []
	for idx, points in enumerate(point_cloud_list):
		plt.scatter(points[:,0], points[:,1])
	# 	l, = plt.plot(points[:,0], points[:,1], 'o', label='Line_'+str(idx))
	# 	line_list.append(l)
	# if not pc_label_list is None:
	# 	assert len(line_list) == len(pc_label_list)
	# 	plt.legend(line_list, pc_label_list)
	if save:
		assert not path is None
		plt.savefig(path)
		plt.clf()
		plt.close()
		return
	plt.show()
	plt.clf()
	plt.close()

def plot_Spline_Line_Control_Point(control_point_list, w, line_point_list,
                                   cp_label_list=None, x_size=128, y_size=128,
									save=False, path=None):
	plt.axis([0, x_size, 0, y_size])
	line_list = []
	for control_points in control_point_list:
		line_point = np.dot(w, control_points)
		l, = plt.plot(line_point[:,0], line_point[:,1])
		plt.plot(control_points[:,0], control_points[:,1], 'o')
		for idx, xy in enumerate(zip(control_points[:,0], control_points[:,1])):
			plt.annotate(str(idx), xy)
		line_list.append(l)
	if not cp_label_list is None:
		assert len(cp_label_list) == len(line_list)
		plt.legend(line_list, cp_label_list)

	line_list = []
	for idx, line_points in enumerate(line_point_list):
		assert line_points.shape == (config.n_line, 4)
		for idx_l, l in enumerate(line_points):
			plt.plot([l[0],l[2]], [l[1],l[3]])
			plt.annotate(str(idx_l*2), (l[0],l[1]))
			plt.annotate(str(idx_l * 2+1), (l[2], l[3]))
	if save:
		assert not path is None
		plt.savefig(path)
		plt.clf()
		plt.close()
		return
	plt.show()
	plt.clf()
	plt.close()

def plot_Spline_Point_Control_Point(control_point_list, w, points_list,
                                   cp_label_list=None, x_size=128, y_size=128,
									save=False, path=None):
	plt.axis([0, x_size, 0, y_size])
	line_list = []
	for control_points in control_point_list:
		line_point = np.dot(w, control_points)
		l, = plt.plot(line_point[:,0], line_point[:,1])
		plt.plot(control_points[:,0], control_points[:,1], 'o')
		for idx, xy in enumerate(zip(control_points[:,0], control_points[:,1])):
			plt.annotate(str(idx), xy)
		line_list.append(l)
	if not cp_label_list is None:
		assert len(cp_label_list) == len(line_list)
		plt.legend(line_list, cp_label_list)

	for idx, points in enumerate(points_list):
		for idx_p, p in enumerate(points):
			plt.scatter(p[0], p[1])
			plt.annotate(str(idx)+'_'+str(idx_p), (p[0],p[1]))
	if save:
		assert not path is None
		plt.savefig(path)
		plt.clf()
		plt.close()
		return
	plt.show()
	plt.clf()
	plt.close()

def plot_Spline_Bezier_Control_Point(spline_cp_list, spline_w, bezier_cp_list, bezier_w,
                                   cp_label_list=None, x_size=128, y_size=128,
									save=False, path=None):
	plt.axis([0, x_size, 0, y_size])
	line_list = []
	for control_points in spline_cp_list:
		line_point = np.dot(spline_w, control_points)
		l, = plt.plot(line_point[:,0], line_point[:,1])
		plt.plot(control_points[:,0], control_points[:,1], 'o')
		for idx, xy in enumerate(zip(control_points[:,0], control_points[:,1])):
			plt.annotate(str(idx), xy)
		line_list.append(l)

	for idx_b, control_points in enumerate(bezier_cp_list):
		line_point = np.dot(bezier_w, control_points)
		l, = plt.plot(line_point[:,0], line_point[:,1])
		plt.plot(control_points[:,0], control_points[:,1], 'o')
		for idx, xy in enumerate(zip(control_points[:,0], control_points[:,1])):
			plt.annotate(str(idx_b)+'_'+str(idx), xy)
		line_list.append(l)


	if not cp_label_list is None:
		assert len(cp_label_list) == len(spline_cp_list) + len(bezier_cp_list)
		plt.legend(line_list, cp_label_list)

	if save:
		assert not path is None
		plt.savefig(path)
		plt.clf()
		plt.close()
		return
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
	s = Spline(n_point, n_degree)
	w = s.get_Nodes_Eval_Matrix(N)
	plot_Spline_Curve_Control_Point([control_points], w)
	plot_Spline_Curve_Point_Cloud([np.dot(w, control_points)])
	img = get_Spline_Matrix_Control_Point(control_points, w)
	plot_Spline_Img(img)
	point_cloud = sample_Point_Cloud(img.reshape(128,128,1))
	plot_Spline_Curve_Point_Cloud([point_cloud])
