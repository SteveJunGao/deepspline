import numpy as np
from utils import progress_bar, init_params
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
import config
import random

# from spline_Utils import plot_Curve, get_Linear_Point_Cloud, get_Img_Matrix_Point_Cloud, plot_Img
from spline_Utils import  plot_Spline_Curve_Control_Point_Multi_Set
# from spline_Utils_jiahui import plot_Spline_Curve_Control_Point
score_state = torch.load(os.path.join(config.checkpoint_path, 'prediction/result_epoch_%d'%(
						config.check_epoch)))

test_predict = score_state['test_predict']
test_label = score_state['test_label']

if not os.path.exists(config.check_save_path):
	os.makedirs(config.check_save_path)

if not os.path.exists(config.check_save_input_path):
	os.makedirs(config.check_save_input_path)

for i in range( config.check_num):
	batch_idx = int(i/config.test_batch_size)
	idx = i%config.test_batch_size

	label = test_label[batch_idx][idx].float().view(config.max_line, config.max_point, config.n_dim).numpy()*128
	if config.debugging: print(f"Extract Label: {label}")
	non_zero_pos = np.nonzero(label[:, 0, 0])
	n_line = np.max(non_zero_pos)+1
	label = label[:n_line,:,:]
	if config.debugging: print(f"After label: {label}")

	pos_pred, prob_pred = test_predict[batch_idx]
	pos_pred = pos_pred[idx].float().view(config.max_line, config.max_point, config.n_dim).numpy()*128
	prob_pred = prob_pred[idx*config.max_line : (idx+1)*config.max_line, 1].float().numpy()
	# if i < 55 and i > 50:
	# 	print(i)
	# 	print(f"Label is {label}")
	# 	print(f"Predict pos is {pos_pred}")
	# 	print(f"Predict prob is {prob_pred}")
	if config.debugging:
		print(f"Predict position: {pos_pred}")
		print(f"Predict probability:{prob_pred}")
	# if config.debugging: print(pos_pred, prob_pred)
	n_pred_line = 1
	for idx in range(config.max_line):
		if prob_pred[idx] < 0.5:
			n_pred_line += 1
	pos_pred = pos_pred[:n_pred_line,:,:]
	# print(pos_pred)

	# point_cloud = get_Linear_Point_Cloud(label, 1024)
	# img = get_Img_Matrix_Point_Cloud(point_cloud)
	label = [label[idx] for idx in range(n_line)]
	if i == 36:
		print(pos_pred)
		print(label)
	pos_pred = [pos_pred[idx] for idx in range(n_pred_line)]
	plot_Spline_Curve_Control_Point_Multi_Set(control_points_target = label, control_points_pred = pos_pred,
	                                target_label_list = ['Target']*n_line, pred_label_list=['Predict']*n_pred_line,save=True,
									path=os.path.join(config.check_save_path, str(i)+'.jpg'))
	# plot_Img(img, save=True, path=os.path.join(config.check_save_input_path, str(i)+'.jpg'))
	progress_bar(i, config.check_num)
