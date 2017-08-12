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
from spline_Utils import  plot_Spline_Curve_Control_Point
# from spline_Utils_jiahui import plot_Spline_Curve_Control_Point
score_state = torch.load(os.path.join(config.checkpoint_path, 'prediction/result_epoch_%d'%(
						config.check_epoch)))

test_predict = score_state['test_predict']
test_label = score_state['test_label']

plot_num = [0,4,22]
if not os.path.exists(config.check_save_path):
	os.makedirs(config.check_save_path)

if not os.path.exists(config.check_save_input_path):
	os.makedirs(config.check_save_input_path)

for i in range(config.check_num):
	batch_idx = int(i/config.test_batch_size)
	idx = i%config.test_batch_size

	label = test_label[batch_idx][idx].float().view(config.max_point, config.n_dim).numpy()*128
	# if config.debugging: print(label)
	non_zero_pos = np.nonzero(label[:,0])
	n_label_point = np.max(non_zero_pos)+1
	label = label[:n_label_point,:]
	# if config.debugging: print(label)
	pos_pred, prob_pred = test_predict[batch_idx]
	pos_pred = pos_pred[idx].float().view(config.max_point, config.n_dim).numpy()*128
	prob_pred = prob_pred[idx*config.max_point : (idx+1)*config.max_point, 1].float().numpy()
	if config.debugging: print(prob_pred)
	# if config.debugging: print(pos_pred, prob_pred)
	n_point = 1
	for idx in range(config.max_point):
		if prob_pred[idx] < 0.5:
			n_point += 1
	pos_pred = pos_pred[:n_point,:]
	if i in plot_num:
		print(f"Target: {label}")
		print(f"Predict: {pos_pred}")

	# point_cloud = get_Linear_Point_Cloud(label, 1024)
	# img = get_Img_Matrix_Point_Cloud(point_cloud)
	plot_Spline_Curve_Control_Point([label, pos_pred], label_list=['Target', 'Predict'],save=True,
									path=os.path.join(config.check_save_path, str(i)+'.jpg'))
	# plot_Img(img, save=True, path=os.path.join(config.check_save_input_path, str(i)+'.jpg'))
	progress_bar(i, config.check_num)
