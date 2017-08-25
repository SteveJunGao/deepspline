"""
This code is only used for visiualize the attention map of multispline curves
Therefore only be used for testing data

"""

import numpy as np
from dataset_Reader import Multi_Vary_Spline_Dataset
from utils import progress_bar, init_params, my_print
import torch
import os
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import torch.backends.cudnn as cudnn
import torchvision.transforms as transforms
from nn_model import ResNet34
from multi_vary_spline_model import Multi_Vary_Line_Model_v1, Multi_Vary_Line_Loss
from spline_Utils import  plot_attention
# from model import Multi_Point_Model, Multi_Point_Loss, Multi_Point_Scheduled_Sampling_Model
import config
import random
import sys
random.seed(config.manual_seed)
torch.manual_seed(config.manual_seed+1)


test_transform = transforms.Compose([
	transforms.ToTensor(),
	])


testset = Multi_Vary_Spline_Dataset(path = config.dataset_path, n_all = config.n_data,
			train_part=config.train_part, train=False, transform=test_transform)


testloader = torch.utils.data.DataLoader(testset, batch_size=config.test_batch_size,
			shuffle=False, num_workers=1)

print('Loading Model')

trained_net = torch.load(os.path.join(config.checkpoint_path, 'chkt_'+str(config.check_epoch)))['net']
net = Multi_Vary_Line_Model_v1(ss_prob = 1, fix_pretrained = config.fix_pretrained, vis_attention = True)
net.cuda()
print(trained_net)
print(net)
print('Assign Pretrained Value')
net.load_pretrained_weight(trained_net)
criterion = Multi_Vary_Line_Loss()
criterion.cuda()


if not os.path.exists(os.path.join(config.check_save_path, 'attention')):
	os.makedirs(os.path.join(config.check_save_path, 'attention'))

def vis_attention():
	net.eval()
	cul_mse_loss = 0
	cul_point_clf_loss = 0
	cul_spline_clf_loss = 0
	for batch_idx, (inputs, labels, line_masks, clf_line, point_masks, clf_point) in enumerate(testloader):
		inputs = inputs.float().cuda()
		labels = labels.float().cuda()
		line_masks = line_masks.float().cuda()
		clf_line = clf_line.long().cuda()
		point_masks = point_masks.float().cuda()
		clf_point = clf_point.long().cuda()
		inputs = Variable(inputs)
		labels = Variable(labels, requires_grad=False)
		line_masks = Variable(line_masks, requires_grad=False)
		clf_line = Variable(clf_line, requires_grad=False)
		point_masks = Variable(point_masks, requires_grad=False)
		clf_point = Variable(clf_point, requires_grad=False)

		point_pos_pred, point_prob_pred, line_prob_pred, attention_map= net(inputs, config.max_line)
		mse_loss, spline_clf_loss, point_clf_loss = criterion(point_pos_pred, point_prob_pred, line_prob_pred,
		                                                      labels, line_masks, clf_line, point_masks, clf_point)
		# loss = mse_loss + config.clf_weight * (2 * point_clf_loss + spline_clf_loss)
		attention_map = np.asarray(attention_map).squeeze()
		# print(attention_map.shape)
		attention_map = np.transpose(attention_map, (1,0,2,3))
		# print(attention_map.shape)
		line_prob_pred = line_prob_pred.view(inputs.size(0), config.max_line, 2).data.cpu().numpy()
		# print(line_prob_pred)
		# print(labels)
		for i in range(inputs.size(0)):
			n_pred_line = 1
			for idx in range(config.max_line):
				if line_prob_pred[i, idx, 1] < 0.5:
					n_pred_line += 1
				else:
					break
			pos_pred = point_pos_pred[i, :n_pred_line, :, :]*128
			# print(attention_map[i])
			pos_target = labels[i, :n_pred_line, :]*128
			# Deal With predict  control point prob here
			target = []
			pred = []
			# if n_pred_line == 1:
			# 	pos_pred = pos_pred.unsqueeze(0)
			# 	pos_target = pos_target.unsqueeze(0)
			for i_line in range(n_pred_line):
				n_true_point = int(torch.sum(point_masks[i, i_line]).data[0])
				target.append(pos_target[i_line, :n_true_point,:].data.cpu().numpy())
				n_pred_point = 1
				for i_p in range(config.max_point):
					if point_prob_pred[i, i_line, i_p, 1].data[0] < 0.5:
						n_pred_point += 1
					else: break
				pred.append(pos_pred[i_line, :n_pred_point, :].data.cpu().numpy())
			# print(pos_pred)
			# print(point_prob_pred)
			# print(pred)
			# print(pos_target)
			# print(target)
			plot_attention(inputs[i, 0].data.cpu().numpy(), attention_map[i, : n_pred_line], pred, target,
			               n_pred_line, save = True, path = os.path.join(config.check_save_path, 'attention', str(batch_idx*inputs.size(0)+i)+'.jpg'))


		cul_mse_loss += mse_loss.data[0]
		cul_point_clf_loss += point_clf_loss.data[0]
		cul_spline_clf_loss += spline_clf_loss.data[0]

		progress_bar(batch_idx, len(testloader), 'Loss: %.5f %.5f %.5f'%(cul_mse_loss/(batch_idx+1),
		                                                                 cul_spline_clf_loss/(batch_idx+1),
		                                                                 cul_point_clf_loss/(batch_idx+1)))




vis_attention()
