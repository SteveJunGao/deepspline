"""
This code is only used for visiualize the attention map of multispline curves
Therefore only be used for testing data

"""

import numpy as np
from dataset_Reader import Multi_Spline_Dataset_v1
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
from multi_spline_model import  Multi_Line_Model_v1, Multi_Line_Loss, Multi_Line_Model_on_going, Multi_Line_Model_on_going_2
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


testset = Multi_Spline_Dataset_v1(path = config.dataset_path, n_all = config.n_data,
			train_part=config.train_part, train=False, transform=test_transform)


testloader = torch.utils.data.DataLoader(testset, batch_size=2,
			shuffle=False, num_workers=1)

print('Loading Model')

trained_net = torch.load(os.path.join(config.checkpoint_path, 'chkt_'+str(config.check_epoch)))['net']
net = Multi_Line_Model_v1(ss_prob = 1, fix_pretrained = config.fix_pretrained, vis_attention = True)
net.cuda()
print(trained_net)
print('Assign Pretrained Value')
net.load_pretrained_weight(trained_net)
criterion = Multi_Line_Loss()
criterion.cuda()


if not os.path.exists(os.path.join(config.check_save_path, 'attention')):
	os.makedirs(os.path.join(config.check_save_path, 'attention'))
#
# print(net.attention.ModuleList.Conv2d)
# print(net.attention.weight.data.cpu().squeeze()[1:100])
# print(net.attention.weight.data.cpu().squeeze()[-100:])
def vis_attention():
	net.eval()
	test_loss = 0.0
	cul_mse_loss = 0
	cul_clf_loss = 0
	loss = 0
	n_batch = 0
	for batch_idx, (inputs, labels, line_masks, clf_line) in enumerate(testloader):
		inputs = inputs.float().cuda()
		labels = labels.float().cuda()
		line_masks = line_masks.float().cuda()
		clf_line = clf_line.long().cuda()
		inputs = Variable(inputs)
		labels = Variable(labels, requires_grad=False)
		line_masks = Variable(line_masks, requires_grad=False)
		clf_line = Variable(clf_line, requires_grad=False)

		pred_pos, pred_prob, attention_map = net(inputs, config.max_line, labels, clf_line)
		attention_map = np.asarray(attention_map).squeeze()
		# print(attention_map.shape)
		attention_map = np.transpose(attention_map, (1,0,2,3))
		# print(attention_map.shape)
		pred_prob = pred_prob.view(inputs.size(0), config.max_line, 2).data.cpu().numpy()
		labels = labels.data.cpu().numpy()
		# print(pred_prob)
		# print(pred_pos)
		for i in range(inputs.size(0)):
			n_pred_line = 1
			for idx in range(config.max_line):
				if pred_prob[i, idx, 1] < 0.5:
					n_pred_line += 1
				else:
					break
			pos_pred = pred_pos[i, :n_pred_line, :]
			# print(attention_map[i])
			pos_target = labels[i, :n_pred_line, :]
			plot_attention(inputs[i, 0].data.cpu().numpy(), attention_map[i, : n_pred_line], pos_pred.data.cpu().numpy(), pos_target,
			               n_pred_line, save = True, path = os.path.join(config.check_save_path, 'attention', str(batch_idx*inputs.size(0)+i)+'.jpg'))



		# test_predict.append((pred_pos.cpu().data, pred_prob.cpu().data))
		progress_bar(batch_idx, len(testloader))
		# mse_loss, clf_loss = criterion(pred_pos, pred_prob, labels, line_masks, clf_line)
		# loss = mse_loss + config.clf_weight*clf_loss
		# test_loss += loss.data[0]
		# cul_mse_loss += mse_loss.data[0]
		# cul_clf_loss += clf_loss.data[0]
		# n_batch += 1
		# progress_bar(batch_idx, len(testloader), 'Loss: %.5f %.5f %.5f %.5f %.5f %.5f'
		#              % (mse_loss.data[0], clf_loss.data[0], loss.data[0],
		#                 cul_mse_loss / (batch_idx + 1), cul_clf_loss / (batch_idx + 1), test_loss/(batch_idx+1)))


	# if not os.path.isdir(config.checkpoint_path):
	# 	os.makedirs(config.checkpoint_path)
	# torch.save(state, os.path.join(config.checkpoint_path, 'chkt_%d'%(epoch)))
	# score_state = {
	# 	'test_predict': test_predict,
	# 	'test_label': test_label,
	# }
	# test_predict = []
	# test_label = []
	# if not os.path.isdir(os.path.join(config.checkpoint_path, 'prediction')):
	# 	os.makedirs(os.path.join(config.checkpoint_path, 'prediction'))
	# torch.save(score_state, os.path.join(config.checkpoint_path, 'prediction/result_epoch_%d'%(epoch)))
	# print('Saved at %s'%(config.checkpoint_path))




vis_attention()
