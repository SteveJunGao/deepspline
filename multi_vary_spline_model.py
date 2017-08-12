import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.autograd import Variable
import config
from sklearn.neighbors import NearestNeighbors, KDTree
from scipy.optimize import linear_sum_assignment
from nn_model import ResNet34
import torchvision.models as th_mdl

class MyModuleList(nn.Module):
	def __init__(self, nn_list):
		super(MyModuleList, self).__init__()
		self.nn_list = nn.ModuleList(nn_list)

	def forward(self, x):
		for n in self.nn_list:
			x = n(x)
		return x

class Multi_Vary_Line_Model_v1(nn.Module):
	'''
	This model is used to predict multiple spline with variable number of control points
	two RNN will be used
	Scheduled Sampling will not be used, because of no ground truth. Therefore, I wrongly used scheduled sampling before....
	'''
	def __init__(self, ss_prob = 0.1, fix_pretrained=False, vis_attention=False, no_attention = False):
		super(Multi_Vary_Line_Model_v1, self).__init__()
		vgg = th_mdl.vgg16(pretrained = True)
		self.img_feature = vgg.features
		# self.img_feature = ResNet34()
		self.vis_attention = vis_attention
		self.no_attention = no_attention
		if fix_pretrained:
			for p in self.img_feature.parameters():
				# print(p)
				p.requires_grad = False
		# self.img_feature = ResNet34()
		self.img_feature_size = 512
		self.spline_hidden_size = 512
		self.point_hidden_size = 512

		self.hidden_predion = MyModuleList(self.hidden_pred_list(self.img_feature_size, self.spline_hidden_size))
		self.attention = MyModuleList(self.attention_list(self.img_feature_size+self.spline_hidden_size)) # Generate the output attention map

		self.spline_gru = nn.GRU(input_size = self.img_feature_size, hidden_size = self.spline_hidden_size)
		self.point_gru = nn.GRU(input_size = self.spline_hidden_size, hidden_size = self.point_hidden_size)
		self.point_pred = nn.Linear(self.point_hidden_size, 4) # Predict the point position and point stop probability
		self.line_pred = nn.Linear(self.spline_hidden_size, 2) # Predict spline stop probability


	def attention_list(self, input_size):
		att = []
		att.append(nn.Conv2d(input_size, 256, 1))
		att.append(nn.ReLU())
		att.append(nn.Conv2d(256, 1, 1))
		return att

	def hidden_pred_list(self, input_size, output_size):
		hidden_pred = []
		hidden_pred.append(nn.Linear(input_size, 256))
		hidden_pred.append(nn.ReLU())
		hidden_pred.append(nn.Linear(256, output_size))
		return hidden_pred


	def forward(self, imgs, n_lines):
		'''
		:param imgs: a batch of images: [batch_size, 3, 128, 128]
		:param n_lines: maximum number of line
		:return:
		'''
		img_features = self.img_feature(imgs)
		batch_size = img_features.size(0)
		point_pos_output = Variable(torch.zeros(batch_size, n_lines, config.max_point, config.n_dim)).cuda()
		point_prob_output = Variable(torch.zeros(batch_size, n_lines, config.max_point, 2)).cuda()
		line_prob_output = Variable(torch.zeros(batch_size, n_lines, 2)).cuda()
		if self.vis_attention:
			attention_map = []

		for idx in range(n_lines):
			if idx == 0:
				spline_hidden = self.init_spline_hidden(img_features) # hidden: init hidden vector

			# print(hidden[0, 0, 1:10])
			if self.no_attention:
				att_output = Variable(torch.zeros(batch_size, 1, img_features.size(2), img_features.size(3)).cuda()+1/(img_features.size(2)*img_features.size(3)))
			else:
				hidden_map = spline_hidden.view(batch_size, self.spline_hidden_size, 1, 1).expand(batch_size, self.spline_hidden_size,
				                                                                    img_features.size(2),
				                                                                    img_features.size(3)).contiguous()
				att_input = torch.cat((img_features, hidden_map), 1)
				att_output = self.attention(att_input)
				# att_output = F.sigmoid(att_output)
				# print(att_output[0, 0])
				# att_output = att_output.view(batch_size, -1)
				# att_output = F.softmax(att_output) * self.beta
			if config.debugging:
				print(f"Image Feature size is {img_features.size()}")
				print(f"attention output size is {att_output.size()}")
			att_output = att_output.view(batch_size, 1, img_features.size(2), img_features.size(3))
			if self.vis_attention: attention_map.append(att_output.data.cpu().numpy())
			att_img_feature = torch.mul(img_features, att_output.expand_as(img_features))
			att_img_feature = torch.sum(att_img_feature.view(batch_size, self.img_feature_size, -1), 2).squeeze().unsqueeze(0)
			if config.debugging:
				print(f"attention Image Feature is {att_img_feature.size()}")

			# RNN Recurrent
			spline_features, spline_hidden = self.spline_gru(att_img_feature, spline_hidden)

			## Predict spline stop prob.
			spline_features = spline_features.view(batch_size, self.spline_hidden_size)
			line_stop_p = self.line_pred(F.relu(spline_features))
			line_prob_output[:, idx, :] = line_stop_p

			## Predict control point position ans stop prob.
			point_hidden = self.init_hidden_point(batch_size)
			for id_p in range(config.max_point):
				point_features, point_hidden = self.point_gru(spline_features.unsqueeze(0), point_hidden)
				point_features = point_features.view(batch_size, self.point_hidden_size)
				point_pred = self.point_pred(F.relu(point_features))
				point_pos_output[:, idx, id_p, :] = point_pred[:, 0:2]
				point_prob_output[:, idx, id_p, :] = point_pred[:, 2:4]

		# print(point_prob_output.size(), point_pos_output.size())
		line_prob_output = line_prob_output.view(-1, 2)
		line_prob_output = F.softmax(line_prob_output)
		point_prob_output = point_prob_output.view(-1, 2)
		point_prob_output = F.softmax(point_prob_output)
		line_prob_output = line_prob_output.view(batch_size, n_lines, 2)
		point_prob_output = point_prob_output.view(batch_size, n_lines, config.max_point, 2)
		if self.vis_attention:
			return point_pos_output, point_prob_output, line_prob_output, attention_map
		return point_pos_output, point_prob_output, line_prob_output

	def init_spline_hidden(self, img_feature):
		# Use the average of image feature to predict the first hidden layer.
		n_feature = img_feature.size(2)*img_feature.size(3)
		avg_img_feature = img_feature.data / n_feature
		avg_img_feature = avg_img_feature.view(img_feature.size(0), img_feature.size(1), -1)
		avg_img_feature = torch.sum(avg_img_feature, 2).squeeze()
		avg_img_feature = Variable(avg_img_feature)
		hidden = self.hidden_predion(avg_img_feature)
		hidden = hidden.unsqueeze(0)
		return hidden

	def init_hidden_point(self, batch_size):
		result = Variable(torch.zeros(1, batch_size, self.point_hidden_size)).cuda()
		return result

	def load_pretrained_weight(self, pretrained_net):
		self.load_state_dict(pretrained_net.state_dict())

class Multi_Vary_Line_Loss(nn.Module):
	def __init__(self):
		super(Multi_Vary_Line_Loss, self).__init__()
		self.clf_criterion = nn.CrossEntropyLoss(size_average = False)
	def forward(self, point_pos_pred, point_prob_pred, line_prob_pred,
		        target_pos, line_masks, clf_line, point_masks, clf_point):
		'''
		:param point_pos_pred: [batch_size, n_line, n_point, n_dim]
		:param point_prob_pred:  [batch_size, n_line, n_point, 2]
		:param line_prob_pred:  [batch_size, n_line, 2]
		:param target_pos:  [batch_size, n_line, n_point, n_dim]
		:param line_masks:  [batch_size, n_line]
		:param clf_line:  [batch_size, n_line]
		:param point_masks:  [batch_size, n_line, n_point]
		:param clf_point: [batch_size, n_line, n_point]
		:return:
		'''
		if config.debugging:
			print(point_pos_pred)
			print(point_prob_pred)
			print(line_prob_pred)
			print(target_pos)
			print(line_masks)
			print(clf_line)
			print(point_masks)
			print(clf_point)
		# Match the ground truth label
		target_pos, point_masks, clf_point = match_lines_sequence(point_pos_pred, target_pos, line_masks, point_masks, clf_point)
		# print(type(target_pos))
		# print(target_pos.cpu())
		square_mse = torch.sum(torch.pow(point_pos_pred - target_pos, 2), dim=3).squeeze()*point_masks
		# if config.debugging: print(square_mse.cpu())
		mask_mse = torch.sum(square_mse) / (torch.sum(point_masks)) * 5
		# print(square_mse.size())

		#Line Classification loss
		line_prob_pred = line_prob_pred.view(-1, 2)
		clf_line = clf_line.view(-1, 1)
		# if config.debugging: print(clf_point)
		lg_sftmx = -torch.log(line_prob_pred)
		# if config.debugging: print(lg_sftmx)
		line_clf_loss = torch.gather(lg_sftmx, 1, clf_line) * line_masks.view(-1,1)
		# if config.debugging: print(clf_loss)
		line_clf_loss = torch.sum(line_clf_loss) / (torch.sum(line_masks))

		#Point Classification loss
		point_prob_pred = point_prob_pred.view(-1, 2)
		clf_point = clf_point.view(-1, 1)
		lg_sftmx = -torch.log(point_prob_pred)
		point_clf_loss = torch.gather(lg_sftmx, 1, clf_point)*point_masks.view(-1, 1)
		point_clf_loss = torch.sum(point_clf_loss) / torch.sum(point_masks)
		return mask_mse, line_clf_loss, point_clf_loss

def match_lines_sequence(pos_pred, target_pos, line_mask, point_masks, clf_point):
	"""
	Match the prediction line and target line, and return the matched ground truth.
	Using Hungarian Algorithm to optimize it
	:param pos_pred:
	:param target_pos:
	:param line_mask:
	:return:

	"""
	batch_size = pos_pred.size(0)
	pos_t = torch.zeros(batch_size, config.max_line, config.max_point, config.n_dim).cuda()
	pos_m = torch.zeros(batch_size, config.max_line, config.max_point).cuda()
	pos_c = torch.zeros(batch_size, config.max_line, config.max_point).long().cuda()

	for i_batch in range(batch_size):
		i_n_line = int(line_mask[i_batch].sum().data[0])
		if config.debugging: print(f"Number of Line = {i_n_line}")
		if i_n_line == 1:
			pos_t[i_batch] = target_pos[i_batch].data
			pos_m[i_batch] = point_masks[i_batch].data
			pos_c[i_batch] = clf_point[i_batch].data
			continue
		if config.debugging: print(i_n_line)
		i_pos_pred = pos_pred[i_batch].data.cpu()[:int(i_n_line)]
		i_pos_target = target_pos[i_batch].data.cpu()[:int(i_n_line)]
		i_point_mask = point_masks[i_batch].data.cpu()[:int(i_n_line)]
		if config.debugging:
			print(i_pos_pred)
			print(i_pos_target)
			print(line_mask[i_batch])

		# Just enumearte to get the Euclidean mask, because of low computational cost
		Euclidean_m = np.zeros((i_n_line, i_n_line))
		for i_pred in range(i_n_line):
			for i_target in range(i_n_line):
				p = i_pos_pred[i_pred]
				t = i_pos_target[i_target]
				if config.debugging: print(f"Before Mask p: {p}")
				n_point = torch.sum(i_point_mask[i_target])

				p = p * i_point_mask[i_target].unsqueeze(1).expand_as(p)
				if config.debugging: print(f"Masked P: {p}")
				dis = p - t
				dis = torch.sum(dis*dis) / n_point
				Euclidean_m[i_pred, i_target] = dis
		if config.debugging:
			print(Euclidean_m)
		row_ind, col_ind = linear_sum_assignment(Euclidean_m)
		# In this case, row_ind is the same as np.arrange(Eiclidean_m.shape(0))
		if config.debugging: print(row_ind, col_ind)
		col_ind = torch.from_numpy(col_ind).cuda().long()
		tmp_target = target_pos[i_batch][col_ind]
		pos_t[i_batch, :i_n_line] = tmp_target.data
		pos_m[i_batch, :i_n_line] = point_masks[i_batch][col_ind].data
		pos_c[i_batch, :i_n_line] = clf_point[i_batch][col_ind].data
		if config.debugging:
			print(f"After, pos target is {pos_t}")
			print(f"After, pos mask is {pos_m}")
			print(f"After, pos clf is {pos_c}")

	return Variable(pos_t, requires_grad = False), Variable(pos_m, requires_grad=False), Variable(pos_c, requires_grad=False)

