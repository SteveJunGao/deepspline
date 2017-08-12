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

class MSE_Predict(nn.Module):
	def __init__(self):
		super(MSE_Predict, self).__init__()
		self.img_feature = ResNet34()
		self.hidden_size = 512
		self.gru = nn.GRU(input_size=256*4, hidden_size=self.hidden_size)
		self.predict = nn.Linear(self.hidden_size, 2) # Predict two position here.

	def forward(self, imgs, n_points):
		features = self.img_feature(imgs)
		batch_size = features.size(0)
		hidden = self.init_hidden(batch_size)
		features = features.view(1, batch_size, 256*4)
		outputs = Variable(torch.zeros(batch_size, n_points*config.n_dim)).cuda()

		for idx in range(n_points):
			output, hidden = self.gru(features, hidden)
			output = output.view(batch_size, self.hidden_size)
			# print(output.size())
			output = self.predict(F.relu(output))
			# print(output.size())
			outputs[:, idx*2:(idx+1)*2] = output
		return outputs

	def load_pretrain_weight(self, model_name):
		# Load a pretrained network
		pass

	def init_hidden(self, batch_size):
		result = Variable(torch.zeros(1, batch_size, self.hidden_size), requires_grad=False).cuda()


class Chamfer_Distance(nn.Module):
	def __init__(self, N):
		super(Chamfer_Distance, self).__init__()
		self.N = N

	def forward(self, pred, target):
		pred = pred.view(-1, self.N, config.n_dim)
		target = target.view(-1, self.N, config.n_dim)

		assert  pred.size(0) == target.size(0)
		batch_size = pred.size(0)

		###################################################
		# Use Nearest Neighbor
		###################################################
		target_numpy = target.data.cpu().numpy()
		pred_numpy = pred.data.cpu().numpy()
		# Trying to use CPU to compute NN.
		target_nn = Variable(torch.FloatTensor(batch_size, self.N, 2).cuda())
		for i in range(batch_size):
			nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(target_numpy[i, :, :])
			sample_index = torch.from_numpy(nbrs.kneighbors(pred_numpy[i, :, :])[1].reshape(self.N)).cuda()
			# comma segments each dimension.
			target_nn[i, :, :] = target[i, :, :].index_select(0, Variable(sample_index))
		# Get the l2 Euclidean distance matrix of two point clouds
		pred_to_target = torch.sum(torch.pow(target_nn - pred, 2))

		pred_nn = Variable(torch.FloatTensor(batch_size, self.N, 2).cuda())
		for i in range(batch_size):
			nbrs = NearestNeighbors(n_neighbors=1, algorithm='auto').fit(pred_numpy[i, :, :])
			sample_index = torch.from_numpy(nbrs.kneighbors(target_numpy[i, :, :])[1].reshape(self.N)).cuda()
			# print(sample_index)
			# comma segments each dimension.
			pred_nn[i, :, :] = pred[i, :, :].index_select(0, Variable(sample_index))
		# Get the l2 Euclidean distance matrix of two point clouds
		target_to_pred = torch.sum(torch.pow(pred_nn - target, 2))

		return pred_to_target / (self.N * pred.data.size(0)), target_to_pred / (self.N * pred.data.size(0))

class MyModuleList(nn.Module):
	def __init__(self, nn_list):
		super(MyModuleList, self).__init__()
		self.nn_list = nn.ModuleList(nn_list)

	def forward(self, x):
		for n in self.nn_list:
			# print(n.weight)
			# print(x)
			x = n(x)
		return x
	# def cuda(self, device_id=None):
	# 	self.nn_list = [n.cuda(device_id) for n in self.nn_list]

class Multi_Line_Model_v1(nn.Module):
	"""
	This model will only contain one RNN which would predict one spline at a step.
	The number of control point is fixed here.
	Attention Module would be used here.
	Importance Sampling will also be used to improve performance.

	"""
	def __init__(self, ss_prob = 0.1, fix_pretrained=False, vis_attention=False, no_attention = False):
		super(Multi_Line_Model_v1, self).__init__()
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
		self.feature_size = 512
		self.hidden_size = 512
		self.output_size = config.max_point * config.n_dim + 2
		self.rnn_input_size = self.feature_size + self.output_size
		self.hidden_predion = MyModuleList(self.hidden_pred_list(self.feature_size, self.hidden_size))
		self.gru = nn.GRU(input_size=self.rnn_input_size, hidden_size=self.hidden_size)
		self.attention = MyModuleList(self.attention_list(self.feature_size+self.hidden_size)) # Generate the output attention map
		# self.dropout_1 = nn.Dropout(0.5)
		self.predict_linear1 = nn.Linear(self.hidden_size, 512)
		# self.dropout_2 = nn.Dropout(0.5)
		self.predict_linear2 = nn.Linear(512, self.output_size) # Predict the position and existence probability
		self.ss_prob = ss_prob # probability to choose from ground truth
		self.beta = 1 # manually set to 2, will be learnt later

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


	def forward(self, imgs, n_lines, ground_truth, line_prob):
		'''

		:param imgs: a batch of images: [batch_size, 1, 128, 128]
		:param n_lines: maximum number of line
		:param ground_truth: ground truth of point position [batch_size, max_line, max_point*n_dim]
		:param line_prob:
		:return:
		'''
		img_features = self.img_feature(imgs)
		batch_size = img_features.size(0)
		point_pos_output = Variable(torch.zeros(batch_size, n_lines, config.max_point*config.n_dim)).cuda()
		line_prob_output = Variable(torch.zeros(batch_size, n_lines, 2)).cuda()
		if self.vis_attention:
			attention_map = []

		for idx in range(n_lines):
			if idx == 0:
				hidden, previous_output = self.init_hidden(batch_size, img_features) # hidden: init hidden vector, pred: init previous prediction
				ss_input = previous_output
			else:
				# Scheduled Sampling
				pos_gt = ground_truth.data[:, idx - 1, :].clone()
				pos_gt.view(-1, config.max_point*config.n_dim)
				prob_gt = line_prob.data[:, idx - 1].clone()
				prob_gt = prob_gt.float().view(-1, 1)
				inv_prob_gt = 1 - prob_gt
				ss_input = torch.cat((pos_gt, inv_prob_gt, prob_gt), 1)
				if self.ss_prob > 0.0:
					sample_prob = torch.zeros(batch_size).uniform_(0,1).cuda()
					sample_mask = sample_prob < self.ss_prob
					if sample_mask.sum() != 0:
						sample_ind = sample_mask.nonzero().view(-1)
						ss_input.index_copy_(0, sample_ind, previous_output.data.index_select(0, sample_ind))
				ss_input = Variable(ss_input, requires_grad = False).cuda()
				ss_input = ss_input.unsqueeze(0)
				if config.debugging:
					print(f"SS Input size is {ss_input.size()}")
			## Attention Module
			# print(hidden[0, 0, 1:10])
			hidden_map = hidden.view(batch_size, self.hidden_size, 1, 1).expand(batch_size, self.hidden_size, img_features.size(2), img_features.size(3)).contiguous()
			# print(hidden_map[0, 1:10, 0, 0])
			# print(hidden_map[0, 1:10, 1, 0])
			# print(hidden_map[0, 1:10, 0, 1])
			# print(hidden_map[0, 1:10, 1, 1])
			att_input = torch.cat((img_features, hidden_map), 1)
			if self.no_attention:
				att_output = Variable(torch.zeros(batch_size, 1, img_features.size(2), img_features.size(3)).cuda()+1)
			else:
				att_output = self.attention(att_input)
			att_output = F.sigmoid(att_output)
			# print(att_output[0, 0])
			# att_output = att_output.view(batch_size, -1)
			# att_output = F.softmax(att_output) * self.beta
			if config.debugging:
				print(f"Image Feature size is {img_features.size()}")
				print(f"attention output size is {att_output.size()}")
			att_output = att_output.view(batch_size, 1, img_features.size(2), img_features.size(3))
			if self.vis_attention: attention_map.append(att_output.data.cpu().numpy())
			att_img_feature = torch.mul(img_features, att_output.expand_as(img_features))
			att_img_feature = torch.sum(att_img_feature.view(batch_size, self.feature_size, -1), 2).squeeze().unsqueeze(0)
			if config.debugging:
				print(f"attention Image Feature is {att_img_feature.size()}")

			# RNN Recurrent
			rnn_input_vec = torch.cat((att_img_feature, ss_input), 2)
			output, hidden = self.gru(rnn_input_vec, hidden)

			## Predict output position and line probability
			output = output.view(batch_size, self.hidden_size)
			# print(output.size())
			# output = self.dropout_1(output)
			output = self.predict_linear1(F.relu(output))
			# output = self.dropout_2(output)
			output = self.predict_linear2(F.relu(output))

			# print(output.size())
			point_pos_output[:, idx, :] = output[:, 0:config.max_point*config.n_dim]
			line_prob_output[:, idx, :] = output[:, config.max_point*config.n_dim:]
			# Replace RNN output Vector
			previous_output = output

		# print(point_prob_output.size(), point_pos_output.size())
		line_prob_output = line_prob_output.view(-1, 2)
		line_prob_output = F.softmax(line_prob_output)
		if self.vis_attention:
			return point_pos_output, line_prob_output, attention_map
		return point_pos_output, line_prob_output

	def init_hidden(self, batch_size, img_feature):
		# Use the average of image feature to predict the first hidden layer.
		n_feature = img_feature.size(2)*img_feature.size(3)
		avg_img_feature = img_feature.data / n_feature
		avg_img_feature = avg_img_feature.view(img_feature.size(0), img_feature.size(1), -1)
		avg_img_feature = torch.sum(avg_img_feature, 2).squeeze()
		avg_img_feature = Variable(avg_img_feature)
		hidden = self.hidden_predion(avg_img_feature)
		pred = Variable(torch.zeros(1, batch_size, self.output_size)).cuda()
		hidden = hidden.unsqueeze(0)
		return hidden, pred

	def load_pretrained_weight(self, pretrained_net):
		self.load_state_dict(pretrained_net.state_dict())

class Multi_Line_Model_on_going(nn.Module):
	"""
	This model is used for testing different modifications w.r.t the newest version
	I would seperate the image feature branch s.t the attention module will not influence the gradient
	"""
	def __init__(self, fix_pretrained=False):
		super(Multi_Line_Model_on_going, self).__init__()
		self.img_feature = ResNet34()
		self.feature_size = 512
		self.hidden_size = 512
		self.output_size = config.max_point * config.n_dim + 2
		self.rnn_input_size = self.feature_size + self.output_size

		self.gru = nn.GRU(input_size=self.rnn_input_size, hidden_size=self.hidden_size)
		self.attention = nn.Conv2d(self.feature_size+self.hidden_size, 1, 1) # Generate the output attention map
		self.predict_linear1 = nn.Linear(self.hidden_size, 512)
		self.predict_linear2 = nn.Linear(512, self.output_size) # Predict the position and existence probability
		self.ss_prob = 0.1 # probability to choose from ground truth
		self.beta = 2 # manually set to 2, will be learnt later

	def forward(self, imgs, n_lines, ground_truth, line_prob):
		'''

		:param imgs: a batch of images: [batch_size, 1, 128, 128]
		:param n_lines: maximum number of line
		:param ground_truth: ground truth of point position [batch_size, max_line, max_point*n_dim]
		:param line_prob:
		:return:
		'''
		img_features = self.img_feature(imgs)
		img_features_for_att = Variable(img_features.data.clone(), requires_grad = False)

		batch_size = img_features.size(0)
		point_pos_output = Variable(torch.zeros(batch_size, n_lines, config.max_point*config.n_dim)).cuda()
		line_prob_output = Variable(torch.zeros(batch_size, n_lines, 2)).cuda()

		for idx in range(n_lines):
			if idx == 0:
				hidden, previous_output = self.init_hidden(batch_size) # hidden: init hidden vector, pred: init previous prediction
				ss_input = previous_output
			else:
				# Scheduled Sampling
				pos_gt = ground_truth.data[:, idx - 1, :].clone()
				pos_gt.view(-1, config.max_point*config.n_dim)
				prob_gt = line_prob.data[:, idx - 1].clone()
				prob_gt = prob_gt.float().view(-1, 1)
				inv_prob_gt = 1 - prob_gt
				ss_input = torch.cat((pos_gt, inv_prob_gt, prob_gt), 1)
				if self.ss_prob > 0.0:
					sample_prob = torch.zeros(batch_size).uniform_(0,1).cuda()
					sample_mask = sample_prob < self.ss_prob
					if sample_mask.sum() != 0:
						sample_ind = sample_mask.nonzero().view(-1)
						ss_input.index_copy_(0, sample_ind, previous_output.data.index_select(0, sample_ind))
				ss_input = Variable(ss_input, requires_grad = False).cuda()
				ss_input = ss_input.unsqueeze(0)
				if config.debugging:
					print(f"SS Input size is {ss_input.size()}")
			## Attention Module
			hidden_map = hidden.view(batch_size, self.hidden_size, 1, 1).expand(batch_size, self.hidden_size, img_features.size(2), img_features.size(3))
			hidden_map_for_att = Variable(hidden_map.data.clone(), requires_grad=False)
			att_input = torch.cat((img_features_for_att, hidden_map_for_att), 1)
			att_output = self.attention(att_input)
			att_output = att_output.view(batch_size, -1)
			att_output = F.softmax(att_output) * self.beta
			if config.debugging:
				print(f"Image Feature size is {img_features.size()}")
				print(f"attention output size is {att_output.size()}")
			att_output = att_output.view(batch_size, 1, img_features.size(2), img_features.size(3))
			att_img_feature = torch.mul(img_features, att_output.expand_as(img_features))
			att_img_feature = torch.sum(att_img_feature.view(batch_size, self.feature_size, -1), 2).squeeze().unsqueeze(0)
			if config.debugging:
				print(f"attention Image Feature is {att_img_feature.size()}")

			# RNN Recurrent
			rnn_input_vec = torch.cat((att_img_feature, ss_input), 2)
			output, hidden = self.gru(rnn_input_vec, hidden)

			## Predict output position and line probability
			output = output.view(batch_size, self.hidden_size)
			# print(output.size())
			output = self.predict_linear1(F.relu(output))
			output = self.predict_linear2(F.relu(output))

			# print(output.size())
			point_pos_output[:, idx, :] = output[:, 0:config.max_point*config.n_dim]
			line_prob_output[:, idx, :] = output[:, config.max_point*config.n_dim:]
			# Replace RNN output Vector
			previous_output = output

		# print(point_prob_output.size(), point_pos_output.size())
		line_prob_output = line_prob_output.view(-1, 2)
		line_prob_output = F.softmax(line_prob_output)
		return point_pos_output, line_prob_output

	def init_hidden(self, batch_size):
		hidden = Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()
		pred = Variable(torch.zeros(1, batch_size, self.output_size),
		                requires_grad = False).cuda()
		return hidden, pred


class Multi_Line_Model_on_going_2(nn.Module):
	"""
	This model will only contain one RNN which would predict one spline at a step.
	The number of control point is fixed here.
	Attention Module would be used here.
	Importance Sampling will also be used to improve performance.

	"""
	def __init__(self, fix_pretrained=False):
		super(Multi_Line_Model_on_going_2, self).__init__()
		self.img_feature = ResNet34()
		self.feature_size = 512
		self.hidden_size = 512
		self.output_size = config.max_point * config.n_dim + 2
		self.rnn_input_size = self.feature_size + self.output_size

		self.lstm = nn.LSTM(input_size=self.rnn_input_size, hidden_size=self.hidden_size)
		self.attention = nn.Conv2d(self.feature_size+self.hidden_size, 1, 1) # Generate the output attention map
		self.predict_linear1 = nn.Linear(self.hidden_size, 512)
		self.predict_linear2 = nn.Linear(512, self.output_size) # Predict the position and existence probability
		self.ss_prob = 0.1 # probability to choose from ground truth
		self.beta = 2 # manually set to 2, will be learnt later

	def forward(self, imgs, n_lines, ground_truth, line_prob):
		'''

		:param imgs: a batch of images: [batch_size, 1, 128, 128]
		:param n_lines: maximum number of line
		:param ground_truth: ground truth of point position [batch_size, max_line, max_point*n_dim]
		:param line_prob:
		:return:
		'''
		img_features = self.img_feature(imgs)
		batch_size = img_features.size(0)
		point_pos_output = Variable(torch.zeros(batch_size, n_lines, config.max_point*config.n_dim)).cuda()
		line_prob_output = Variable(torch.zeros(batch_size, n_lines, 2)).cuda()

		for idx in range(n_lines):
			if idx == 0:
				hidden, previous_output, c = self.init_hidden(batch_size) # hidden: init hidden vector, pred: init previous prediction
				ss_input = previous_output
			else:
				# Scheduled Sampling
				pos_gt = ground_truth.data[:, idx - 1, :].clone()
				pos_gt.view(-1, config.max_point*config.n_dim)
				prob_gt = line_prob.data[:, idx - 1].clone()
				prob_gt = prob_gt.float().view(-1, 1)
				inv_prob_gt = 1 - prob_gt
				ss_input = torch.cat((pos_gt, inv_prob_gt, prob_gt), 1)
				if self.ss_prob > 0.0:
					sample_prob = torch.zeros(batch_size).uniform_(0,1).cuda()
					sample_mask = sample_prob < self.ss_prob
					if sample_mask.sum() != 0:
						sample_ind = sample_mask.nonzero().view(-1)
						ss_input.index_copy_(0, sample_ind, previous_output.data.index_select(0, sample_ind))
				ss_input = Variable(ss_input, requires_grad = False).cuda()
				ss_input = ss_input.unsqueeze(0)
				if config.debugging:
					print(f"SS Input size is {ss_input.size()}")
			## Attention Module
			hidden_map = hidden.view(batch_size, self.hidden_size, 1, 1).expand(batch_size, self.hidden_size, img_features.size(2), img_features.size(3))

			att_input = torch.cat((img_features, hidden_map), 1)
			att_output = self.attention(att_input)
			att_output = att_output.view(batch_size, -1)
			att_output = F.softmax(att_output) * self.beta
			if config.debugging:
				print(f"Image Feature size is {img_features.size()}")
				print(f"attention output size is {att_output.size()}")
			att_output = att_output.view(batch_size, 1, img_features.size(2), img_features.size(3))
			att_img_feature = torch.mul(img_features, att_output.expand_as(img_features))
			att_img_feature = torch.sum(att_img_feature.view(batch_size, self.feature_size, -1), 2).squeeze().unsqueeze(0)
			if config.debugging:
				print(f"attention Image Feature is {att_img_feature.size()}")

			# RNN Recurrent
			rnn_input_vec = torch.cat((att_img_feature, ss_input), 2)
			output, (hidden, c) = self.lstm(rnn_input_vec, (hidden, c))

			## Predict output position and line probability
			output = output.view(batch_size, self.hidden_size)
			# print(output.size())
			output = self.predict_linear1(F.relu(output))
			output = self.predict_linear2(F.relu(output))

			# print(output.size())
			point_pos_output[:, idx, :] = output[:, 0:config.max_point*config.n_dim]
			line_prob_output[:, idx, :] = output[:, config.max_point*config.n_dim:]
			# Replace RNN output Vector
			previous_output = output

		# print(point_prob_output.size(), point_pos_output.size())
		line_prob_output = line_prob_output.view(-1, 2)
		line_prob_output = F.softmax(line_prob_output)
		return point_pos_output, line_prob_output

	def init_hidden(self, batch_size):
		hidden = Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()
		pred = Variable(torch.zeros(1, batch_size, self.output_size),
		                requires_grad = False).cuda()
		c = Variable(torch.zeros(1, batch_size, self.hidden_size)).cuda()
		return hidden, pred, c


class Multi_Line_Loss(nn.Module):
	def __init__(self):
		super(Multi_Line_Loss, self).__init__()
		self.clf_criterion = nn.CrossEntropyLoss(size_average = False)
	def forward(self, pos_pred, prob_pred, target_pos, line_mask, clf_line):
		'''

		:param pos_pred:
		:param prob_pred:
		:param target_pos:
		:param point_mask:
		:return:
		'''
		# if config.debugging:
		# 	print(pos_pred.cpu())
		# 	print(prob_pred.cpu())
		# 	print(target_pos.cpu())
		# 	print(line_mask.cpu())
		# 	print(clf_line.cpu())
		target_pos = target_pos.view(-1, config.max_line, config.max_point*config.n_dim)
		target_pos = match_lines_sequence(pos_pred, target_pos, line_mask)
		# print(type(target_pos))
		# print(target_pos.cpu())
		square_mse = torch.sum(torch.pow(pos_pred - target_pos, 2), dim=2)*line_mask
		# if config.debugging: print(square_mse.cpu())
		mask_mse = torch.sum(square_mse) / (torch.sum(line_mask))
		# print(square_mse.size())

		#Classification loss
		clf_point = clf_line.view(-1, 1)
		# if config.debugging: print(clf_point)
		lg_sftmx = -torch.log(prob_pred)
		# if config.debugging: print(lg_sftmx)
		clf_loss = torch.gather(lg_sftmx, 1, clf_point) * line_mask.view(-1,1)
		# if config.debugging: print(clf_loss)
		clf_loss = torch.sum(clf_loss)
		clf_loss = clf_loss / (torch.sum(line_mask))
		# clf_loss = lg_sftmx.select

		return mask_mse, clf_loss

class Multi_Line_Loss_Attention(nn.Module):
	def __init__(self):
		super(Multi_Line_Loss_Attention, self).__init__()
		self.clf_criterion = nn.CrossEntropyLoss(size_average = False)
	def forward(self, pos_pred, prob_pred, target_pos, line_mask, clf_line):
		'''

		:param pos_pred:
		:param prob_pred:
		:param target_pos:
		:param point_mask:
		:return:
		'''
		# if config.debugging:
		# 	print(pos_pred.cpu())
		# 	print(prob_pred.cpu())
		# 	print(target_pos.cpu())
		# 	print(line_mask.cpu())
		# 	print(clf_line.cpu())
		target_pos = target_pos.view(-1, config.max_line, config.max_point*config.n_dim)
		target_pos = match_lines(pos_pred, target_pos, line_mask)
		# print(type(target_pos))
		# print(target_pos.cpu())
		square_mse = torch.sum(torch.pow(pos_pred - target_pos, 2), dim=2)*line_mask
		# if config.debugging: print(square_mse.cpu())
		mask_mse = torch.sum(square_mse) / (torch.sum(line_mask))
		# print(square_mse.size())

		#Classification loss
		clf_point = clf_line.view(-1, 1)
		# if config.debugging: print(clf_point)
		lg_sftmx = -torch.log(prob_pred)
		# if config.debugging: print(lg_sftmx)
		clf_loss = torch.gather(lg_sftmx, 1, clf_point) * line_mask.view(-1,1)
		# if config.debugging: print(clf_loss)
		clf_loss = torch.sum(clf_loss)
		clf_loss = clf_loss / (torch.sum(line_mask))
		# clf_loss = lg_sftmx.select

		return mask_mse, clf_loss


def match_lines(pos_pred, target_pos, line_mask):
	"""
	Match the prediction line and target line, and return the matched ground truth.
	Using Hungarian Algorithm to optimize it
	:param pos_pred:
	:param target_pos:
	:param line_mask:
	:return:

	"""
	batch_size = pos_pred.size(0)
	pos_t = torch.zeros(batch_size, config.max_line, config.max_point*config.n_dim).cuda()

	for i_batch in range(batch_size):
		i_n_line = int(line_mask[i_batch].sum().data[0])
		if i_n_line == 1:
			pos_t[i_batch] = target_pos[i_batch].data
			continue
		if config.debugging: print(i_n_line)
		i_pos_pred = pos_pred[i_batch].data.cpu()[:int(i_n_line), :]
		i_pos_target = target_pos[i_batch].data.cpu()[:int(i_n_line), :]
		if config.debugging:
			print(i_pos_pred)
			print(i_pos_target)
			print(line_mask[i_batch])
		i_pos_pred = i_pos_pred.view(1, i_n_line, config.max_point*config.n_dim).expand(i_n_line, i_n_line, config.max_point*config.n_dim)
		i_pos_target = i_pos_target.view(i_n_line, 1, config.max_point*config.n_dim).expand(i_n_line, i_n_line, config.max_point*config.n_dim)
		Euclidean_m = i_pos_target - i_pos_pred
		Euclidean_m = torch.sum(Euclidean_m*Euclidean_m, 2).squeeze().t().numpy()
		if config.debugging:
			print(Euclidean_m)
		row_ind, col_ind = linear_sum_assignment(Euclidean_m)
		# In this case, row_ind is the same as np.arrange(Eiclidean_m.shape(0))
		if config.debugging: print(row_ind, col_ind)
		col_ind = torch.from_numpy(col_ind).cuda().long()
		tmp_target = target_pos[i_batch][col_ind]
		pos_t[i_batch, :i_n_line, :] = tmp_target.data
		if config.debugging:
			print(f"After, pos target is {pos_t}")

	return Variable(pos_t, requires_grad = False)

def match_lines_sequence(pos_pred, target_pos, line_mask):
	"""
	Match the prediction line and target line, and return the matched ground truth.
	Using Hungarian Algorithm to optimize it
	:param pos_pred:
	:param target_pos:
	:param line_mask:
	:return:

	"""
	batch_size = pos_pred.size(0)
	pos_t = torch.zeros(batch_size, config.max_line, config.max_point*config.n_dim).cuda()

	for i_batch in range(batch_size):
		i_n_line = int(line_mask[i_batch].sum().data[0])
		if i_n_line == 1:
			pos_t[i_batch] = target_pos[i_batch].data
			continue
		if config.debugging: print(i_n_line)
		i_pos_pred = pos_pred[i_batch].data.cpu()[:int(i_n_line), :]
		i_pos_target = target_pos[i_batch].data.cpu()[:int(i_n_line), :]
		if config.debugging:
			print(i_pos_pred)
			print(i_pos_target)
			print(line_mask[i_batch])
		i_pos_pred = i_pos_pred.view(1, i_n_line, config.max_point*config.n_dim).expand(i_n_line, i_n_line, config.max_point*config.n_dim)
		i_pos_target = i_pos_target.view(i_n_line, 1, config.max_point*config.n_dim).expand(i_n_line, i_n_line, config.max_point*config.n_dim)
		Euclidean_m = i_pos_target - i_pos_pred
		Euclidean_m = torch.sum(Euclidean_m*Euclidean_m, 2).squeeze().t().numpy()
		if config.debugging:
			print(Euclidean_m)
		row_ind, col_ind = linear_sum_assignment(Euclidean_m)
		# In this case, row_ind is the same as np.arrange(Eiclidean_m.shape(0))
		if config.debugging: print(row_ind, col_ind)
		col_ind = torch.from_numpy(col_ind).cuda().long()
		tmp_target = target_pos[i_batch][col_ind]
		pos_t[i_batch, :i_n_line, :] = tmp_target.data
		if config.debugging:
			print(f"After, pos target is {pos_t}")

	return Variable(pos_t, requires_grad = False)

